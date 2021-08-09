"""
Policy gradient with neural policy parameterizations
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import numpy as np
import tf_agents
import tensorflow.compat.v2 as tf
from tf_agents.specs import tensor_spec
from tf_agents.policies import tf_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common as tfagents_common
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import dice_rl.data.dataset as dataset_lib
import dice_rl.utils.common as common_lib
import dice_rl.estimators.estimator as estimator_lib



from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class NeuralPgDice(object):
  """Policy optimization with DICE."""

  def __init__(
      self,
      dataset_spec,
      policy_network,
      target_policy,
      nu_network,
      zeta_network,
      policy_optimizer,
      gamma: Union[float, tf.Tensor],
      num_samples: Optional[int] = None,
  ):
    """Initializes the solver.

    Args:
      dataset_spec: The spec of the dataset that will be given.
      policy_netowrk: A tensorflow policy network.
      policy_optimizer: An optimizer compatible with tensorflow.
      gamma: The discount factor to use.
      num_samples: Number of samples to take from policy to estimate average
        next nu value. If actions are discrete, this defaults to computing
        average explicitly. If actions are not discrete, this defaults to using
        a single sample.
      weight_by_gamma: Weight nu and zeta losses by gamma ** step_num.
    """
    self._zeta_normalizer = 1
    self._dataset_spec = dataset_spec
    self._policy_network = policy_network
    self._nu_network = nu_network
    self._zeta_network = zeta_network
    self._policy_network.create_variables()
    self._policy_optimizer = policy_optimizer
    self._policy = target_policy

    self._gamma = gamma
    self._num_samples = num_samples
    reward_fn = lambda env_step: env_step.reward
    self._reward_fn = reward_fn

    self._categorical_action = common_lib.is_categorical_spec(
        self._dataset_spec.action)
    if self._categorical_action:
        self._action_dim = self._dataset_spec.action.maximum - \
                          self._dataset_spec.action.minimum + 1
    if not self._categorical_action and self._num_samples is None:
      self._num_samples = 1

    self._initialize()
    
  def _initialize(self):
    pass

  def train_loss(self, initial_env_step, env_step, next_env_step):
    initial_gae = self._get_weighted_logprobs(self._nu_network,
                                          initial_env_step)
    next_gae = self._get_weighted_logprobs(self._nu_network,
                                       next_env_step)
    
    dist_correction = self._get_value(self._zeta_network, env_step)
    dist_correction /= self._zeta_normalizer
    
    policy_loss = -(1-self._gamma) * initial_gae - \
                  self._gamma * dist_correction * next_gae
    
    return policy_loss

  """Update theta network"""
  def get_policy_network(self,):
    return self._policy_network
  
  def update_policy(self, target_policy):
    self._policy = target_policy

  def get_parameters(self,):
    return self._policy_network.variables

  def update(self, theta):
    tf.nest.map_structure(lambda var, value: var.assign(value), self._policy_network.variables, theta)

  @tf.function
  def train_step(self, initial_env_step: dataset_lib.EnvStep,
                 experience: dataset_lib.EnvStep):
    """Performs a single training step based on batch.

    Args:
      initial_env_step: A batch of initial steps.
      experience: A batch of transitions. Elements must have shape [batch_size,
        2, ...].
      policy: The policy to optimize.

    Returns:
      The policy losses and the train op.
    """
    env_step = tf.nest.map_structure(lambda t: t[:, 0, ...], experience)
    next_env_step = tf.nest.map_structure(lambda t: t[:, 1, ...], experience)

    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      tape.watch(self._policy_network.variables)
      policy_loss = self.train_loss(initial_env_step, env_step, 
                                    next_env_step)

    policy_grads = tape.gradient(policy_loss, self._policy_network.variables)
    policy_grad_op = self._policy_optimizer.apply_gradients(
        zip(policy_grads, self._policy_network.variables))

    return tf.reduce_mean(policy_loss)

  def _get_value(self, network, env_step):
    return network((env_step.observation, env_step.action))[0]

  def _get_policy_averaged_value(self, network, env_step):
    tf_dist, _ = self._policy_network(env_step.observation)
    action_weights = tf_dist.probs_parameter()
    action_dtype = self._dataset_spec.action.dtype
    batch_size = tf.shape(action_weights)[0]
    
    num_actions = tf.shape(action_weights)[-1]
    actions = (  # Broadcast actions
            tf.ones([batch_size, 1], dtype=action_dtype) *
            tf.range(num_actions, dtype=action_dtype)[None, :])
    flat_actions = tf.reshape(actions, [batch_size * num_actions] +
                              actions.shape[2:].as_list())
    flat_observations = tf.reshape(
        tf.tile(env_step.observation[:, None, ...],
                [1, num_actions] + [1] * len(env_step.observation.shape[1:])),
        [batch_size * num_actions] + env_step.observation.shape[1:].as_list())
    flat_q_values, _ = network((flat_observations, flat_actions))
    
    q_values = tf.reshape(flat_q_values, [batch_size, num_actions]
                          + flat_q_values.shape[1:].as_list())
    return tf.stop_gradient(tf.reduce_sum(
                            q_values * action_weights, axis=1))

  def _get_weighted_logprobs(self, network, env_step):
    '''
     Gives the batched log probabilities weighted by batched values.
    '''
    tfagents_step = dataset_lib.convert_to_tfagents_timestep(env_step)
    if self._categorical_action and self._num_samples is None:
        action_weights = self._policy.distribution(
            tfagents_step).action.probs_parameter()
        action_dtype = self._dataset_spec.action.dtype
        batch_size = tf.shape(action_weights)[0]
        num_actions = tf.shape(action_weights)[-1]
        actions = (  # Broadcast actions
            tf.ones([batch_size, 1], dtype=action_dtype) *
            tf.range(num_actions, dtype=action_dtype)[None, :])
    else:  
        batch_size = tf.shape(env_step.observation)[0]
        num_actions = self._num_samples
        action_weights = tf.ones([batch_size, num_actions]) / num_actions
        actions = tf.stack(
            [self._policy.action(tfagents_step).action for _ in range(num_actions)],
            axis=1)

    flat_actions = tf.reshape(actions, [batch_size * num_actions] +
                              actions.shape[2:].as_list())
    flat_observations = tf.reshape(
        tf.tile(env_step.observation[:, None, ...],
                [1, num_actions] + [1] * len(env_step.observation.shape[1:])),
        [batch_size * num_actions] + env_step.observation.shape[1:].as_list())   
    flat_values, _ = network((flat_observations, flat_actions))
    
    if self._categorical_action:
        # alternative code
        tf_dist, _  = self._policy_network(flat_observations)
        log_probs = tfagents_common.log_probability(tf_dist, flat_actions, 
                                                    self._dataset_spec.action)
        
        # tf_dist, _  = self._policy_network(flat_observations, training=True)
        # logits = tf_dist.logits_parameter()
        # onehot_actions = tf.one_hot(flat_actions, self._action_dim)
        # # stable in log prob
        # log_probs = tf.nn.softmax_cross_entropy_with_logits(onehot_actions, 
        #                                                       logits)
    else:
        tf_dist, _  = self._policy_network(flat_observations)
        log_probs = tfagents_common.log_probability(tf_dist, flat_actions, 
                                                    self._dataset_spec.action)

    flat_log_probs = tf.reshape(log_probs, [batch_size * num_actions] +
                              log_probs.shape[2:].as_list()) 
    flat_valued_logprobs = flat_values * flat_log_probs
    
    valued_logprobs = tf.reshape(flat_valued_logprobs, [batch_size, num_actions]
                                 + flat_valued_logprobs.shape[1:].as_list())
    return tf.reduce_sum(
        valued_logprobs * common_lib.reverse_broadcast(
                          action_weights, valued_logprobs), axis=1)

  def get_zeta_normalizer(self, dataset):
      all_env_steps = dataset.get_all_steps(limit=None)
      zeta_values = self._get_value(self._zeta_network, all_env_steps)
      self._zeta_normalizer = tf.reduce_mean(zeta_values, axis=0)
      return self._zeta_normalizer

  def evaluate_policy(self, env, num_traj, traj_len):
      ''' 
      Evaluate the policy online.
      Args:
          env: A gym environment.
          num_traj: Total number of trajectories to collect.
          traj_len: Length of per trajectory collected.
          policy: A TFPolicy used to interact with the env.
      return:
          traj_avg_r: Average reward of policy.
      '''
      traj_avg_r = 0
      for traj in range(num_traj):
          obs = np.array([env.reset()])
          avg_r = 0
          for step in range(traj_len):
              tfagents_step = ts.restart(obs)
              action = self._policy.action(tfagents_step).action
            #   print(action.numpy()[0])
              obs, r, done, _ = env.step(action.numpy()[0])
              obs = obs.reshape((1, -1))
              if done:
                  raise Exception('Only accept infinite horizon env.')
              avg_r += r
          traj_avg_r += avg_r/traj_len
      traj_avg_r /= num_traj
      tf.summary.scalar('avg_reward', traj_avg_r)
      tf.print('step', tf.summary.experimental.get_step(), 
               'Avg reward =', traj_avg_r)
      
      return traj_avg_r
  
  def fullbatch_train_step(self, dataset):
    """Performs a single training step based on full batch.

    Args:
      dataset: The off-policy dataset.

    Returns:
      The fullbatch policy losses and the train op.
    """
    steps = dataset.get_all_steps(limit=None)

    q_func = self._get_value(self._nu_network, steps)
    v_func = self._get_policy_averaged_value(self._nu_network, steps)
    
    dist_correction = self._get_value(self._zeta_network, steps)
    dist_correction /= self._zeta_normalizer
    
    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      tape.watch(self._policy_network.variables)
      tf_dist, _  = self._policy_network(steps.observation)
      log_probs = tfagents_common.log_probability(tf_dist, steps.action, 
                                                self._dataset_spec.action)
      policy_loss = -dist_correction * (q_func-v_func) * log_probs

    policy_grads = tape.gradient(policy_loss, self._policy_network.variables)
    policy_grad_op = self._policy_optimizer.apply_gradients(
        zip(policy_grads, self._policy_network.variables))

    return tf.reduce_mean(policy_loss)
  
  
  ''' imported from NeuralDICE '''
  def estimate_average_reward(self, dataset: dataset_lib.OffpolicyDataset,
                              target_policy: tf_policy.TFPolicy):
    """Estimates value (average per-step reward) of policy.

    Args:
      dataset: The dataset to sample experience from.
      target_policy: The policy whose value we want to estimate.

    Returns:
      Estimated average per-step reward of the target policy.
    """

    def weight_fn(env_step):
      zeta = self._get_value(self._zeta_network, env_step)
      policy_ratio = 1.0
      return zeta * common_lib.reverse_broadcast(policy_ratio, zeta)

    def init_nu_fn(env_step, valid_steps):
      """Computes average initial nu values of episodes."""
      # env_step is an episode, and we just want the first step.
      if tf.rank(valid_steps) == 1:
        first_step = tf.nest.map_structure(lambda t: t[0, ...], env_step)
      else:
        first_step = tf.nest.map_structure(lambda t: t[:, 0, ...], env_step)
      value = self._get_average_value(self._nu_network, first_step,
                                      target_policy)
      return value

    nu_zero = (1 - self._gamma) * estimator_lib.get_fullbatch_average(
        dataset,
        limit=None,
        by_steps=False,
        truncate_episode_at=1,
        reward_fn=init_nu_fn)

    dual_step = estimator_lib.get_fullbatch_average(
        dataset,
        limit=None,
        by_steps=True,
        reward_fn=self._reward_fn,
        weight_fn=weight_fn)

    tf.summary.scalar('nu_zero', nu_zero)
    tf.summary.scalar('dual_step', dual_step)
    tf.print('step', tf.summary.experimental.get_step(), 'nu_zero =', nu_zero,
             'dual_step =', dual_step)

    return dual_step

  def _get_average_value(self, network, env_step, policy):
    tfagents_step = dataset_lib.convert_to_tfagents_timestep(env_step)
    if self._categorical_action and self._num_samples is None:
      action_weights = policy.distribution(
          tfagents_step).action.probs_parameter()
      action_dtype = self._dataset_spec.action.dtype
      batch_size = tf.shape(action_weights)[0]
      num_actions = tf.shape(action_weights)[-1]
      actions = (  # Broadcast actions
          tf.ones([batch_size, 1], dtype=action_dtype) *
          tf.range(num_actions, dtype=action_dtype)[None, :])
    else:
      batch_size = tf.shape(env_step.observation)[0]
      num_actions = self._num_samples
      action_weights = tf.ones([batch_size, num_actions]) / num_actions
      actions = tf.stack(
          [policy.action(tfagents_step).action for _ in range(num_actions)],
          axis=1)

    flat_actions = tf.reshape(actions, [batch_size * num_actions] +
                              actions.shape[2:].as_list())
    flat_observations = tf.reshape(
        tf.tile(env_step.observation[:, None, ...],
                [1, num_actions] + [1] * len(env_step.observation.shape[1:])),
        [batch_size * num_actions] + env_step.observation.shape[1:].as_list())

    flat_values, _ = network((flat_observations, flat_actions))
    values = tf.reshape(flat_values, [batch_size, num_actions] +
                        flat_values.shape[1:].as_list())
    return tf.reduce_sum(
        values * common_lib.reverse_broadcast(action_weights, values), axis=1)
    

   
  def get_policy_network(self):
    return self._policy_network

  def get_policy_parameters(self):
    return self._policy_network.variables

  def update(self,theta):
    tf.nest.map_structure(lambda var, value: var.assign(value), self._policy_network.variables, theta)