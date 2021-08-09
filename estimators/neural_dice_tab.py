# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeuralDICE with tabular policy parameterization
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf
from tf_agents.specs import tensor_spec
from tf_agents.policies import tf_policy
from tf_agents.utils import common as tfagents_common
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import dice_rl.data.dataset as dataset_lib
import dice_rl.utils.common as common_lib
import dice_rl.estimators.estimator as estimator_lib

def softmax(x):
  e_x = np.exp(x)
  # s = np.sum(e_x)
  s = np.sum(e_x, axis=1)
  return e_x / s.reshape((-1,1))

def normalize(x):
  s = np.sum(x, axis=1)
  r = x / s.reshape((-1,1))
  return x / s.reshape((-1,1))

def get_tabular_obs(s):
  return s[:,0]*10 + s[:,1]


class NeuralDice(object):
  """Policy evaluation with DICE."""

  def __init__(
      self,
      dataset_spec,
      nu_network,
      zeta_network,
      theta_network,
      nu_optimizer,
      zeta_optimizer,
      lam_optimizer,
      theta_learning_rate,
      dataset,
      gamma: Union[float, tf.Tensor],
      zero_reward=False,
      reward_fn: Callable = None,
      solve_for_state_action_ratio: bool = True,
      f_exponent: float = 1.5,
      primal_form: bool = False,
      num_samples: Optional[int] = None,
      primal_regularizer: float = 0.,
      dual_regularizer: float = 1.,
      norm_regularizer: bool = False,
      nu_regularizer: float = 0.,
      zeta_regularizer: float = 0.,
      weight_by_gamma: bool = False,
  ):
    """Initializes the solver.
    Args:
      dataset_spec: The spec of the dataset that will be given.
      nu_network: The nu-value network.
      zeta_network: The zeta-value network.
      nu_optimizer: The optimizer to use for nu.
      zeta_optimizer: The optimizer to use for zeta.
      lam_optimizer: The optimizer to use for lambda.
      gamma: The discount factor to use.
      zero_reward: Not including the reward in computing the residual.
      reward_fn: A function that takes in an EnvStep and returns the reward for
        that step. If not specified, defaults to just EnvStep.reward.
      solve_for_state_action_ratio: Whether to solve for state-action density
        ratio. Defaults to True.
      f_exponent: Exponent p to use for f(x) = |x|^p / p.
      primal_form: Whether to use primal form of DualDICE, which optimizes for
        nu independent of zeta. This form is biased in stochastic environments.
        Defaults to False, which uses the saddle-point formulation of DualDICE.
      num_samples: Number of samples to take from policy to estimate average
        next nu value. If actions are discrete, this defaults to computing
        average explicitly. If actions are not discrete, this defaults to using
        a single sample.
      primal_regularizer: Weight of primal varibale regularizer.
      dual_regularizer: Weight of dual varibale regularizer.
      norm_regularizer: Weight of normalization constraint.
      nu_regularizer: Regularization coefficient on nu network.
      zeta_regularizer: Regularization coefficient on zeta network.
      weight_by_gamma: Weight nu and zeta losses by gamma ** step_num.
    """
    self._dataset_spec = dataset_spec
    self._nu_network = nu_network
    self._nu_network.create_variables()
    self._zeta_network = zeta_network
    self._zeta_network.create_variables()
    self._zero_reward = zero_reward

    self._nu_optimizer = nu_optimizer
    self._zeta_optimizer = zeta_optimizer
    self._lam_optimizer = lam_optimizer
    self._nu_regularizer = nu_regularizer
    self._zeta_regularizer = zeta_regularizer
    self._weight_by_gamma = weight_by_gamma

    self._theta_network = theta_network
    self._policy = softmax(self._theta_network)
    self._theta_lr = theta_learning_rate

    self._dataset = dataset
    self._dD = self.get_dD()

    # self._nu_cf = np.load('/home/snowfly/Documents/dice_rl/results/ope/nu_0.npy')
    # self._zeta_cf = np.load('/home/snowfly/Documents/dice_rl/results/ope/zeta_0.npy')
    # self._nu_values = self.get_tabular_value(self._nu_cf)
    # self._nu_values = self._nu_values / np.linalg.norm(self._nu_values)
    # self._zeta_values = self.get_tabular_value(self._zeta_cf)
    # self._zeta_values = self._nu_values / np.linalg.norm(self._zeta_values)

    self._gamma = gamma
    if reward_fn is None:
      reward_fn = lambda env_step: env_step.reward
    self._reward_fn = reward_fn
    self._num_samples = num_samples

    self._solve_for_state_action_ratio = solve_for_state_action_ratio
    if (not self._solve_for_state_action_ratio and
        not self._dataset_spec.has_log_probability()):
      raise ValueError('Dataset must contain log-probability when '
                       'solve_for_state_action_ratio is False.')

    if f_exponent <= 1:
      raise ValueError('Exponent for f must be greater than 1.')
    fstar_exponent = f_exponent / (f_exponent - 1)
    self._f_fn = lambda x: tf.abs(x)**f_exponent / f_exponent
    self._fstar_fn = lambda x: tf.abs(x)**fstar_exponent / fstar_exponent

    self._categorical_action = common_lib.is_categorical_spec(
        self._dataset_spec.action)
    if not self._categorical_action and self._num_samples is None:
      self._num_samples = 1

    self._primal_form = primal_form
    self._primal_regularizer = primal_regularizer
    self._dual_regularizer = dual_regularizer
    self._norm_regularizer = norm_regularizer
    self._lam = tf.Variable(0.0)
    self._initialize()

  def _initialize(self):
    pass

  def get_policy_distribution(self, s):
    return self._policy[s]
  
  def get_tabular_value(self, tab_net):
    steps = self._dataset.get_all_steps(limit=None)
    s = steps.observation.numpy()
    a = steps.action.numpy()
    values = np.zeros(s.shape[0])
    for i in range(s.shape[0]):
      values[i] = tab_net[s[i]*4 + a[i]]
    return values

  def get_dD(self):
    steps = self._dataset.get_all_steps(limit=None)
    s = steps.observation.numpy()
    a = steps.action.numpy()
    freq = np.zeros(self._theta_network.shape)
    for idx in range(s.shape[0]):
      freq[s[idx], a[idx]] += 1

    freq *= (1/s.shape[0])
    return freq

  def _get_full_gradient(self, zeta_norm):    
    grad = np.zeros(self._theta_network.shape)
    for i in range(grad.shape[0]):
      si = tf.ones((4,), dtype=tf.dtypes.int64)*i
      ai = tf.convert_to_tensor(np.array(range(4)).reshape((-1,)), dtype=tf.dtypes.int64)
      nu = self._nu_network((si, ai))[0].numpy()
      weights = self._theta_network[i,:]
      weights = np.exp(weights)
      weights = weights / np.sum(weights)
      v = np.dot(nu, weights)
      for j in range(grad.shape[1]):
        zeta = self._zeta_network((tf.convert_to_tensor([i], dtype=tf.dtypes.int64), tf.convert_to_tensor([j], dtype=tf.dtypes.int64)))[0].numpy()
        grad[i,j] = self._dD[i,j] * zeta / zeta_norm * (nu[j] - v) 
    grad /= (1-self._gamma)
    return grad



  def _get_value(self, network, env_step):
    if self._solve_for_state_action_ratio:
      return network((env_step.observation, env_step.action))[0]
    else:
      return network((env_step.observation,))[0]

  def _get_average_value1(self, network, env_step):
    if self._solve_for_state_action_ratio:
      tfagents_step = dataset_lib.convert_to_tfagents_timestep(env_step)
      if self._categorical_action and self._num_samples is None:

        s = env_step.observation
        s = tf.one_hot(s, depth = self._theta_network.shape[0])

        action_weights = tf.matmul(s, self._policy)

        action_dtype = self._dataset_spec.action.dtype
        batch_size = action_weights.shape[0]
        num_actions = action_weights.shape[-1]
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
      return tf.reduce_sum(values * action_weights, axis=1)
    else:
      return network((env_step.observation,))[0]
  

  def _get_average_value(self, network, env_step):
    if self._solve_for_state_action_ratio:
      tfagents_step = dataset_lib.convert_to_tfagents_timestep(env_step)
      if self._categorical_action and self._num_samples is None:

        s = env_step.observation
        s = tf.one_hot(s, depth = self._theta_network.shape[0]).numpy()

        action_weights = np.dot(s, self._policy)

        action_dtype = self._dataset_spec.action.dtype
        batch_size = action_weights.shape[0]
        num_actions = action_weights.shape[-1]
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
      return tf.reduce_sum(values * action_weights, axis=1)
    else:
      return network((env_step.observation,))[0]

  def _orthogonal_regularization(self, network):
    reg = 0
    for layer in network.layers:
      if isinstance(layer, tf.keras.layers.Dense):
        prod = tf.matmul(tf.transpose(layer.kernel), layer.kernel)
        reg += tf.reduce_sum(tf.math.square(prod * (1 - tf.eye(prod.shape[0]))))
    return reg


  def train_loss(self, initial_env_step, env_step, next_env_step):
    nu_values = self._get_value(self._nu_network, env_step)
    initial_nu_values = self._get_average_value1(self._nu_network,
                                                initial_env_step)
    next_nu_values = self._get_average_value1(self._nu_network, next_env_step)

    zeta_values = self._get_value(self._zeta_network, env_step)

    # zeta_reg = zeta_values / zeta_norm

    # init_log_prob = self._get_initial_log_prob(initial_env_step, initial_nu_values.numpy())
    # next_log_prob = self._get_next_log_prob(next_env_step, next_nu_values.numpy(), zeta_reg.numpy())
    # theta_grad = (1-self._gamma) * init_log_prob + self._gamma * next_log_prob

    discounts = self._gamma * next_env_step.discount
    policy_ratio = 1.0
    if not self._solve_for_state_action_ratio:
      tfagents_step = dataset_lib.convert_to_tfagents_timestep(env_step)
      policy_log_probabilities = policy.distribution(
          tfagents_step).action.log_prob(env_step.action)
      policy_ratio = tf.exp(policy_log_probabilities -
                            env_step.get_log_probability())

    bellman_residuals = (
        common_lib.reverse_broadcast(discounts * policy_ratio, nu_values) *
        next_nu_values - nu_values - self._norm_regularizer * self._lam)
    if not self._zero_reward:
      bellman_residuals += policy_ratio * self._reward_fn(env_step)

    zeta_loss = -zeta_values * bellman_residuals
    nu_loss = (1 - self._gamma) * initial_nu_values
    lam_loss = self._norm_regularizer * self._lam
    if self._primal_form:
      nu_loss += self._fstar_fn(bellman_residuals)
      lam_loss = lam_loss + self._fstar_fn(bellman_residuals)
    else:
      nu_loss += zeta_values * bellman_residuals
      lam_loss = lam_loss - self._norm_regularizer * zeta_values * self._lam

    nu_loss += self._primal_regularizer * self._f_fn(nu_values)
    zeta_loss += self._dual_regularizer * self._f_fn(zeta_values)

    if self._weight_by_gamma:
      weights = self._gamma**tf.cast(env_step.step_num, tf.float32)[:, None]
      weights /= 1e-6 + tf.reduce_mean(weights)
      nu_loss *= weights
      zeta_loss *= weights

    return nu_loss, zeta_loss, lam_loss

  @tf.function
  def train_step(self, initial_env_step: dataset_lib.EnvStep,
                 experience: dataset_lib.EnvStep):
    """Performs a single training step based on batch.
    Args:
      initial_env_step: A batch of initial steps.
      experience: A batch of transitions. Elements must have shape [batch_size,
        2, ...].
      target_policy: The policy whose value we want to estimate.
    Returns:
      The losses and the train op.
    """
    env_step = tf.nest.map_structure(lambda t: t[:, 0, ...], experience)
    next_env_step = tf.nest.map_structure(lambda t: t[:, 1, ...], experience)

    with tf.GradientTape(
        watch_accessed_variables=False, persistent=True) as tape:
      tape.watch(self._nu_network.variables)
      tape.watch(self._zeta_network.variables)
      tape.watch([self._lam])
      nu_loss, zeta_loss, lam_loss, = self.train_loss(initial_env_step, env_step,
                                                     next_env_step)
      nu_loss += self._nu_regularizer * self._orthogonal_regularization(
          self._nu_network)
      zeta_loss += self._zeta_regularizer * self._orthogonal_regularization(
          self._zeta_network)

    nu_grads = tape.gradient(nu_loss, self._nu_network.variables)
    nu_grad_op = self._nu_optimizer.apply_gradients(
        zip(nu_grads, self._nu_network.variables))

    zeta_grads = tape.gradient(zeta_loss, self._zeta_network.variables)
    zeta_grad_op = self._zeta_optimizer.apply_gradients(
        zip(zeta_grads, self._zeta_network.variables))

    lam_grads = tape.gradient(lam_loss, [self._lam])
    lam_grad_op = self._lam_optimizer.apply_gradients(
        zip(lam_grads, [self._lam]))



    return (tf.reduce_mean(nu_loss), tf.reduce_mean(zeta_loss),
            tf.reduce_mean(lam_loss))

  def update_pg(self):
    steps = self._dataset.get_all_steps(limit=None)
    zeta_norm = self.weight_fn(steps)
    zeta_norm = tf.reduce_mean(zeta_norm)
    theta_grad = self._get_full_gradient(zeta_norm)
    self._theta_network += self._theta_lr * theta_grad
    self._policy = softmax(self._theta_network)

  def weight_fn(self, env_step):
    zeta = self._get_value(self._zeta_network, env_step)
    policy_ratio = 1.0
    # if not self._solve_for_state_action_ratio:
    #   tfagents_timestep = dataset_lib.convert_to_tfagents_timestep(env_step)
    #   target_log_probabilities = target_policy.distribution(
    #       tfagents_timestep).action.log_prob(env_step.action)
    #   policy_ratio = tf.exp(target_log_probabilities -
    #                         env_step.get_log_probability())
    return zeta * common_lib.reverse_broadcast(policy_ratio, zeta)

  def estimate_average_reward(self, step):
    """Estimates value (average per-step reward) of policy.
    Args:
      dataset: The dataset to sample experience from.
      target_policy: The policy whose value we want to estimate.
    Returns:
      Estimated average per-step reward of the target policy.
    """

    def init_nu_fn(env_step, valid_steps):
      """Computes average initial nu values of episodes."""
      # env_step is an episode, and we just want the first step.
      if tf.rank(valid_steps) == 1:
        first_step = tf.nest.map_structure(lambda t: t[0, ...], env_step)
      else:
        first_step = tf.nest.map_structure(lambda t: t[:, 0, ...], env_step)
      value = self._get_average_value(self._nu_network, first_step)
      return value

    nu_zero = (1 - self._gamma) * estimator_lib.get_fullbatch_average(
        self._dataset,
        limit=None,
        by_steps=False,
        truncate_episode_at=1,
        reward_fn=init_nu_fn)

    dual_step = estimator_lib.get_fullbatch_average(
        self._dataset,
        limit=None,
        by_steps=True,
        reward_fn=self._reward_fn,
        weight_fn=self.weight_fn)

    tf.summary.scalar('nu_zero', nu_zero)
    tf.summary.scalar('lam', self._norm_regularizer * self._lam)
    tf.summary.scalar('dual_step', dual_step)

    
    tf.print('step', step, 'nu_zero =', nu_zero,
             'lam =', self._norm_regularizer * self._lam, 'dual_step =',
             dual_step)

    # steps = self._dataset.get_all_steps(limit=None)
    # nu_pred = self._get_value(self._nu_network, steps).numpy()
    # nu_pred = self._nu_values / np.linalg.norm(nu_pred)
    # zeta_pred = self._get_value(self._zeta_network, steps).numpy()
    # zeta_pred = self._nu_values / np.linalg.norm(zeta_pred)

    # print("nu error: ", np.linalg.norm(nu_pred - self._nu_values))
    # print("zeta error: ", np.linalg.norm(zeta_pred - self._zeta_values))

    return dual_step, nu_zero
  
  def get_nu_parameters(self):
    return self._nu_network.variables

  def get_zeta_parameters(self):
    return self._zeta_network.variables

  def get_theta_parameters(self):
    return self._theta_network

  def update(self, nu, zeta, theta):
    tf.nest.map_structure(lambda var, value: var.assign(value), self._nu_network.variables, nu)
    tf.nest.map_structure(lambda var, value: var.assign(value), self._zeta_network.variables, zeta)
    # self._nu_network.assign(nu)
    # self._zeta_network.assign(zeta)
    # n2=self._nu_network.variables[0].numpy()
    self._theta_network = theta
    self._policy = softmax(self._theta_network)
