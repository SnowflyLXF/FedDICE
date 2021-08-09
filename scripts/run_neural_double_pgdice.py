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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
import os
import sys
import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()
import pickle

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import actor_policy

from dice_rl.environments.env_policies import get_target_policy
import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.tree as tree
import dice_rl.environments.gridworld.taxi as taxi
from dice_rl.estimators.neural_dice import NeuralDice
from dice_rl.optimizers.neural_pgdice import NeuralPgDice
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
from dice_rl.networks.policy_network import PolicyNetwork
from dice_rl.networks.tabular_policy_network import TabularSoftmaxPolicyNetwork
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
'''
TODO:clean up import once done debugging
'''
import dice_rl.utils.common as common_lib
from dice_rl.environments.infinite_cartpole import InfiniteCartPole


# BEGIN GOOGLE-INTERNAL
# import google3.learning.deepmind.xmanager2.client.google as xm
# END GOOGLE-INTERNAL
'''
Remove this log blocker once retracing problem is solved
'''
# import logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# logging.getLogger('tensorflow').setLevel(logging.FATAL)
     
 
FLAGS = flags.FLAGS

flags.DEFINE_string('load_dir', 'E:/dice/dice_rl/tests/testdata', 'Directory to load dataset from.')
flags.DEFINE_string('save_dir', None,
                    'Directory to save the model and estimation results.')
flags.DEFINE_string('env_name', 'grid', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_bool('tabular_obs', True, 'Whether to use tabular observations.')
flags.DEFINE_integer('num_trajectory', 400,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 100,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.0, 'How close to target policy.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_float('nu_learning_rate', 0.00003, 'Learning rate for nu.')
flags.DEFINE_float('zeta_learning_rate', 0.00003, 'Learning rate for zeta.')
flags.DEFINE_float('policy_learning_rate', 0.000001, 'Learning rate for policy.')
flags.DEFINE_float('nu_regularizer', 0.0, 'Ortho regularization on nu.')
flags.DEFINE_float('zeta_regularizer', 0.0, 'Ortho regularization on zeta.')
flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 2048, 'Batch size.')

flags.DEFINE_float('f_exponent', 2, 'Exponent for f function.')
flags.DEFINE_bool('primal_form', False,
                  'Whether to use primal form of loss for nu.')

flags.DEFINE_float('primal_regularizer', 1.,
                   'LP regularizer of primal variables.')
flags.DEFINE_float('dual_regularizer', 1., 'LP regularizer of dual variables.')
flags.DEFINE_bool('zero_reward', False,
                  'Whether to ignore reward in optimization.')
flags.DEFINE_float('norm_regularizer', 1.,
                   'Weight of normalization constraint.')
flags.DEFINE_bool('zeta_pos', True, 'Whether to enforce positivity constraint.')

flags.DEFINE_float('scale_reward', 1., 'Reward scaling factor.')
flags.DEFINE_float('shift_reward', 0., 'Reward shift factor.')
flags.DEFINE_string(
    'transform_reward', None, 'Non-linear reward transformation'
    'One of [exp, cuberoot, None]')

flags.DEFINE_integer('eval_trajectory', 100,
                     'Number of trajectories in online policy evaluation.')
flags.DEFINE_integer('eval_trajectory_length', 100,
                     'Length of trajectory in online policy evaluation.')
flags.DEFINE_integer('opt_interval', 1,
                     'Policy is updated every opt_interval steps.')
flags.DEFINE_integer('eval_interval', 1000,
                     'Policy is evaluated every eval_interval steps.')
flags.DEFINE_integer('warmup_step', 0,
                     'Update policy after warmup_step steps.')
flags.DEFINE_float('double_dual_regularizer', 0, 
                   'LP regularizer of double dual variables.')


def main(argv):
  load_dir = FLAGS.load_dir
  save_dir = FLAGS.save_dir
  env_name = FLAGS.env_name
  seed = FLAGS.seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  gamma = FLAGS.gamma
  nu_learning_rate = FLAGS.nu_learning_rate
  zeta_learning_rate = FLAGS.zeta_learning_rate
  policy_learning_rate = FLAGS.policy_learning_rate
  nu_regularizer = FLAGS.nu_regularizer
  zeta_regularizer = FLAGS.zeta_regularizer
  num_steps = FLAGS.num_steps
  batch_size = FLAGS.batch_size

  f_exponent = FLAGS.f_exponent
  primal_form = FLAGS.primal_form

  primal_regularizer = FLAGS.primal_regularizer
  dual_regularizer = FLAGS.dual_regularizer
  zero_reward = FLAGS.zero_reward
  norm_regularizer = FLAGS.norm_regularizer
  zeta_pos = FLAGS.zeta_pos

  scale_reward = FLAGS.scale_reward
  shift_reward = FLAGS.shift_reward
  transform_reward = FLAGS.transform_reward
  
  num_traj = FLAGS.eval_trajectory
  traj_len = FLAGS.eval_trajectory_length
  opt_interval = FLAGS.opt_interval
  eval_interval = FLAGS.eval_interval
  warmup_step = FLAGS.warmup_step
  double_dual_regularizer = FLAGS.double_dual_regularizer

  def reward_fn(env_step):
    reward = env_step.reward * scale_reward + shift_reward
    if transform_reward is None:
      return reward
    if transform_reward == 'exp':
      reward = tf.math.exp(reward)
    elif transform_reward == 'cuberoot':
      reward = tf.sign(reward) * tf.math.pow(tf.abs(reward), 1.0 / 3.0)
    else:
      raise ValueError('Reward {} not implemented.'.format(transform_reward))
    return reward

  hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    TAB=tabular_obs,
                    ALPHA=alpha,
                    SEED=seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=max_trajectory_length)
  train_hparam_str = (
      'nlr{NLR}_zlr{ZLR}_zeror{ZEROR}_preg{PREG}_dreg{DREG}_nreg{NREG}_'
      'pform{PFORM}_fexp{FEXP}_zpos{ZPOS}_'
      'scaler{SCALER}_shiftr{SHIFTR}_transr{TRANSR}').format(
          NLR=nu_learning_rate,
          ZLR=zeta_learning_rate,
          ZEROR=zero_reward,
          PREG=primal_regularizer,
          DREG=dual_regularizer,
          NREG=norm_regularizer,
          PFORM=primal_form,
          FEXP=f_exponent,
          ZPOS=zeta_pos,
          SCALER=scale_reward,
          SHIFTR=shift_reward,
          TRANSR=transform_reward)
  if save_dir is not None:
    save_dir = os.path.join(save_dir, hparam_str, train_hparam_str)
    summary_writer = tf.summary.create_file_writer(logdir=save_dir)
    summary_writer.set_as_default()
  else:
    tf.summary.create_noop_writer()

  directory = os.path.join(load_dir, hparam_str)
  print('Loading dataset from', directory)
  dataset = Dataset.load(directory)
  all_steps = dataset.get_all_steps()
  max_reward = tf.reduce_max(all_steps.reward)
  min_reward = tf.reduce_min(all_steps.reward)
  print('num loaded steps', dataset.num_steps)
  print('num loaded total steps', dataset.num_total_steps)
  print('num loaded episodes', dataset.num_episodes)
  print('num loaded total episodes', dataset.num_total_episodes)
  print('min reward', min_reward, 'max reward', max_reward)
  print('behavior per-step',
        estimator_lib.get_fullbatch_average(dataset, gamma=gamma))
  target_dataset = Dataset.load(directory)
  # target_dataset = Dataset.load(
  #     directory.replace('alpha{}'.format(alpha), 'alpha1.0'))
  print('target per-step',
        estimator_lib.get_fullbatch_average(target_dataset, gamma=1.))

  activation_fn = tf.nn.relu
  kernel_initializer = tf.keras.initializers.GlorotUniform()
  hidden_dims = (64, 64)
  input_spec = (dataset.spec.observation, dataset.spec.action)
  output_spec = dataset.spec.action
  nu_network = ValueNetwork(
      input_spec,
      fc_layer_params=hidden_dims,
      activation_fn=activation_fn,
      kernel_initializer=kernel_initializer,
      last_kernel_initializer=kernel_initializer)
  double_nu_network = ValueNetwork(
      input_spec,
      fc_layer_params=hidden_dims,
      activation_fn=activation_fn,
      kernel_initializer=kernel_initializer,
      last_kernel_initializer=kernel_initializer)
  output_activation_fn = tf.math.square if zeta_pos else tf.identity
  zeta_network = ValueNetwork(
      input_spec,
      fc_layer_params=hidden_dims,
      activation_fn=activation_fn,
      output_activation_fn=output_activation_fn,
      kernel_initializer=kernel_initializer,
      last_kernel_initializer=kernel_initializer)
  double_zeta_network = ValueNetwork(
      input_spec,
      fc_layer_params=hidden_dims,
      activation_fn=activation_fn,
      output_activation_fn=output_activation_fn,
      kernel_initializer=kernel_initializer,
      last_kernel_initializer=kernel_initializer)


  if common_lib.is_categorical_spec(dataset.spec.observation) and \
     common_lib.is_categorical_spec(dataset.spec.action):
      init_dist = get_distribution_table(load_dir, env_name, alpha=alpha)
      init_dist = (1-4/5) * tf.ones(init_dist.shape)/init_dist.shape[1] + \
                   4/5 * init_dist
      policy_network = TabularSoftmaxPolicyNetwork(
          dataset.spec.observation,
          output_spec,
          initial_distribution=init_dist)
  else:
      policy_network = PolicyNetwork(
          dataset.spec.observation,
          output_spec)

  nu_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, clipvalue=1.0)
  double_nu_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, clipvalue=1.0)
  zeta_optimizer = tf.keras.optimizers.Adam(zeta_learning_rate, clipvalue=1.0)
  double_zeta_optimizer = tf.keras.optimizers.Adam(zeta_learning_rate, clipvalue=1.0)
  lam_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, clipvalue=1.0)
  policy_optimizer = tf.keras.optimizers.Adam(policy_learning_rate)

  nu_estimator = NeuralDice(
      dataset.spec,
      nu_network,
      double_zeta_network,
      nu_optimizer,
      double_zeta_optimizer,
      lam_optimizer,
      gamma,
      zero_reward=zero_reward,
      f_exponent=f_exponent,
      primal_form=primal_form,
      reward_fn=reward_fn,
      primal_regularizer=primal_regularizer,
      dual_regularizer=double_dual_regularizer,
      norm_regularizer=0.,
      nu_regularizer=nu_regularizer,
      zeta_regularizer=zeta_regularizer)
  zeta_estimator = NeuralDice(
      dataset.spec,
      double_nu_network,
      zeta_network,
      double_nu_optimizer,
      zeta_optimizer,
      lam_optimizer,
      gamma,
      zero_reward=zero_reward,
      f_exponent=f_exponent,
      primal_form=primal_form,
      reward_fn=reward_fn,
      primal_regularizer=0.,
      dual_regularizer=dual_regularizer,
      norm_regularizer=norm_regularizer,
      nu_regularizer=nu_regularizer,
      zeta_regularizer=zeta_regularizer) 
  optimizer = NeuralPgDice(
      dataset.spec,
      policy_network,
      nu_network,
      zeta_network,
      policy_optimizer,
      gamma)

  global_step = tf.Variable(0, dtype=tf.int64)
  tf.summary.experimental.set_step(global_step)
  
  avg_rews = []
  env, target_policy = get_env_tfpolicy(env_name, policy_network, tabular_obs)
  avg_rew = optimizer.evaluate_policy(env, num_traj*2, traj_len, target_policy)
  avg_rews.append(avg_rew)
  print('Initial running avg reward:',np.mean(avg_rews))
  # running_losses = []
  # running_estimates = []
  avg_rews = []
  for step in range(num_steps):
    transitions_batch = dataset.get_step(batch_size, num_steps=2)
    initial_steps_batch, _ = dataset.get_episode(
        batch_size, truncate_episode_at=1)
    initial_steps_batch = tf.nest.map_structure(lambda t: t[:, 0, ...],
                                                initial_steps_batch)
    losses = nu_estimator.train_step(initial_steps_batch, transitions_batch,
                                  target_policy)
    losses = zeta_estimator.train_step(initial_steps_batch, transitions_batch,
                                  target_policy)
    # running_losses.append(losses)
    if step % eval_interval == 0 or step == num_steps - 1:
      estimate = optimizer.estimate_average_reward(dataset, target_policy)
      # running_estimates.append(estimate)
      # running_losses = []
      
    if (step-warmup_step) % opt_interval == 0 and step >= warmup_step:
        optimizer.get_zeta_normalizer(dataset)
        # policy_losses = optimizer.train_step(initial_steps_batch, 
        #                                      transitions_batch,
        #                                      target_policy)
        policy_loss = optimizer.fullbatch_train_step(dataset)
        _, target_policy = get_env_tfpolicy(env_name, policy_network, 
                                            tabular_obs)
        if (step-warmup_step) % eval_interval == 0:
            avg_rew = optimizer.evaluate_policy(env, num_traj, traj_len, 
                                                  target_policy)
            avg_rews.append(avg_rew)
            print('Running avg reward:',np.mean(avg_rews))
        
    global_step.assign_add(1)

  print('Done!')

'''
TODO: Move these functions to other files once done debugging
'''
def get_env_tfpolicy(env_name,
                     policy_network,
                     tabular_obs=False,
                     env_seed=0):
    ''' Converts a policy network to a TFPolicy. '''
    
    if env_name == 'grid':
        env = navigation.GridWalk(tabular_obs=tabular_obs)
        env.seed(env_seed)
        tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
        policy = actor_policy.ActorPolicy(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            policy_network)
    elif env_name == 'small_grid':
        env = navigation.GridWalk(length=5,tabular_obs=tabular_obs)
        env.seed(env_seed)
        tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
        policy = actor_policy.ActorPolicy(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            policy_network)
    elif env_name == 'cartpole':
        env = InfiniteCartPole()
        env.seed(env_seed)
        tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
        policy = actor_policy.ActorPolicy(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            policy_network)
    else:
        raise ValueError('Unrecognized environment %s.' % env_name)
  
    return env, policy

def get_distribution_table(load_dir, env_name, tabular_obs=True, alpha=0.):
    ''' Gives the distribution table of a policy alpha-close to optimal policy. '''
    
    init_policy = get_target_policy(load_dir, env_name, tabular_obs, alpha=alpha)
    n_state = init_policy.time_step_spec.observation.maximum - \
              init_policy.time_step_spec.observation.minimum + 1
    state_range = tf.range(n_state)
    tfagent_timestep = ts.restart(state_range, state_range.shape[0])
    init_dist = init_policy.distribution(tfagent_timestep).info['distribution']
    
    return init_dist

        
if __name__ == '__main__':
  app.run(main)
