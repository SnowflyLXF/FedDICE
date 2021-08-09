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

import threading


# BEGIN GOOGLE-INTERNAL
# import google3.learning.deepmind.xmanager2.client.google as xm
# END GOOGLE-INTERNAL
'''
Remove this log blocker once retracing problem is solved
'''
# import logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# logging.getLogger('tensorflow').setLevel(logging.FATAL)
     
 
load_dir ='./tests/testdata/grid5'
save_dir = './tests/testdata/'
env_name ='small_grid'
seed = [0,1,2,3,4,5,6,7]
tabular_obs = True
num_trajectory = 400
max_trajectory_length = 100
alpha = 0.0
gamma = 0.99
nu_learning_rate = 0.00003
zeta_learning_rate = 0.00003
policy_learning_rate = 0.001
nu_regularizer = 0.0
zeta_regularizer = 0.0
num_steps = 1000000
batch_size = 2048

f_exponent = 2
primal_form = False

primal_regularizer = 1.
dual_regularizer = 1.
zero_reward = False
norm_regularizer = 1.
zeta_pos = True
scale_reward = 1.
shift_reward = 0.
transform_reward = None

eval_trajectory = 100
eval_trajectory_length = 100
opt_interval = 1000
eval_interval = 1000
warmup_step = 30000
double_dual_regularizer = 1e-6

n_worker = 5

global nu, nu_double, zeta, zeta_double, theta

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

def get_distribution_table(load_dir, env_name, tabular_obs=True, alpha=0.):
    ''' Gives the distribution table of a policy alpha-close to optimal policy. '''
    
    init_policy = get_target_policy(load_dir, env_name, tabular_obs, alpha=alpha)
    n_state = init_policy.time_step_spec.observation.maximum - \
              init_policy.time_step_spec.observation.minimum + 1
    state_range = tf.range(n_state)
    tfagent_timestep = ts.restart(state_range, state_range.shape[0])
    init_dist = init_policy.distribution(tfagent_timestep).info['distribution']
    
    return init_dist

datasets, nu_estimators, zeta_estimators, optimizers = [],[],[],[]
for i in range(n_worker):
  hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
              'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                  ENV_NAME=env_name,
                  TAB=tabular_obs,
                  ALPHA=alpha,
                  SEED=seed[i],
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
  datasets.append(dataset)
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
  nu_estimators.append(nu_estimator)
  zeta_estimators.append(zeta_estimator)
  optimizers.append(optimizer)


def main(argv):
  global_step = tf.Variable(0, dtype=tf.int64)
  tf.summary.experimental.set_step(global_step)

  lock = threading.Lock()
  t0 = threading.Thread(target=train_local, args=(0, lock,))
  t1 = threading.Thread(target=train_local, args=(1, lock,))
  t2 = threading.Thread(target=train_local, args=(2, lock,))
  t3 = threading.Thread(target=train_local, args=(3, lock,))
  t4 = threading.Thread(target=train_local, args=(4, lock,))

  t0.start()
  t1.start()
  t2.start()
  t3.start()
  t4.start()

  t0.join()
  t1.join()
  t2.join()
  t3.join()
  t4.join()
  

def synchronization(kk):
  global nu, nu_double, zeta, zeta_double, theta
  nu, zeta_double = nu_estimators[0].get_parameters()
  nu_double, zeta = zeta_estimators[0].get_parameters()
  theta = optimizers[0].get_policy_parameters()

  for kk in range(1, n_worker):
    nu_i, zeta_double_i = nu_estimators[kk].get_parameters()
    nu_double_i, zeta_i = zeta_estimators[kk].get_parameters()
    theta_i = optimizers[kk].get_policy_parameters()
    for ll in range(len(nu)):
      nu[ll] = nu[ll] + nu_i[ll]
      nu_double[ll] = nu_double[ll] + nu_double_i[ll]
    for ll in range(len(zeta)):
      zeta[ll] = zeta[ll] + zeta_i[ll]
      zeta_double[ll] = zeta_double[ll] + zeta_double_i[ll]
    for ll in range(len(theta)):
      theta[ll] = theta[ll] + theta_i[ll]


    for ll in range(len(nu)):
      nu[ll] = nu[ll] / n_worker
      nu_double[ll] = nu_double[ll] / n_worker
    for ll in range(len(zeta)):
      zeta[ll] = zeta[ll] / n_worker
      zeta_double[ll] = zeta_double[ll] / n_worker
    for ll in range(len(theta)):
      theta[ll] = theta[ll] / n_worker

    nu_estimators[kk].update(nu, zeta_double)
    zeta_estimators[kk].update(nu_double, zeta)
    optimizers[kk].update(theta)

def train_local(kk, lock):  
  avg_rews = []
  env, target_policy = get_env_tfpolicy(env_name, policy_network, tabular_obs)
  avg_rew = optimizer.evaluate_policy(env, eval_trajectory*2, eval_trajectory_length, target_policy)
  avg_rews.append(avg_rew)
  print('Initial running avg reward:',np.mean(avg_rews))

  for step in range(num_steps):
    lock.acquire()
    synchronization(kk)
    lock.release()
    transitions_batch = datasets[kk].get_step(batch_size, num_steps=2)
    initial_steps_batch, _ = datasets[kk].get_episode(batch_size, truncate_episode_at=1)
    initial_steps_batch = tf.nest.map_structure(lambda t: t[:, 0, ...],
                                                    initial_steps_batch)
    losses = nu_estimators[kk].train_step(initial_steps_batch, transitions_batch,
                                  target_policy)
    losses = zeta_estimators[kk].train_step(initial_steps_batch, transitions_batch,
                                   target_policy)


    if step % eval_interval == 0 or step == num_steps - 1:
      estimate = optimizers[kk].estimate_average_reward(dataset, target_policy)
        
    if (step-warmup_step) % opt_interval == 0 and step >= warmup_step:
      optimizers[kk].get_zeta_normalizer(datasets[kk])
      policy_loss = optimizers[kk].fullbatch_train_step(dataset)
      policy = optimizers[kk].get_policy_network()
      _, target_policy = get_env_tfpolicy(env_name, policy, 
                                          tabular_obs)
      if (step-warmup_step) % eval_interval == 0:
        avg_rew = optimizers[kk].evaluate_policy(env, eval_trajectory, eval_trajectory_length, 
                                              target_policy)
        avg_rews.append(avg_rew)
        print('step', step, 'Running avg reward:',np.mean(avg_rews))

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



        
if __name__ == '__main__':
  app.run(main)
