"""
FedDICE with neural policy parameterizations for non-tabular state space. Tested on Cartpole
"""

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
import csv
import pickle
import gym
import random

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.networks import q_network
from tf_agents.policies import greedy_policy
from tf_agents import specs
from tf_agents.policies import actor_policy

from dice_rl.environments.env_policies import get_target_policy
from dice_rl.environments.env_policies import get_env_and_dqn_policy
from dice_rl.environments.env_policies import EpsilonGreedyPolicy
import dice_rl.environments.gridworld.navigation as navigation
import dice_rl.environments.gridworld.tree as tree
import dice_rl.environments.gridworld.taxi as taxi
from dice_rl.environments.infinite_cartpole import InfiniteCartPole
from dice_rl.estimators.neural_dice import NeuralDice
from dice_rl.optimizers.neural_pgdice import NeuralPgDice
from dice_rl.estimators import estimator as estimator_lib
from dice_rl.networks.value_network import ValueNetwork
from dice_rl.networks.policy_network import PolicyNetwork
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
import dice_rl.data.tf_agents_onpolicy_dataset as tf_agents_onpolicy_dataset
import dice_rl.environments.env_policies as env_policies


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# from tensorflow.python.client import device_lib
# print( device_lib.list_local_devices())

FLAGS = flags.FLAGS

flags.DEFINE_string('load_dir', '/home/snowfly/dice/dice_rl/tests/testdata/cartpole', 'Directory to load dataset from.')
flags.DEFINE_string('save_dir','/home/snowfly/dice/dice_rl/tests/testdata',
                    'Directory to save the model and estimation results.')
flags.DEFINE_string('env_name', 'cartpole', 'Environment name.')
flags.DEFINE_integer('seed', 0, 'Initial random seed.')
flags.DEFINE_bool('tabular_obs', False, 'Whether to use tabular observations.')
flags.DEFINE_integer('num_trajectory', 400,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 250,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.0, 'How close to target policy.')
flags.DEFINE_float('gamma', 0.99, 'Discount factor.')
flags.DEFINE_float('nu_learning_rate', 0.0000714, 'Learning rate for nu.')
flags.DEFINE_float('zeta_learning_rate', 0.0000714, 'Learning rate for zeta.')
flags.DEFINE_float('policy_learning_rate', 0.0000714, 'Learning rate for theta.')
flags.DEFINE_float('nu_regularizer', 0.0, 'Ortho regularization on nu.')
flags.DEFINE_float('zeta_regularizer', 0.0, 'Ortho regularization on zeta.')
flags.DEFINE_integer('num_steps', 100000, 'Number of training steps.')
flags.DEFINE_integer('batch_size', 2048, 'Batch size.')
flags.DEFINE_integer('n_worker', 5, 'Number of clients.')
flags.DEFINE_integer('warmup_step', 20000, 'Update policy after warmup steps.')
flags.DEFINE_integer('opt_interval', 50, 'Policy is updated every opt_interval steps.')
flags.DEFINE_integer('eval_interval', 50, 'Policy is evaluated every eval_interval steps.')
flags.DEFINE_integer('avg_interval', 100, 'All parameters are averaged over clients every eval_interval steps.')
flags.DEFINE_integer('eval_trajectory', 50, 'Number of trajectories for test cases')
flags.DEFINE_integer('eval_trajectory_length', 100, 'trajectory length for test cases')

flags.DEFINE_float('f_exponent', 2, 'Exponent for f function.')
flags.DEFINE_bool('primal_form', False,
                  'Whether to use primal form of loss for nu.')

flags.DEFINE_float('primal_regularizer', 1.,
                   'LP regularizer of primal variables.')
flags.DEFINE_float('dual_regularizer', 1e-6, 'LP regularizer of dual variables.')
flags.DEFINE_bool('zero_reward', False,
                  'Whether to ignore reward in optimization.')
flags.DEFINE_float('norm_regularizer', 0.,
                   'Weight of normalization constraint.')
flags.DEFINE_bool('zeta_pos', True, 'Whether to enforce positivity constraint.')

flags.DEFINE_float('scale_reward', 1., 'Reward scaling factor.')
flags.DEFINE_float('shift_reward', 0., 'Reward shift factor.')
flags.DEFINE_string(
    'transform_reward', None, 'Non-linear reward transformation'
    'One of [exp, cuberoot, None]')

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
  n_worker = FLAGS.n_worker
  nu_learning_rate = FLAGS.nu_learning_rate * n_worker
  zeta_learning_rate = FLAGS.zeta_learning_rate * n_worker
  policy_learning_rate = FLAGS.policy_learning_rate * n_worker
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
  avg_interval = FLAGS.avg_interval
  warmup_step = FLAGS.warmup_step

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

  activation_fn = tf.nn.relu
  kernel_initializer = tf.keras.initializers.GlorotUniform()
  hidden_dims = (64,64)

  lam_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, clipvalue=1.0)
  nu_optimizer = tf.keras.optimizers.Adam(nu_learning_rate, clipvalue=1.0)
  zeta_optimizer = tf.keras.optimizers.Adam(zeta_learning_rate, clipvalue=1.0)
  policy_optimizer = tf.keras.optimizers.Adam(policy_learning_rate)


  estimator_list = []
  optimizer_list = []

  if env_name == 'grid' or env_name == 'small_grid':
    env = navigation.GridWalk(tabular_obs=tabular_obs, length=gridsize)
    env.seed(seed)
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
  if env_name =='cartpole':
    tf_env, policy = env_policies.get_env_and_dqn_policy(
          'CartPole-v0',
          os.path.join(load_dir, 'CartPole-v0-250', 'train', 'policy'),
          env_seed=seed,
          epsilon=0.45)
    env = InfiniteCartPole()
    tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))


  # load datasets
  dataset_list = []
  for i in range(n_worker):
    hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                  'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}_{INDEX}').format(
                      ENV_NAME=env_name,
                      TAB=tabular_obs,
                      ALPHA=alpha,
                      SEED=seed,
                      NUM_TRAJ=num_trajectory,
                      MAX_TRAJ=max_trajectory_length,
                      INDEX=i)
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
    dataset_list.append(dataset)
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

  

    # Initialize estimators    
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
    output_activation_fn = tf.math.square if zeta_pos else tf.identity
    zeta_network = ValueNetwork(
        input_spec,
        fc_layer_params=hidden_dims,
        activation_fn=activation_fn,
        output_activation_fn=output_activation_fn,
        kernel_initializer=kernel_initializer,
        last_kernel_initializer=kernel_initializer)

    # Initialize optimizers
    policy_network = PolicyNetwork(
            dataset.spec.observation,
            output_spec)

    estimator = NeuralDice(
        dataset.spec,
        nu_network,
        zeta_network,
        nu_optimizer,
        zeta_optimizer,
        lam_optimizer,
        policy,
        gamma,
        zero_reward=zero_reward,
        f_exponent=f_exponent,
        primal_form=primal_form,
        reward_fn=reward_fn,
        primal_regularizer=primal_regularizer,
        dual_regularizer=dual_regularizer,
        norm_regularizer=norm_regularizer,
        nu_regularizer=nu_regularizer,
        zeta_regularizer=zeta_regularizer) 
    optimizer = NeuralPgDice(
        dataset.spec,
        policy_network,
        policy,
        nu_network,
        zeta_network,
        policy_optimizer,
        gamma)


    estimator_list.append(estimator)
    optimizer_list.append(optimizer)

  global_step = tf.Variable(0, dtype=tf.int64)
  tf.summary.experimental.set_step(global_step)

  """save results"""
  filename = "/home/snowfly/dice/dice_rl/results/records/cartpole_"+str(n_worker)+"_"+str(nu_learning_rate)+"_i_"+str(avg_interval)+"_seed_"+str(seed)+"_.csv"
  # filename = "/home/snowfly/dice/dice_rl/results/records/policy_"+str(n_worker)+"_"+"_seed_"+str(seed)+"_.csv"

  with open(filename, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['step', 'primal', 'dual', 'reward', 'running reward'])

  # target_dataset = Dataset.load(
  #     directory.replace('alpha{}'.format(alpha), 'alpha1.0'))
  # print('target per-step',
  #       estimator_lib.get_fullbatch_average(target_dataset, gamma=1.))
  
  running_rewards = [[] for _ in range(n_worker)]
  for step in range(num_steps):
    """
    FedAvg, average estimators and optimizers across clients
    """
    if step % avg_interval == 0:
      nu, zeta = estimator_list[0].get_parameters()
      theta = optimizer_list[0].get_parameters()

      for kk in range(1, n_worker):
        nu_i, zeta_i = estimator_list[kk].get_parameters()
        theta_i = optimizer_list[kk].get_parameters()
        for ll in range(len(nu)):
          nu[ll] = nu[ll] + nu_i[ll]
        for ll in range(len(zeta)):
          zeta[ll] = zeta[ll] + zeta_i[ll]
        for ll in range(len(theta)):
          theta[ll] = theta[ll] + theta_i[ll]


      for ll in range(len(nu)):
        nu[ll] = nu[ll] / n_worker
      for ll in range(len(zeta)):
        zeta[ll] = zeta[ll] /n_worker
      for ll in range(len(theta)):
        theta[ll] = theta[ll] /n_worker

      for kk in range(n_worker):
        estimator_list[kk].update(nu, zeta)
        optimizer_list[kk].update(theta)

    """Training """
    for kk in range(n_worker):
      """ Updating primal and dual estimators """
      transitions_batch = dataset_list[kk].get_step(batch_size, num_steps=2)
      initial_steps_batch, _ = dataset_list[kk].get_episode(
          batch_size, truncate_episode_at=1)
      initial_steps_batch = tf.nest.map_structure(lambda t: t[:, 0, ...],
                                                  initial_steps_batch)
      
      losses = estimator.train_step(initial_steps_batch, transitions_batch)
      if step % eval_interval == 0 or step == num_steps - 1:
        """ Policy evaluation for primal and dual estimators. """
        primal, dual = estimator.estimate_average_reward(dataset)

      if (step-warmup_step) % opt_interval == 0 and step >= warmup_step:
        """ Updating Policy parameters. """
        optimizer.get_zeta_normalizer(dataset)
        policy_losses = optimizer.train_step(initial_steps_batch, 
                                             transitions_batch)
        policy_network = optimizer.get_policy_network()
        target_policy = actor_policy.ActorPolicy(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            policy_network)
        # Policy_network is wrapped into target_policy that can directly sample distribtion
        estimator.update_policy(target_policy)
        optimizer.update_policy(target_policy)
        if (step-warmup_step) % eval_interval == 0:
            """ Estimating policy (average step rewards). """
            reward = optimizer.evaluate_policy(env, num_traj, traj_len)
            running_rewards[kk].append(reward)
            with open(filename, "a") as csvfile:
              writer = csv.writer(csvfile)
              writer.writerow([step, primal, dual, reward, np.mean(running_rewards[kk])])
        

    global_step.assign_add(1)

  print('Done!')



def get_env_tfpolicy(env_name,
                     policy_network,
                     tabular_obs=False,
                     env_seed=0):
    ''' Returns a gym env and converts a policy network to a TFPolicy. '''
    
    if env_name == 'grid':
        env = navigation.GridWalk(tabular_obs=tabular_obs)
        env.seed(env_seed)
        tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
        policy = actor_policy.ActorPolicy(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            policy_network)
    elif env_name == 'small_grid':
        env = navigation.GridWalk(length=3,tabular_obs=tabular_obs)
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
