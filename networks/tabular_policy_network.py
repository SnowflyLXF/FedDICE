
import gin
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tf_agents.networks import network
from tf_agents.networks import normal_projection_network
from tf_agents.networks import utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils
from tf_agents.specs import distribution_spec

import dice_rl.utils.common as common_lib


def _categorical_projection_net(action_spec, 
                                init_dist,
                                logits_init_output_factor=0.1):
  return MyCategoricalProjectionNetwork(action_spec, 
            init_dist=init_dist, 
            logits_init_output_factor=logits_init_output_factor)



class TabularSoftmaxPolicyNetwork(network.DistributionNetwork):
  """Creates a policy network."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               initial_distribution=None,
               discrete_projection_net=_categorical_projection_net,
               name='PolicyNetwork'):
    """Creates an instance of `TabularSoftmaxNetwork`.

    Args:
      input_tensor_spec: A possibly nested container of
        `tensor_spec.TensorSpec` representing the inputs.
      output_tensor_spec: A possibly nested container of
        `tensor_spec.TensorSpec` representing the outputs.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      output_activation_fn: Activation function for the last layer. This can be
        used to restrict the range of the output. For example, one can pass
        tf.keras.activations.sigmoid here to restrict the output to be bounded
        between 0 and 1.
      kernel_initializer: kernel initializer for all layers except for the value
        regression layer. If None, a VarianceScaling initializer will be used.
      last_kernel_initializer: kernel initializer for the value regression
         layer. If None, a RandomUniform initializer will be used.
      discrete_projection_net: projection layer for discrete actions.
      continuous_projection_net: projection layer for continuous actions.
      name: A string representing name of the network.
    """

    def map_proj(spec, init=initial_distribution):
        return discrete_projection_net(spec, init)

    projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
    output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                        projection_networks)

    action_dim = np.unique(output_tensor_spec.maximum -
                           output_tensor_spec.minimum + 1)

    super(TabularSoftmaxPolicyNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    self._flat_specs = tf.nest.flatten(input_tensor_spec)


    self._projection_networks = projection_networks
    self._output_tensor_spec = output_tensor_spec

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self, inputs, step_type=(), network_state=(), training=False, mask=None):

    flat_inputs = tf.nest.flatten(inputs)
    del step_type  # unused.

    processed_inputs = []
    for single_input, input_spec in zip(flat_inputs, self._flat_specs):
      if common_lib.is_categorical_spec(input_spec):
        processed_input = tf.one_hot(single_input, input_spec.maximum + 1)
      else:
        if len(input_spec.shape) != 1:  # Only allow vector inputs.
          raise ValueError('Invalid input spec shape %s.' % input_spec.shape)
        processed_input = single_input
      processed_inputs.append(processed_input)

    joint = tf.concat(processed_inputs, -1)

    outer_rank = nest_utils.get_outer_rank(inputs, self.input_tensor_spec)

    def call_projection_net(proj_net):
      distribution, _ = proj_net(
          joint, outer_rank, training=training, mask=mask)
      return distribution

    output_actions = tf.nest.map_structure(call_projection_net,
                                           self._projection_networks)
    return output_actions, network_state



@gin.configurable
class MyCategoricalProjectionNetwork(network.DistributionNetwork):
  """Generates a tfp.distribution.Categorical by predicting logits."""

  def __init__(self,
               sample_spec,
               init_dist=None,
               logits_init_output_factor=0.1,
               name='CategoricalProjectionNetwork'):
    """Creates an instance of CategoricalProjectionNetwork.
    Args:
      sample_spec: A `tensor_spec.BoundedTensorSpec` detailing the shape and
        dtypes of samples pulled from the output distribution.
      init_dist: If is not None, the policy is initialized to this distribution.
      logits_init_output_factor: Output factor for initializing kernel logits
        weights used when init_dist is None.
      name: A string representing name of the network.
    """
    unique_num_actions = np.unique(sample_spec.maximum - sample_spec.minimum +
                                   1)
    if len(unique_num_actions) > 1 or np.any(unique_num_actions <= 0):
      raise ValueError('Bounds on discrete actions must be the same for all '
                       'dimensions and have at least 1 action. Projection '
                       'Network requires num_actions to be equal across '
                       'action dimensions. Implement a more general '
                       'categorical projection if you need more flexibility.')

    output_shape = sample_spec.shape.concatenate([int(unique_num_actions)])
    output_spec = self._output_distribution_spec(output_shape, sample_spec,
                                                 name)

    super(MyCategoricalProjectionNetwork, self).__init__(
        # We don't need these, but base class requires them.
        input_tensor_spec=None,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    if not tensor_spec.is_bounded(sample_spec):
      raise ValueError(
          'sample_spec must be bounded. Got: %s.' % type(sample_spec))

    if not tensor_spec.is_discrete(sample_spec):
      raise ValueError('sample_spec must be discrete. Got: %s.' % sample_spec)

    self._sample_spec = sample_spec
    self._output_shape = output_shape
    
    if init_dist is not None:
        self._projection_layer = tf.keras.layers.Dense(
            self._output_shape.num_elements(),
            kernel_initializer=tf.keras.initializers.Constant(
                value=tf.math.log(init_dist)),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='logits')
    else:
        self._projection_layer = tf.keras.layers.Dense(
            self._output_shape.num_elements(),
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=logits_init_output_factor),
            bias_initializer=tf.keras.initializers.Zeros(),
            name='logits')

  def _output_distribution_spec(self, output_shape, sample_spec, network_name):
    input_param_spec = {
        'logits':
            tensor_spec.TensorSpec(
                shape=output_shape,
                dtype=tf.float32,
                name=network_name + '_logits')
    }

    return distribution_spec.DistributionSpec(
        tfp.distributions.Categorical,
        input_param_spec,
        sample_spec=sample_spec,
        dtype=sample_spec.dtype)

  def call(self, inputs, outer_rank, training=False, mask=None):
    # outer_rank is needed because the projection is not done on the raw
    # observations so getting the outer rank is hard as there is no spec to
    # compare to.
    batch_squash = utils.BatchSquash(outer_rank)
    inputs = batch_squash.flatten(inputs)
    inputs = tf.cast(inputs, tf.float32)

    logits = self._projection_layer(inputs, training=training)
    logits = tf.reshape(logits, [-1] + self._output_shape.as_list())
    logits = batch_squash.unflatten(logits)

    if mask is not None:
      # If the action spec says each action should be shaped (1,), add another
      # dimension so the final shape is (B, 1, A), where A is the number of
      # actions. This will make Categorical emit events shaped (B, 1) rather
      # than (B,). Using axis -2 to allow for (B, T, 1, A) shaped q_values.
      if mask.shape.rank < logits.shape.rank:
        mask = tf.expand_dims(mask, -2)

      # Overwrite the logits for invalid actions to a very large negative
      # number. We do not use -inf because it produces NaNs in many tfp
      # functions.
      almost_neg_inf = tf.constant(logits.dtype.min, dtype=logits.dtype)
      logits = tf.compat.v2.where(
          tf.cast(mask, tf.bool), logits, almost_neg_inf)

    return self.output_spec.build_distribution(logits=logits), ()