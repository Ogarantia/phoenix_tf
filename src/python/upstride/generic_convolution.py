"""When working on this file, it can be usefull to have the file /tensorflow/python/keras/layers/convolutional.py open
"""

import six
from packaging import version
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops, nn
from tensorflow.python.keras import constraints, initializers, regularizers
from .type_generic.custom_op import upstride_conv2d
from .utils import permutation
from .type_generic.tf.keras.layers import TYPE0, append_outermost_dim, upstride_type_to_dimension, CustomInitializer, UpstrideLayer



class Conv2DKernelInitWrapper(CustomInitializer):
  """ Wraps a standard keras kernel initializer to generate a convolution kernel for a given algebra
  Keras initializers may use the kernel shape to adjust parameters of the distribution used to produce the initial
  value of a convolution kernel. The kernel shape is different between UpStride and keras, namely
   - the layouts are different: UpStride type0 kernels are OIHW, while keras kernels are HWIO for regular convolutions
     and HWOI for depthwise convolutions,
   - for an n-dimensional algebra, UpStride kernels are nOIHW.
  This class makes sure the initial distribution matches the expected one when using a standard keras initializer.
  """
  def __init__(self, initializer, depthwise=False):
    self.initializer = tf.keras.initializers.get(initializer)
    # get permutations
    # OIHW is the UpStride way, HWIO / HWOI is expected by TensorFlow
    upstride = 'OIHW'
    tensorflow = 'HWOI' if depthwise else 'HWIO'
    self.up_to_tf = permutation(upstride, tensorflow)
    self.tf_to_up = permutation(tensorflow, upstride)

  def __call__(self, shape, dtype=None):
    # type0 convolutions use 4-dim kernels; other algebras have the 5th outermost blade dimension
    # extend the shape by a singleton dimension to proceed in the same way for different algebras
    if len(shape) == 4:
      shape = (1,) + shape

    # get the shape of a single blade reordering the dimensions from upstride to TF
    blade_shape = tuple(shape[i + 1] for i in self.up_to_tf)

    # initialize blades separately using the keras initializer
    blades = [self.initializer(blade_shape, dtype=dtype) for _ in range(shape[0])]

    # transpose the dimensions back to match the UpStride layout
    blades = [tf.transpose(blade, self.tf_to_up) for blade in blades]

    # concat the blades if needed and return
    return tf.stack(blades, axis=0) if shape[0] > 1 else blades[0]

  def get_config(self):
    return self.initializer.get_config()

  @classmethod
  def from_config(config):
    return Conv2DKernelInitWrapper(super().from_config(config))


class GenericConv2D(tf.keras.layers.Conv2D, UpstrideLayer):
  def __init__(self,
               filters,
               kernel_size,
               strides,
               padding,
               data_format,
               dilation_rate,
               groups,
               activation,
               use_bias,
               kernel_initializer,
               bias_initializer,
               kernel_regularizer,
               bias_regularizer,
               activity_regularizer,
               kernel_constraint,
               bias_constraint,
               **kwargs):

    # Handle group convolution in TF prior to TF2.3
    if version.parse(tf.__version__) < version.parse("2.3"):
      self.groups = groups
    else:
      kwargs["groups"] = groups

    # Intercept keras initializer
    if kernel_initializer and not isinstance(kernel_initializer, CustomInitializer):
      kernel_initializer = Conv2DKernelInitWrapper(kernel_initializer)

    # Call super class constructor
    super().__init__(filters,
                     kernel_size,
                     strides=strides,
                     padding=padding,
                     data_format=data_format,
                     dilation_rate=dilation_rate,
                     activation=activation,
                     use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     activity_regularizer=activity_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint,
                     **kwargs)
    UpstrideLayer.__init__(self)
    self.upstride_conv_op = upstride_conv2d

    # Handling casual paddiing in TF prior to TF2.3
    if version.parse(tf.__version__) < version.parse("2.3"):
      self._is_causal = self.padding.lower() == 'causal'

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)

    # Check if number of input channels is divisible by number of groups
    if input_channel % self.groups != 0:
      raise ValueError('The number of input channels must be evenly divisible by the number of groups. '
                       'Received groups={}, but the input has {} channels (full input shape is {}).'
                       .format(self.groups, input_channel, input_shape))

    # Setup kernel tensor
    kernel_shape = append_outermost_dim(self.upstride_datatype,
                                        (self.filters, input_channel // self.groups) + self.kernel_size)
    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)

    # Setup bias tensor
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=append_outermost_dim(self.upstride_datatype, (self.filters,)),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None

    # specify input spec ("input_spec enable the layer to run input compatibility checks for input
    # structure, input rank, input shape, and input dtype", see TF InputSpec)
    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_channel})
    self.built = True

  def call(self, inputs):
    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
      inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

    output = self.upstride_conv_op(inputs,
                                   self.kernel,
                                   self.bias if self.use_bias else [],
                                   uptype=self.upstride_datatype,
                                   strides=self.strides,
                                   dilations=self.dilation_rate,
                                   padding=self.padding.upper(),
                                   data_format="NCHW" if self.data_format == 'channels_first' else "NHWC",
                                   groups=self.groups,
                                   name=self.name,
                                   require_input_grad=self.require_input_grad,
                                   use_bias=self.use_bias)

    if self.activation is not None:
      return self.activation(output)
    return output

  def compute_mask(self, inputs, previous_mask):
    """ Overrides compute_mask to intercept the graph and call compute_require_input_grad.
    The value of self.require_input_grad depends on self.inbound_nodes, which is defined after the
    method call() is called and before compute_mask() is called.
    """
    super().compute_require_input_grad()
    return super().compute_mask(inputs, previous_mask)

  def get_config(self):
    config = super().get_config()
    config["groups"] = self.groups
    config["require_input_grad"] = self.require_input_grad
    return config


class GenericDepthwiseConv2D(GenericConv2D):
  def __init__(self,
               kernel_size,
               strides,
               padding,
               depth_multiplier,
               data_format,
               dilation_rate,
               activation,
               use_bias,
               depthwise_initializer,
               bias_initializer,
               depthwise_regularizer,
               bias_regularizer,
               activity_regularizer,
               depthwise_constraint,
               bias_constraint,
               **kwargs):

    # Intercept keras initializer
    if depthwise_initializer and not isinstance(depthwise_initializer, CustomInitializer):
      depthwise_initializer = Conv2DKernelInitWrapper(depthwise_initializer, depthwise=True)

    super().__init__(
        filters=None,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        groups=None,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=None,
        bias_initializer=bias_initializer,
        kernel_regularizer=None,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=None,
        bias_constraint=bias_constraint,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = tf.keras.initializers.get(depthwise_initializer)
    self.depthwise_regularizer = tf.keras.regularizers.get(depthwise_regularizer)
    self.depthwise_constraint = tf.keras.constraints.get(depthwise_constraint)


  def build(self, input_shape):
    # check input shape rank
    if len(input_shape) < 4:
      raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. Received input shape:', str(input_shape))

    # retrieve number of input channels (= number of groups for depthwise conv)
    self.groups = self._get_input_channel(input_shape)
    if self.groups is None:
      raise ValueError('The channel dimension of the inputs to `DepthwiseConv2D` should be defined. Found `None`.')

    # compute kernel shape (O, I, H, W)
    depthwise_kernel_shape = append_outermost_dim(self.upstride_datatype,
                                                  (self.groups * self.depth_multiplier,
                                                   1,
                                                   self.kernel_size[0],
                                                   self.kernel_size[1]))

    # initialize kernel tensor
    self.kernel = self.add_weight(
        name='depthwise_kernel',
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint,
        trainable=True,
        dtype=self.dtype)

    # set bias
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=append_outermost_dim(self.upstride_datatype, (self.groups * self.depth_multiplier,)),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None

    # specify input spec ("input_spec enable the layer to run input compatibility checks for input
    # structure, input rank, input shape, and input dtype", see TF InputSpec)
    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: self.groups})
    self.built = True

  def get_config(self):
    """Returns the configuration of the layer as a JSON-serializable dict.

    Returns:
      A JSON-serializable Python dict.
    """
    config = super().get_config()
    config.pop('groups')
    config.pop('filters')
    config.pop('kernel_initializer')
    config.pop('kernel_regularizer')
    config.pop('kernel_constraint')
    config['depth_multiplier'] = self.depth_multiplier
    config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
    config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
    config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
    return config
