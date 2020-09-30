from typing import Dict

from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import load_library
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec

from upstride.generic_convolution import GenericConv2D
from upstride.generic_dense import GenericDense
from upstride.type_generic.tf.keras.layers import SCALAR

from .... import generic_layers
from ....generic_layers import *

generic_layers.upstride_type = 0
generic_layers.blade_indexes = [""]
generic_layers.geometrical_def = (0, 0, 0)

# If you wish to overwrite some layers, please implement them here


class TF2Upstride(Layer):
  """assume this function is called at the begining of the network. 
  Select between several strategies, like putting colors to imaginary parts and grayscale in real, ...

  this function exists only for compatibility with other types, but it does nothing
  """

  def __init__(self, strategy=''):
    pass

  def __call__(self, x):
    return x


class Upstride2TF(Layer):
  """convert multivector back to real values.

  this function exists only for compatibility with other types, but it does nothing
  """

  def __init__(self, strategy=''):
    pass

  def __call__(self, x):
    return x


class Conv2D(GenericConv2D):
  def __init__(self, filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               require_input_grad=True,
               **kwargs):
    super().__init__(filters,
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
                     require_input_grad,
                     **kwargs)
    self.upstride_datatype = SCALAR


class DepthwiseConv2D(Conv2D):
  def __init__(self,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    super().__init__(
        filters=None,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = tf.keras.initializers.get(depthwise_initializer)
    self.depthwise_regularizer = tf.keras.regularizers.get(depthwise_regularizer)
    self.depthwise_constraint = tf.keras.constraints.get(depthwise_constraint)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                       'Received input shape:', str(input_shape))
    input_shape = tensor_shape.TensorShape(input_shape)
    self.groups = input_shape[1] if self.data_format == 'channels_first' else input_shape[3]
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs to '
                       '`DepthwiseConv2D` '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    depthwise_kernel_shape = (self.groups,
                              self.depth_multiplier,
                              self.kernel_size[0],
                              self.kernel_size[1])

    self.depthwise_kernel = self.add_weight(
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        name='depthwise_kernel',
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint)

    if self.use_bias:
      self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                  initializer=self.bias_initializer,
                                  name='bias',
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs):
    outputs = self.upstride_conv_op(
        inputs,
        self.depthwise_kernel,
        self.bias if self.use_bias else [],
        uptype=self.upstride_datatype,
        strides=self.strides,
        padding=self.padding.upper(),
        dilations=self.dilation_rate,
        data_format="NCHW" if self.data_format == 'channels_first' else "NHWC",
        groups=self.groups)

    if self.activation is not None:
      return self.activation(outputs)

    return outputs


class Dense(GenericDense):
  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               require_input_grad=True,
               **kwargs):
    super().__init__(units,
                      activation,
                      use_bias,
                      kernel_initializer,
                      bias_initializer,
                      kernel_regularizer,
                      bias_regularizer,
                      activity_regularizer,
                      kernel_constraint,
                      bias_constraint,
                      require_input_grad,
                      **kwargs)
                      
    self.upstride_datatype = SCALAR