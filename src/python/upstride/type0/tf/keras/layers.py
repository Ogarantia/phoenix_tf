from typing import Dict

from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import load_library
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec

from upstride.generic_convolution import GenericConv2D, GenericDepthwiseConv2D
from upstride.generic_dense import GenericDense
from upstride.type_generic.tf.keras.layers import TYPE0, upstride_type_to_dimension

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


@tf.keras.utils.register_keras_serializable("upstride_type0")
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
    self.upstride_datatype = TYPE0


@tf.keras.utils.register_keras_serializable("upstride_type0")
class DepthwiseConv2D(GenericDepthwiseConv2D):
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
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        depthwise_initializer=depthwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=depthwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    self.upstride_datatype = TYPE0

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
        name=self.name,
        groups=self.groups,
        use_bias=self.use_bias)

    if self.activation is not None:
      return self.activation(outputs)

    return outputs


@tf.keras.utils.register_keras_serializable("upstride_type0")
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
                      
    self.upstride_datatype = TYPE0