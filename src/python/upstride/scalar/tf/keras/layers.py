from typing import Dict

from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import load_library
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import tf_utils

from upstride.generic_convolution import GenericConv2D
from upstride.type_generic.tf.keras.layers import TYPE2

from .... import generic_layers
from ....generic_layers import *
from .dense import Dense

generic_layers.upstride_type = 0
generic_layers.blade_indexes = [""]
generic_layers.geometrical_def = (0, 0, 0)

# If you wish to overwrite some layers, please implements them here


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
    self.upstride_datatype = 0


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
    self.depthwise_initializer = initializers.get(depthwise_initializer)
    self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
    self.depthwise_constraint = constraints.get(depthwise_constraint)
    self.bias_initializer = initializers.get(bias_initializer)

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                       'Received input shape:', str(input_shape))
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs to '
                       '`DepthwiseConv2D` '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    # 1 is the size of the group
    # in pure tensorflow, code is like this :
    # depthwise_kernel_shape = (self.kernel_size[0],
    #                           self.kernel_size[1],
    #                           input_dim,
    #                           self.depth_multiplier)
    # in upstride, the order need to be (o, g, h, w)
    depthwise_kernel_shape = (1,  # number of output channels per group
                              self.depth_multiplier,  # number of groups, so number of channels for depth wise conv
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
        uptype=self.upstride_datatype,
        strides=self.strides,
        padding=self.padding,
        dilation_rate=self.dilation_rate,
        data_format=self.data_format,
        groups=inputs.shape[1])

    if self.use_bias:
      outputs = backend.bias_add(
          outputs,
          self.bias,
          data_format=self.data_format)

    return outputs
