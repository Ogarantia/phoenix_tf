import tensorflow as tf
from upstride.internal import convolution, dense, layers
from upstride.type1.tf.keras.initializers import is_type1_init, CInitializerConv, CInitializerDepthwiseConv, CInitializerDense
from upstride.internal.layers import *

layers.upstride_type = 1
layers.blade_indexes = ["", "12"]
layers.geometrical_def = (2, 0, 0)


class TF2Upstride(layers.GenericTF2Upstride):
  def __init__(self, strategy='', name=None):
    super().__init__(layers.TYPE1, strategy, name=name)


class Upstride2TF(layers.GenericUpstride2TF):
  def __init__(self, strategy='', name=None):
    super().__init__(layers.TYPE1, strategy, name=name)


@tf.keras.utils.register_keras_serializable("upstride_type1")
class Conv2D(convolution.GenericConv2D):
  def __init__(self, filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='up1_init_he',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    # intercept kernel initializer string
    if is_type1_init(kernel_initializer):
      kernel_initializer = CInitializerConv(criterion=kernel_initializer, groups=groups, data_format=data_format)
    super().__init__(filters=filters,
                     kernel_size=kernel_size,
                     strides=strides,
                     padding=padding,
                     data_format=data_format,
                     dilation_rate=dilation_rate,
                     groups=groups,
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
    self.upstride_datatype = layers.TYPE1


@tf.keras.utils.register_keras_serializable("upstride_type1")
class DepthwiseConv2D(convolution.GenericDepthwiseConv2D):
  def __init__(self,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               depthwise_initializer='up1_init_he',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    # intercept kernel initializer string
    if is_type1_init(depthwise_initializer):
      depthwise_initializer = CInitializerDepthwiseConv(criterion=depthwise_initializer, depth_multiplier=depth_multiplier, data_format=data_format)
    super().__init__(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        data_format=data_format,
        dilation_rate=dilation_rate,
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
    self.upstride_datatype = layers.TYPE1


@tf.keras.utils.register_keras_serializable("upstride_type1")
class Dense(dense.GenericDense):
  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='up1_init_he',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    # intercept kernel initializer string
    if is_type1_init(kernel_initializer):
      kernel_initializer = CInitializerDense(criterion=kernel_initializer)
    super().__init__(units=units,
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
    self.upstride_datatype = layers.TYPE1
