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
from .type_generic.tf.keras.layers import append_outermost_dim, upstride_type_to_dimension
from .type2.tf.keras.initializers import is_type2_init, QInitializerConv

layers = tf.keras.layers

class GenericConv2D(layers.Conv2D):
  def __init__(self,
               filters,
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
    # constructor of layers.Conv2D will call initializers.get but we want to be able to send our custom initializers.
    # Here the trick is to change the kernel_initializer to "glorot uniform" and rechange it after
    if is_type2_init(kernel_initializer):
      self.saved_kernel_initializer = kernel_initializer
      kernel_initializer = 'glorot_uniform'
    # self.groups are used for depthwise convolutions, which were introduced in TF 2.3. The following if statement
    # is a patch that allows us to have the attributes that we need in order to compute depthwise convolutions
    # regardless the tf.__version__
    if version.parse(tf.__version__) < version.parse("2.3"):
      self.groups = groups
    else:
      kwargs["groups"] = groups
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
    if version.parse(tf.__version__) < version.parse("2.3"):
      self._is_causal = self.padding == 'causal'
    # for specific implementation, call this __init__ function and change this value
    self.upstride_datatype = None
    self.upstride_conv_op = upstride_conv2d
    self.require_input_grad = require_input_grad

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    if input_channel % self.groups != 0:
      raise ValueError(
          'The number of input channels must be evenly divisible by the number '
          'of groups. Received groups={}, but the input has {} channels '
          '(full input shape is {}).'.format(self.groups, input_channel,
                                             input_shape))

    # change initializer if needed
    if is_type2_init(self.kernel_initializer):
      self.kernel_initializer = QInitializerConv(kernel_size=self.kernel_size, input_dim=input_channel // self.groups,
                                                 weight_dim=self.rank, nb_filters=self.filters,
                                                 criterion=self.saved_kernel_initializer.split("_")[-1], seed=None)

    # setup kernel tensor shape
    self.kernel = self.add_weight(
        name='kernel',
        shape=append_outermost_dim(self.upstride_datatype, (self.filters, input_channel // self.groups) + self.kernel_size),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(upstride_type_to_dimension(self.upstride_datatype), self.filters),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    if self.data_format == 'channels_first':
      channel_axis = -1 - self.rank
      self.data_format = "NCHW"
    else:
      channel_axis = -1
      self.data_format = "NHWC"
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_channel})

    # Convert Keras formats to TF native formats.
    if self.padding == 'causal':
      tf_padding = 'VALID'  # Causal padding handled in `call`.
    elif isinstance(self.padding, six.string_types):
      tf_padding = self.padding.upper()
    else:
      tf_padding = self.padding
    self.tf_padding = tf_padding
    self.tf_dilations = list(self.dilation_rate)
    self.tf_strides = list(self.strides)

    self.built = True

  def call(self, inputs):
    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
      inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

    output = self.upstride_conv_op(inputs, self.kernel, self.bias if self.use_bias else [],
                                   uptype=self.upstride_datatype,
                                   strides=self.strides,
                                   dilations=self.dilation_rate,
                                   padding=self.padding.upper(),
                                   data_format=self.data_format,
                                   groups=self.groups,
                                   name=self.name,
                                   require_input_grad=self.require_input_grad,
                                   use_bias=self.use_bias)

    if self.activation is not None:
      return self.activation(output)
    return output


class GenericDepthwiseConv2D(GenericConv2D):
  def __init__(self,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               dilation_rate=(1, 1),
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
    super(GenericDepthwiseConv2D, self).__init__(
        filters=None,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = depthwise_initializer
    self.depthwise_regularizer = tf.keras.regularizers.get(depthwise_regularizer)
    self.depthwise_constraint = tf.keras.constraints.get(depthwise_constraint)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)

  def get_config(self):
    """Returns the configuration of the layer as a JSON-serializable dict.

    Returns:
      A JSON-serializable Python dict.
    """
    config = super(GenericDepthwiseConv2D, self).get_config()
    config.pop('filters')
    config.pop('kernel_initializer')
    config.pop('kernel_regularizer')
    config.pop('kernel_constraint')
    config['depth_multiplier'] = self.depth_multiplier
    config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
    config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
    config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
    return config