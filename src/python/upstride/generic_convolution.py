"""When working on this file, it can be usefull to have the file /tensorflow/python/keras/layers/convolutional.py open
"""

import six
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops, nn
from .type_generic.custom_op import upstride_conv2d
from .type_generic.tf.keras.layers import SCALAR, upstride_type_to_dimension

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
               **kwargs):
    super().__init__(filters,
                     kernel_size,
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
    # for specific implementation, call this __init__ function and change this value
    self.upstride_datatype = None
    self.upstride_conv_op = upstride_conv2d

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    if input_channel % self.groups != 0:
      raise ValueError(
          'The number of input channels must be evenly divisible by the number '
          'of groups. Received groups={}, but the input has {} channels '
          '(full input shape is {}).'.format(self.groups, input_channel,
                                             input_shape))

    # setup kernel tensor shape
    if self.upstride_datatype == SCALAR:
      kernel_shape = (self.filters, input_channel // self.groups) + self.kernel_size
    else:
      kernel_shape = (upstride_type_to_dimension(self.upstride_datatype), self.filters, input_channel // self.groups) + self.kernel_size

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
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
    tf_dilations = list(self.dilation_rate)
    tf_strides = list(self.strides)

    self.built = True

  def call(self, inputs):
    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
      inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

    output = self.upstride_conv_op(inputs, self.kernel, uptype=self.upstride_datatype, strides=self.strides,
                                   dilations=self.dilation_rate, padding=self.padding.upper(), data_format=self.data_format, name=self.name)

    # TODO gros todo, for now it doesn't work
    if self.use_bias:
      raise NotImplementedError("hello, the bias it not implemented")
      # if self.data_format == 'channels_first':
      #   if self.rank == 1:
      #     # nn.bias_add does not accept a 1D input tensor.
      #     bias = array_ops.reshape(self.bias, (1, self.filters, 1))
      #     outputs += bias
      #   if self.rank == 2:
      #     outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
      #   if self.rank == 3:
      #     # As of Mar 2017, direct addition is significantly slower than
      #     # bias_add when computing gradients. To use bias_add, we collapse Z
      #     # and Y into a single dimension to obtain a 4D input tensor.
      #     outputs_shape = outputs.shape.as_list()
      #     if outputs_shape[0] is None:
      #       outputs_shape[0] = -1
      #     outputs_4d = array_ops.reshape(outputs,
      #                                    [outputs_shape[0], outputs_shape[1],
      #                                     outputs_shape[2] * outputs_shape[3],
      #                                     outputs_shape[4]])
      #     outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
      #     outputs = array_ops.reshape(outputs_4d, outputs_shape)
      # else:
      #   outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(output)
    return output
