import tensorflow as tf
layers = tf.keras.layers


class GenericConv2D(layers.Conv2D):
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
                     **kwargs)
    # for specific implementation, call this __init__ function and change this value
    self.upstride_type_dim = None
    self.upstride_conv_op = None

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    kernel_shape = (self.upstride_type_dim, self.kernel_size[0], self.kernel_size[1], input_dim, self.filters)

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
          shape=(self.upstride_type_dim, self.filters),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_dim})
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    self.built = True

  def call(self, inputs):
    output = self.upstride_conv_op(inputs, self.kernel, self.strides, self.padding, self.data_format, self.dilation_rate, self.name)

    # TODO gros todo, for now it doesn't work
    if self.use_bias:
      raise NotImplementedError("coucou, the bias it not implemented")
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
      return self.activation(outputs)
    return outputs
