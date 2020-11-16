import tensorflow as tf
from .type_generic.custom_op import upstride_dense
from .type_generic.tf.keras.layers import append_outermost_dim, upstride_type_to_dimension
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec


class GenericDense(tf.keras.layers.Dense):
  def __init__(self,
               units,
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
               **kwargs):
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
    self.upstride_datatype = None
    self.upstride_dense_op = upstride_dense
    self.require_input_grad = require_input_grad

  def build(self, input_shape):
    # get number of output dimensions
    input_shape = tensor_shape.TensorShape(input_shape)
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Dense` should be defined. Found `None`.')

    # specify input spec ("input_spec enable the layer to run input compatibility checks for input
    # structure, input rank, input shape, and input dtype", see TF InputSpec)
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})

    # set kernel tensor
    self.kernel = self.add_weight(
        'kernel',
        shape=append_outermost_dim(self.upstride_datatype, (last_dim, self.units)),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)

    # set bias tensor if needed
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=append_outermost_dim(self.upstride_datatype, (self.units,)),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None

    self.built = True

  def call(self, inputs):
    output = self.upstride_dense_op(inputs,
                                    self.kernel,
                                    self.bias if self.use_bias else [],
                                    uptype=self.upstride_datatype,
                                    name=self.name,
                                    require_input_grad=self.require_input_grad,
                                    use_bias=self.use_bias)

    if self.activation is not None:
      return self.activation(output)
    return output

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError('The innermost dimension of input_shape must be defined, but saw: %s' % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = super().get_config()
    config["require_input_grad"] = self.require_input_grad
    return config
