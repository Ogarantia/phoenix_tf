import tensorflow as tf
from tensorflow.python.framework import load_library, ops
from tensorflow.python.platform import resource_loader
from upstride.internal.layers import TYPE0

# Load shared objects
load_library.load_op_library(resource_loader.get_path_to_datafile('libdnnl.so.1'))
upstride_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('libupstride.so'))

upstride_conv2d = upstride_ops.upstride_conv2d
upstride_dense = upstride_ops.upstride_dense

# declare backward operations


@ops.RegisterGradient("UpstrideConv2D")
def _conv2d_grad(op, grad):
  # The bias gradient is the sum of the incoming gradient. The elements of our datatype are interlaced
  # in the incoming gradient. Hence, we reshape to unscramble the interlacement before the computing the addition.
  # op.inputs[2] is the bias. When the bias does not exist, op.inputs[2].shape = [0], which is not handy for the reshape hereafter
  # Given that op.inputs[1].shape[-1] is equal to op.inputs[2].shape[-1] when the bias exist and that we do not care of dbias
  # when the bias does not exist, using op.inputs[1].shape[-1] is preferable over op.inputs[2].shape[-1] to avoid 0
  if op.inputs[2].shape[0] == 0:
    dbias = tf.zeros_like(op.inputs[2].shape)
  elif op.get_attr('uptype') != TYPE0:
    numel_dtype = op.inputs[2].shape[0]
    grad_reshape = tf.reshape(grad, [numel_dtype, -1] + grad.shape[1:].as_list())
    dbias = tf.reduce_sum(grad_reshape, [1, 3, 4])
  else:
    dbias = tf.reduce_sum(grad, [0, 2, 3])
  return upstride_ops.upstride_conv2d_grad(grad, op.inputs[0], op.inputs[1],
                                           uptype=op.get_attr('uptype'),
                                           strides=op.get_attr("strides"),
                                           padding=op.get_attr("padding"),
                                           dilations=op.get_attr("dilations"),
                                           data_format=op.get_attr("data_format"),
                                           groups=op.get_attr("groups"),
                                           require_input_grad=op.get_attr("require_input_grad")), dbias

@ops.RegisterGradient("UpstrideDense")
def _dense_grad(op, grad):
  # The bias gradient is the sum of the incoming gradient. The elements of our datatype are interlaced
  # in the incoming gradient. Hence, we reshape to unscramble the interlacement before the computing the addition.
  # op.inputs[2] is the bias. When the bias does not exist, op.inputs[2].shape = [0], which is not handy for the reshape hereafter
  # Given that op.inputs[1].shape[-1] is equal to op.inputs[2].shape[-1] when the bias exist and that we do not care of dbias
  # when the bias does not exist, using op.inputs[1].shape[-1] is preferable over op.inputs[2].shape[-1] to avoid 0
  if op.inputs[2].shape[0] == 0:
    dbias = tf.zeros_like(op.inputs[2].shape)
  elif op.get_attr('uptype') != TYPE0:
    numel_dtype = op.inputs[2].shape[0]
    grad_reshape = tf.reshape(grad, [numel_dtype, -1, op.inputs[1].shape[-1]])
    dbias = tf.reduce_sum(grad_reshape, 1)
  else:
    dbias = tf.reduce_sum(grad, 0)
  return upstride_ops.upstride_dense_grad(grad, op.inputs[0], op.inputs[1],
                                          uptype=op.get_attr('uptype'),
                                          require_input_grad=op.get_attr("require_input_grad")), dbias
