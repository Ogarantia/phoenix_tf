from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops
import tensorflow as tf

load_library.load_op_library(resource_loader.get_path_to_datafile('libdnnl.so.1'))
upstride_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_upstride.so'))

upstride_conv2d = upstride_ops.upstride_conv2d

# declare backward operations


@ops.RegisterGradient("UpstrideConv2D")
def _conv2d_grad(op, grad):
  if len(op.inputs[2].shape) == 1:
    dbias = tf.reduce_sum(grad[:, 0]) * tf.ones_like(op.inputs[2]) # bias gradient of scalar convolutions is a 1D tensor
  else:
    dbias = tf.reduce_sum(grad[:, 0]) // op.inputs[2].shape[0] * tf.ones_like(op.inputs[2])
  return upstride_ops.upstride_conv2d_grad(grad, op.inputs[0], op.inputs[1],
                                           uptype=op.get_attr('uptype'),
                                           strides=op.get_attr("strides"),
                                           padding=op.get_attr("padding"),
                                           dilations=op.get_attr("dilations"),
                                           data_format=op.get_attr("data_format"),
                                           groups=op.get_attr("groups"),
                                           require_input_grad=op.get_attr("require_input_grad")), dbias
