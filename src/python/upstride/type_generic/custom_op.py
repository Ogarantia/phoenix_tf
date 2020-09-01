from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.framework import ops

upstride_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_upstride.so'))

upstride_conv2d = upstride_ops.upstride_conv2d

# declare backward operations


@ops.RegisterGradient("UpstrideConv2D")
def _conv2d_grad(op, grad):
  return upstride_ops.upstride_conv2d_grad(grad, op.inputs[0], op.inputs[1],
                                           uptype=op.get_attr('uptype'),
                                           strides=op.get_attr("strides"),
                                           padding=op.get_attr("padding"),
                                           dilations=op.get_attr("dilations"),
                                           data_format=op.get_attr("data_format"),
                                           groups=op.get_attr("groups"),
                                           require_input_grad=op.get_attr("require_input_grad")), grad  # bias gradient is equal to the loss function gradient input, so returned as is
