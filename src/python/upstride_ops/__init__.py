from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

upstride_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_upstride.so'))
upstride_input = upstride_ops.upstride_input
upstride_kernel = upstride_ops.upstride_kernel
upstride_output = upstride_ops.upstride_output
