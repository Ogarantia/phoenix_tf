import functools
import tensorflow as tf
from .... import generic_layers
from .... generic_layers import *

# algebra identification
TYPE0 = 0
TYPE1 = 1
TYPE2 = 2
TYPE3 = 3

# FIXME: check use, consider removing; comment if still necessary
generic_layers.blade_indexes = None
generic_layers.geometrical_def = None

@functools.lru_cache(maxsize=1)
def upstride_type_to_dimension(type):
  """ Returns dimensionality of a specific algebra
  """
  dimensions = {
      TYPE0: 1,
      TYPE1: 2,
      TYPE2: 4,
      TYPE3: 8
  }
  return dimensions[type]


def append_outermost_dim(type, shape):
  """ Adds to a tensor shape the outermost dimension matching a specific algebra dimension, if needed
  """
  return shape if type == TYPE0 else (upstride_type_to_dimension(type),) + shape


class CustomInitializer(tf.keras.initializers.Initializer):
  """ Base class for Upstride initializers.
  Standard keras initializers may change the underlying distribution parameters depending on tensor shapes.
  To apply them to multidimensional UpStride datatypes, interception mechanisms are implemented. All the internal
  initializers are assumed to be derived from this class to make sure the interception mechanics works correctly.
  """
  pass
