import functools
from .... import generic_layers
from .... generic_layers import *

TYPE0 = 0
TYPE1 = 1
TYPE2 = 2
TYPE3 = 3


@functools.lru_cache(maxsize=1)
def upstride_type_to_dimension(type):
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


generic_layers.blade_indexes = None
generic_layers.geometrical_def = None


def define_ga(a, b, c, blades):
  generic_layers.blade_indexes = blades
  generic_layers.geometrical_def = (a, b, c)
