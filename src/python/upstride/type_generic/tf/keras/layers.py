import functools
from .... import generic_layers
from .... generic_layers import *

SCALAR = 0
TYPE1 = 1
TYPE2 = 2
TYPE3 = 3


@functools.lru_cache(maxsize=1)
def upstride_type_to_dimension(type):
  dimensions = {
      SCALAR: 1,
      TYPE1: 2,
      TYPE2: 4,
      TYPE3: 8
  }
  return dimensions[type]


generic_layers.blade_indexes = None
generic_layers.geometrical_def = None


def define_ga(a, b, c, blades):
  generic_layers.blade_indexes = blades
  generic_layers.geometrical_def = (a, b, c)
