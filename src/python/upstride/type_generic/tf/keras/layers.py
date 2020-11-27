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

class UpstrideLayer():
  """ This class regroups the methods and attributes common to all Upstride layers.
  Hence, all Upstride layers are expected to inherit from UpstrideLayer.
  """
  def __init__(self):
    """ Defines the attributes common to all Upstride layers.
    """
    self.upstride_datatype = None # Value to specify in subclass
    self.require_input_grad = None

  # TODO consider parsing the graph from the input to the children - rather than from the current node to
  # its parents -, as detailed at
  # https://bitbucket.org/upstride/phoenix_tf/pull-requests/59/feature-pe-170-compute-input-grad#comment-190697967
  def compute_require_input_grad(self):
    """ Sets self.require_input_grad to False iff none of the the parent_nodes (recursively)
    of the inbound_nodes have trainable_weights, in which case the input gradient is not used.
    """
    def have_trainable_weights(parent_nodes):
      """ Recursive function that parses all the parent nodes looking for their trainable_weights.
      Returns False if all the parent nodes have no trainable weights. Otherwise, returns True.
      """
      for parent_node in parent_nodes:
        if parent_node.layer.trainable_weights != []:
          return True
        else:
          return have_trainable_weights(parent_node.parent_nodes)
      return False

    # If require_input_grad has not been computed, then inspect the graph to determine if it is required
    if self.require_input_grad is None:
      self.require_input_grad = False
      for inbound_node in self._inbound_nodes:
        if have_trainable_weights(inbound_node.parent_nodes):
          self.require_input_grad = True
          break


class CustomInitializer(tf.keras.initializers.Initializer):
  """ Base class for Upstride initializers.
  Standard keras initializers may change the underlying distribution parameters depending on tensor shapes.
  To apply them to multidimensional UpStride datatypes, interception mechanisms are implemented. All the internal
  initializers are assumed to be derived from this class to make sure the interception mechanics works correctly.
  """
  pass
