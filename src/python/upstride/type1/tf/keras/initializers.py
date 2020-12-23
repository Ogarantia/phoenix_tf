
import numpy as np
import tensorflow as tf
from upstride.internal.layers import CustomInitializer
from upstride.internal.initializers import ConvInitializer, DepthwiseConvInitializer, DenseInitializer

class CInitializer(CustomInitializer):
  def __init__(self, criterion, seed=None):
    """
    Constructs a Complex initializer.
    Inspired by - Deep Complex Networks - https://arxiv.org/pdf/1705.09792.pdf (section 3.6 page 6)
    Args:
        :criterion:  magnitude initializer setting ('up1_init_glorot', 'up1_init_he')
        :seed:       Random seed value
    """
    self.fan_in = None
    self.fan_out = None
    self.criterion = criterion
    self.seed = seed

  def __call__(self, shape, eps=0.0001, dtype=None):
    """ Returns the weight for the full multivector embedded in the shape provided.
    """
    # ensure fan_in and fan_out are defined
    if self.fan_in is None or self.fan_out is None:
      raise ValueError('CInitializer is not set up: fan_in and/or fan_out not set')

    # compute Rayleigh distribution scale
    if self.criterion == 'up1_init_glorot':
      scale = 1. / (self.fan_in + self.fan_out)
    elif self.criterion == 'up1_init_he':
      scale = 1. / self.fan_in
    else:
      raise ValueError('Invalid criterion: ' + self.criterion)

    # drop first shape dimension
    assert shape[0] == 2
    shape = shape[1:]

    # init magnitude and phase
    rng = np.random.RandomState(self.seed)
    magnitude = rng.rayleigh(scale=scale, size=shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=shape)

    # generate complex tensor
    weights = [
      magnitude * np.cos(phase),
      magnitude * np.sin(phase)
    ]

    # concatenate and return
    weight = np.stack(weights, axis=0)
    return tf.convert_to_tensor(weight, dtype=dtype)

  def get_config(self):
    """Returns the configuration of the initializer as a JSON-serializable dict.

    Returns:
      A JSON-serializable Python dict.
    """
    return {'criterion': self.criterion,
            'seed': self.seed}


@tf.keras.utils.register_keras_serializable("upstride_type1")
class CInitializerConv(CInitializer, ConvInitializer):
  def __init__(self, criterion, groups, seed=None):
    CInitializer.__init__(self, criterion, seed)
    self.groups = groups

  def __call__(self, shape, dtype=None):
    assert shape[0] == 2
    self.compute_fans(shape, self.groups)
    return super().__call__(shape, dtype=dtype)

  def get_config(self):
    config = super().get_config()
    config['groups'] = self.groups
    return config


@tf.keras.utils.register_keras_serializable("upstride_type1")
class CInitializerDepthwiseConv(CInitializer, DepthwiseConvInitializer):
  def __init__(self, criterion, depth_multiplier, seed=None):
    CInitializer.__init__(self, criterion, seed)
    self.depth_multiplier = depth_multiplier

  def __call__(self, shape, dtype=None):
    assert shape[0] == 2
    self.compute_fans(shape, self.depth_multiplier)
    return super().__call__(shape, dtype=dtype)

  def get_config(self):
    config = super().get_config()
    config['depth_multiplier'] = self.depth_multiplier
    return config


@tf.keras.utils.register_keras_serializable("upstride_type1")
class CInitializerDense(CInitializer, DenseInitializer):
  def __call__(self, shape, dtype=None):
    assert shape[0] == 2
    self.compute_fans(shape)
    return super().__call__(shape, dtype=dtype)


def is_type1_init(init_type):
  """
  Determine whether it is a is_type1 initialization or not
  Args:
      init_type: str or tf.keras.initializers.Initializer, initialization type for upstride is_type1, either
      'up1_init_he'  or 'up1_init_glorot' for real valued initialization should be tensorflow
  """
  return isinstance(init_type, str) and init_type.startswith('up1_init')
