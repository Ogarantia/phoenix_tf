"""
Quaternion initialization is described here:https://arxiv.org/pdf/1806.04418.pdf
Implementation is done following this open-source:
 https://github.com/Orkis-Research/Quaternion-Convolutional-Neural-Networks-for-End-to-End-Automatic-Speech-Recognition/blob/master/complexnn/init.py

"""


import numpy as np
import tensorflow as tf
from upstride.internal.layers import CustomInitializer


class QInitializer(CustomInitializer):
  def __init__(self, criterion, seed=None):
    """
    Constructs a quaternion initializer.
    Inspired by https://arxiv.org/pdf/1806.04418.pdf (section 3.4)
    Args:
        :criterion:  magnitude initializer setting ('up2_init_glorot', 'up2_init_he')
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
      raise ValueError('QInitializer is not set up: fan_in and/or fan_out not set')

    # compute Rayleigh distribution scale
    if self.criterion == 'up2_init_glorot':
      scale = 1. / np.sqrt(2 * (self.fan_in + self.fan_out))
    elif self.criterion == 'up2_init_he':
      scale = 1. / np.sqrt(2 * self.fan_in)
    else:
      raise ValueError('Invalid criterion: ' + self.criterion)

    # drop first shape dimension
    assert shape[0] == 4
    shape = shape[1:]

    # init magnitude and phase
    rng = np.random.RandomState(self.seed)
    magnitude = rng.rayleigh(scale=scale, size=shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=shape)

    # generate imaginary quaternions
    # https://arxiv.org/pdf/1806.04418.pdf, section 3.4
    v_i = np.random.uniform(eps, 1.0, shape)
    v_j = np.random.uniform(eps, 1.0, shape)
    v_k = np.random.uniform(eps, 1.0, shape)

    # normalize the imaginary quaternion tensor
    norm = np.sqrt(v_i ** 2 + v_j ** 2 + v_k ** 2)
    v_i /= norm
    v_j /= norm
    v_k /= norm

    # generate quaternion tensor
    sin = np.sin(phase)
    weights = [
      magnitude * np.cos(phase),
      magnitude * v_i * sin,
      magnitude * v_j * sin,
      magnitude * v_k * sin
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


@tf.keras.utils.register_keras_serializable("upstride_type2")
class QInitializerConv(QInitializer):
  def __init__(self, criterion, groups, seed=None):
    super().__init__(criterion, seed)
    self.groups = groups
  
  def __call__(self, shape, dtype=None):
    # assuming [4, O, I, H, W] kernel layout
    assert len(shape) == 5 and shape[0] == 4
    # for grouped convolutions, the convention is [4, g * Og, I, H, W]
    self.fan_in = shape[2] * np.prod(shape[3:])
    self.fan_out = shape[1] * np.prod(shape[3:]) // self.groups
    return super().__call__(shape, dtype=dtype)

  def get_config(self):
    config = super().get_config()
    config['groups'] = self.groups
    return config


class QInitializerDepthwiseConv(QInitializer):
  def __init__(self, criterion, depth_multiplier, seed=None):
    super().__init__(criterion, seed)
    self.depth_multiplier = depth_multiplier

  def __call__(self, shape, dtype=None):
    # assuming [4, O, I, H, W] kernel layout
    assert len(shape) == 5 and shape[0] == 4
    # for depthwise convolutions, the convention is [4, depth_multiplier * I, 1, H, W]
    self.fan_in = np.prod(shape[3:])
    self.fan_out = self.depth_multiplier * np.prod(shape[3:])
    return super().__call__(shape, dtype=dtype)

  def get_config(self):
    config = super().get_config()
    config['depth_multiplier'] = self.depth_multiplier
    return config


@tf.keras.utils.register_keras_serializable("upstride_type2")
class QInitializerDense(QInitializer):
  def __call__(self, shape, dtype=None):
    # assuming [4, I, O] kernel layout
    assert len(shape) == 3 and shape[0] == 4
    self.fan_in = shape[1]
    self.fan_out = shape[2]
    return super().__call__(shape, dtype=dtype)


def is_type2_init(init_type):
  """
  Determine whether it is a is_type2 initialization or not
  Args:
      init_type: str or tf.keras.initializers.Initializer, initialization type for upstride is_type2, either
      'up2_init_he'  or 'up2_init_glorot' for real valued initialization should be tensorflow
  """
  return isinstance(init_type, str) and init_type.startswith('up2_init')
