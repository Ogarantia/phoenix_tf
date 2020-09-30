"""
Quaternion initialization is described here:https://arxiv.org/pdf/1806.04418.pdf
Implementation is done following this open-source:
 https://github.com/Orkis-Research/Quaternion-Convolutional-Neural-Networks-for-End-to-End-Automatic-Speech-Recognition/blob/master/complexnn/init.py

"""


import numpy as np
from numpy.random import RandomState
import tensorflow as tf
import math


def _compute_fans(shape, data_format='channels_last'):
  """Computes the number of input and output units for a weight shape.
  Arguments:
      shape: Integer shape tuple.
      data_format: Image data format to use for convolution kernels.
          Note that all kernels in Keras are standardized on the
          `channels_last` ordering (even when inputs are set
          to `channels_first`).

  Returns:
      A tuple of scalars, `(fan_in, fan_out)`.

  Raises:
      ValueError: in case of invalid `data_format` argument.
  """
  if len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  elif len(shape) in {3, 4, 5}:
    # Assuming convolution kernels (1D, 2D or 3D).
    # TH kernel shape: (depth, input_depth, ...)
    # TF kernel shape: (..., input_depth, depth)
    if data_format == 'channels_first':
      receptive_field_size = np.prod(shape[2:])
      fan_in = shape[1] * receptive_field_size
      fan_out = shape[0] * receptive_field_size
    elif data_format == 'channels_last':
      receptive_field_size = np.prod(shape[:2])
      fan_in = shape[-2] * receptive_field_size
      fan_out = shape[-1] * receptive_field_size
    else:
      raise ValueError('Invalid data_format: ' + data_format)
  else:
    # No specific assumptions.
    fan_in = math.sqrt(np.prod(shape))
    fan_out = math.sqrt(np.prod(shape))

  return fan_in, fan_out


class QInitializer(tf.keras.initializers.Initializer):
  def __init__(self, shape, fan_in, fan_out, criterion, seed):
    """
    Args:
        shape: shape of the weight tensor
        fan_in: number of input features/feature maps
        fan_out: number of output features/feature maps
        criterion: initialization criterion for modulus of weight matrix ('glorot', 'he')
        seed: Random seed value
    """
    self.shape = shape
    self.fan_in = fan_in
    self.fan_out = fan_out
    self.criterion = criterion
    self.seed = seed
    self.modulus, self.phase, self.v_i, self.v_j, self.v_k = self.initialize()

  def initialize(self):
    # Quaternion operations start here
    if self.criterion == 'glorot':
      s = 1. / np.sqrt(2 * (self.fan_in + self.fan_out))
    elif self.criterion == 'he':
      s = 1. / np.sqrt(2 * self.fan_in)
    else:
      raise ValueError('Invalid criterion: ' + self.criterion)

    # Generating randoms and purely imaginary quaternions :
    number_of_weights = np.prod(self.shape)

    v_i = np.random.uniform(0.0, 1.0, number_of_weights)
    v_j = np.random.uniform(0.0, 1.0, number_of_weights)
    v_k = np.random.uniform(0.0, 1.0, number_of_weights)

    # Make these purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
      norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
      v_i[i] /= norm
      v_j[i] /= norm
      v_k[i] /= norm
    v_i = v_i.reshape(self.shape)
    v_j = v_j.reshape(self.shape)
    v_k = v_k.reshape(self.shape)

    rng = RandomState(self.seed)
    modulus = rng.rayleigh(scale=s, size=self.shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=self.shape)

    return modulus, phase, v_i, v_j, v_k

  def __call__(self, shape, dtype=None):
    """ Returns the weight for the full multivector embedded in the shape provided.
    """
    weight_blade = []
    weight_blade.append(self.modulus * np.cos(self.phase))
    weight_blade.append(self.modulus * self.v_i * np.sin(self.phase))
    weight_blade.append(self.modulus * self.v_j * np.sin(self.phase))
    weight_blade.append(self.modulus * self.v_k * np.sin(self.phase))
    weight = np.stack(weight_blade, axis=0)
    if dtype is not None:
      dtype = str(dtype).split("'")[1] # casting dtype from TF type to agnostic type
      weight.astype(dtype)
    return weight


class QInitializerConv(QInitializer):
  def __init__(self, kernel_size, input_dim, weight_dim, nb_filters=None,
               criterion='he', seed=None):
    """
    Args:
        kernel_size: Convolution kernel shape
        input_dim: number of input features/feature maps
        weight_dim:
        nb_filters: number of output features/feature maps
        criterion: initialization criterion for modulus of weight matrix ('glorot', 'he')
        seed: Random seed value

        `weight_dim` is used as a parameter for sanity check
        as we should not pass an integer as kernel_size when
        the weight dimension is >= 2.
        nb_filters == 0 if weights are not convolutional (matrix instead of filters)
        then in such a case, weight_dim = 2.
        (in case of 2D input):
            nb_filters == None and len(kernel_size) == 2 and_weight_dim == 2
        conv1D: len(kernel_size) == 1 and weight_dim == 1
        conv2D: len(kernel_size) == 2 and weight_dim == 2
        conv3d: len(kernel_size) == 3 and weight_dim == 3
    """
    assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
    if nb_filters is not None:
      kernel_shape = (nb_filters, int(input_dim)) + tuple(kernel_size)
    else:
      kernel_shape = (int(input_dim), kernel_size[-1])

    fan_in, fan_out = _compute_fans(
        tuple(kernel_size) + (input_dim, nb_filters)
    )
    super(QInitializerConv, self).__init__(kernel_shape, fan_in, fan_out, criterion, seed)


class QInitializerDense(QInitializer):
  def __init__(self, shape, criterion='he', seed=None):
    """
    Args:
        shape: Shape of the weight tensor
        criterion: initialization criterion for modulus of weight matrix ('glorot', 'he')
        seed: Random seed value
    """
    fan_in = shape[0]
    fan_out = shape[1]
    super(QInitializerDense, self).__init__(shape, fan_in, fan_out, criterion, seed)


def is_type2_init(init_type):
  """
  Determine whether it is a is_type2 initialization or not
  Args:
      init_type: str or tf.keras.initializers.Initializer, initialization type for upstride is_type2, either
      'up2_init_he'  or 'up2_init_glorot' for real valued initialization should be tensorflow
  """

  if isinstance(init_type, str) and 'up2_init' in init_type:
    return True

  return False
