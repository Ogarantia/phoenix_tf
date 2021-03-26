import numpy as np

class ConvInitializer:
  def compute_fans(self, shape, groups, data_format):
    # assuming [N, O, I, H, W] kernel layout
    assert len(shape) == 5
    if data_format == 'channels_first':
      # for grouped convolutions, the convention is [N, g * Og, I, H, W]
      self.fan_in = shape[2] * np.prod(shape[3:])
      self.fan_out = shape[1] * np.prod(shape[3:]) // groups
    elif data_format == 'channels_last':
      # for grouped convolutions, the convention is [N, g * Og, H, W, I]
      self.fan_in = shape[-1] * np.prod(shape[2:-1])
      self.fan_out = shape[2] * np.prod(shape[2:-1]) // groups
    else:
      raise ValueError('[compute_fans] Invalid data_format: ' + data_format)


class DepthwiseConvInitializer:
  def compute_fans(self, shape, depth_multiplier, data_format):
    # assuming [N, O, I, H, W] kernel layout
    assert len(shape) == 5
    if data_format == 'channels_first':
      assert shape[2] == 1
      # for depthwise convolutions, the convention is [N, depth_multiplier * I, 1, H, W]
      self.fan_in = np.prod(shape[3:])
      self.fan_out = depth_multiplier * np.prod(shape[3:])
    elif data_format == 'channels_last':
      assert shape[-1] == 1
      # for depthwise convolutions, the convention is [N, depth_multiplier * I, H, W, 1]
      self.fan_in = np.prod(shape[2:-1])
      self.fan_out = depth_multiplier * np.prod(shape[2:-1])
    else:
      raise ValueError('[compute_fans] Invalid data_format: ' + data_format)


class DenseInitializer:
  def compute_fans(self, shape):
    # assuming [N, I, O] kernel layout
    assert len(shape) == 3
    self.fan_in = shape[1]
    self.fan_out = shape[2]