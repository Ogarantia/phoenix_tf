import numpy as np

class ConvInitializer:
  def compute_fans(self, shape, groups):
    # assuming [N, O, I, H, W] kernel layout
    assert len(shape) == 5
    # for grouped convolutions, the convention is [N, g * Og, I, H, W]
    self.fan_in = shape[2] * np.prod(shape[3:])
    self.fan_out = shape[1] * np.prod(shape[3:]) // groups


class DepthwiseConvInitializer:
  def compute_fans(self, shape, depth_multiplier):
    # assuming [N, O, I, H, W] kernel layout
    assert len(shape) == 5 and shape[2] == 1
    # for depthwise convolutions, the convention is [N, depth_multiplier * I, 1, H, W]
    self.fan_in = np.prod(shape[3:])
    self.fan_out = depth_multiplier * np.prod(shape[3:])


class DenseInitializer:
  def compute_fans(self, shape):
    # assuming [N, I, O] kernel layout
    assert len(shape) == 3
    self.fan_in = shape[1]
    self.fan_out = shape[2]