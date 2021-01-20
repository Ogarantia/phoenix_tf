import tensorflow as tf
import pytest
from .test import Conv2DTestBase
import platform

reason = "Too large for JetsonNano"
is_jetson_nano = "tegra" in platform.uname().release

@pytest.mark.slow
class PointwiseConv2DExhaustiveTestSet(Conv2DTestBase):
  """ Exhaustive test set for pointwise Conv2D operation
  """
  def setup(self, clifford_product, test_op_class):
    super().setup(clifford_product, tf.keras.layers.Conv2D, test_op_class, 'kernel_initializer', tf.float32)
    self.DEFAULT_BATCH_SIZE = 4

  @pytest.mark.parametrize('img_side_length, input_channels, output_channels', [
    pytest.param
      (112, 8, 8,
      marks=pytest.mark.skipif(is_jetson_nano, reason=reason)), # expanded_conv_project
    pytest.param
      (112, 8, 48,
      marks=pytest.mark.skipif(is_jetson_nano, reason=reason)), # block_1_expand
    pytest.param
      (56, 48, 8,
      marks=pytest.mark.skipif(is_jetson_nano, reason=reason)), # block_{1, 2}_project
    pytest.param
      (56, 8, 48,
      marks=pytest.mark.skipif(is_jetson_nano, reason=reason)), # block_{2, 3}_expand
    (28, 48, 8),          # block_{3, 4, 5}_project
    (28, 8, 48),          # block_{4, 5, 6}_expand
    (14, 48, 16),         # block_6_project
    (14, 16, 96),         # block_{7, 8, 9, 10}_expand
    (14, 96, 16),         # block_{7, 8, 9}_project
    (14, 96, 24),         # block_10_project
    (14, 24, 144),        # block_{11, 12, 13}_expand
    (14, 144, 24),        # block_{11, 12}_project
    (7, 144, 40),         # block_13_project
    (7, 40, 240),         # block_{14, 15, 16}_expand
    (7, 240, 40),         # block_{14, 15}_project
    (7, 240, 80),         # block_16_project
    (7, 80, 320),         # Conv_1
  ])
  def test_pointwise_mobilenetv2_224_configs(self, img_side_length, input_channels, output_channels):
    """ PointwiseConv2D test exploring convolution configurations used in MobileNetV2 for images 224 x 224 x 3 """
    self.run_test_instance(
      test_shape=(self.DEFAULT_BATCH_SIZE, img_side_length, img_side_length, input_channels),
      filters=output_channels,
      kernel_size=1,
      use_bias=False
    )
