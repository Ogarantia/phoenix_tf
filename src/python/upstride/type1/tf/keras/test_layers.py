from src.python.upstride.internal.test import setUpModule, Conv2DTestSet, PointwiseConv2DTestSet, DepthwiseConv2DTestSet, DenseTestSet, InputGradientAndTypeTest
from upstride.internal.clifford_product import CliffordProduct
from upstride.type1.tf.keras.layers import DepthwiseConv2D, Conv2D, Dense


clifford_product = CliffordProduct((2, 0, 0), ["", "12"])
setUpModule()


class TestSetType1Conv2D(Conv2DTestSet):
  def setup(self):
    super().setup(clifford_product, Conv2D)


class TestSetType1PointwiseConv2D(PointwiseConv2DTestSet):
  def setup(self):
    super().setup(clifford_product, Conv2D)


class TestSetType1DepthwiseConv2D(DepthwiseConv2DTestSet):
  def setup(self):
    super().setup(clifford_product, DepthwiseConv2D)


class TestSetType1Dense(DenseTestSet):
  def setup(self):
    super().setup(clifford_product, Dense)


class TestSetType1InputGradientAndType(InputGradientAndTypeTest):
  def setup(self):
    from upstride.type1.tf.keras import layers
    super().setup(layers)
