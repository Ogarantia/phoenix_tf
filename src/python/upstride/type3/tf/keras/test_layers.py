import unittest
from src.python.upstride.internal.test import setUpModule, Conv2DTestSet, PointwiseConv2DTestSet, DepthwiseConv2DTestSet, DenseTestSet, InputGradientAndTypeTest
from upstride.internal.clifford_product import CliffordProduct
from upstride.type3.tf.keras.layers import DepthwiseConv2D, Conv2D, Dense
# FIXME: temporary patch to resolve the issue with clean_up in test.py
from upstride.type3.tf.keras import layers


clifford_product = CliffordProduct((3, 0, 0), ["", "1", "2", "3", "12", "13", "23", "123"])
setUpModule()


class TestSetType3Conv2D(Conv2DTestSet):
  def setup(self):
    super().setup(clifford_product, Conv2D)


class TestSetType3PointwiseConv2D(PointwiseConv2DTestSet):
  def setup(self):
    super().setup(clifford_product, Conv2D)


class TestSetType3DepthwiseConv2D(DepthwiseConv2DTestSet):
  def setup(self):
    super().setup(clifford_product, DepthwiseConv2D)


class TestSetType3Dense(DenseTestSet):
  def setup(self):
    super().setup(clifford_product, Dense)


class TestSetType3InputGradientAndType(InputGradientAndTypeTest):
  def setup(self):
    from upstride.type3.tf.keras import layers
    super().setup(layers)
