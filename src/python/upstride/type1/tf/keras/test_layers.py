import unittest
from upstride.type1.tf.keras import layers
from src.python.upstride.internal.test import setUpModule, Conv2DTestSet, PointwiseConv2DTestSet, DepthwiseConv2DTestSet, DenseTestSet, InputGradientAndTypeTest
from upstride.internal.clifford_product import CliffordProduct


clifford_product = CliffordProduct((2, 0, 0), ["", "12"])
setUpModule()

class Type1Conv2DTestSet(Conv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.Conv2D)


class Type1PointwiseConv2DTestSet(PointwiseConv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.Conv2D)


class Type1DepthwiseConv2DTestSet(DepthwiseConv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.DepthwiseConv2D)


class Type1DenseTestSet(DenseTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.Dense)


class Type1InputGradientAndTypeTest(InputGradientAndTypeTest, unittest.TestCase):
  def setUp(self):
    self.setup(layers)
