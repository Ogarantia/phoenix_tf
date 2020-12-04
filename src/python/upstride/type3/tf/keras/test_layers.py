import unittest
from upstride.type3.tf.keras import layers
from upstride.internal.test import setUpModule, Conv2DTestSet, PointwiseConv2DTestSet, DepthwiseConv2DTestSet, DenseTestSet
from upstride.internal.clifford_product import CliffordProduct


clifford_product = CliffordProduct((3, 0, 0), ["", "1", "2", "3", "12", "13", "23", "123"])
setUpModule()

class Type3Conv2DTestSet(Conv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.Conv2D)


class Type3PointwiseConv2DTestSet(PointwiseConv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.Conv2D)


class Type3DepthwiseConv2DTestSet(DepthwiseConv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.DepthwiseConv2D)


class Type3DenseTestSet(DenseTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.Dense)