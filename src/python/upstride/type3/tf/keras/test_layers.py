import unittest
from . import layers
from ....type_generic.test import setUpModule, Conv2DTestSet, DepthwiseConv2DTestSet, DenseTestSet
from ....type_generic.clifford_product import CliffordProduct


clifford_product = CliffordProduct((3, 0, 0), ["", "1", "2", "3", "12", "13", "23", "123"])
setUpModule()

class Type3Conv2DTestSet(Conv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.Conv2D)


class Type3DepthwiseConv2DTestSet(DepthwiseConv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.DepthwiseConv2D)


class Type3DenseTestSet(DenseTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.Dense)