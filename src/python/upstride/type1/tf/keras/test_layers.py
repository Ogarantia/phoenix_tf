import unittest
from . import layers
from ....type_generic.test import setUpModule, Conv2DTestSet, DepthwiseConv2DTestSet, DenseTestSet
from ....type_generic.clifford_product import CliffordProduct


clifford_product = CliffordProduct((2, 0, 0), ["", "12"])
setUpModule()

class Type1Conv2DTestSet(Conv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.Conv2D)


class Type1DepthwiseConv2DTestSet(DepthwiseConv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.DepthwiseConv2D)


class Type1DenseTestSet(DenseTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.Dense)