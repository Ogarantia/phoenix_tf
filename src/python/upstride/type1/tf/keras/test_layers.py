import unittest
import tensorflow as tf
import numpy as np
from . import layers
from ....type_generic.test import setUpModule, Conv2DTestSet
from ....type_generic.clifford_product import CliffordProduct


clifford_product = CliffordProduct((2, 0, 0), ["", "12"])
setUpModule()

class Conv2DTestSet(Conv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, layers.Conv2D)