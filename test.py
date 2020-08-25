import sys

sys.path.append("src/python")

import unittest
import numpy as np
import tensorflow as tf

from upstride.type2.tf.keras.test_layers import TestType2LayersTF2Upstride, TestType2Upstride2TF, TestType2Conv2D
from tests.python_unittests.test_conv2d import TestConv2D, TestConv2DGrad
#from tests.python_unittests.test_depthwiseconv2d import TestDepthwiseConv2D, TestDepthwiseConv2DGrad

if __name__ == "__main__":
  unittest.main()
