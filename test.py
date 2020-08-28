import sys

sys.path.append("src/python")

import unittest
import numpy as np
import tensorflow as tf

from upstride.type2.tf.keras.test_layers import TestType2LayersTF2Upstride, TestType2Upstride2TF, TestType2Conv2D
from upstride.scalar.tf.keras.test_layers import TestConv2D, TestConv2DGrad, TestDepthwiseConv2D, TestDepthwiseConv2DGrad

if __name__ == "__main__":
  unittest.main()
