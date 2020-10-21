import sys

sys.path.append("src/python")

import unittest
import numpy as np
import tensorflow as tf

from upstride.type2.tf.keras.test_layers import TestType2Conv2D, TestType2Conv2DBasic, TestType2Dense, TestType2DepthwiseConv2d
from upstride.type2.tf.keras.test_layers import TestType2LayersTF2Upstride, TestType2Upstride2TF
from upstride.type0.tf.keras.test_layers import TestDepthwiseConv2D, TestDepthwiseConv2DGrad, TestConv2D, TestConv2DGrad, TestDense


if __name__ == "__main__":
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  unittest.main()
