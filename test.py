import sys

sys.path.append("src/python")

import unittest
import numpy as np
import tensorflow as tf

from upstride.type2.tf.keras.test_layers import TestType2Conv2D, TestType2Conv2DBasic, TestType2LayersTF2Upstride, TestType2Upstride2TF, TestType2Conv2D
from upstride.scalar.tf.keras.test_layers import TestDepthwiseConv2D, TestDepthwiseConv2DGrad
from upstride.scalar.tf.keras.test_layers import TestConv2D, TestConv2DGrad


if __name__ == "__main__":
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  unittest.main()
