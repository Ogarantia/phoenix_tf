import sys, os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append("src/python")

import unittest
import tensorflow as tf

# import testcases
from src.python.upstride.type_generic.clifford_product import TestCliffordProduct
from upstride.type0.tf.keras.test_layers import Type0Conv2DTestSet, Type0PointwiseConv2DTestSet, Type0DepthwiseConv2DTestSet, Type0DenseTestSet
from upstride.type1.tf.keras.test_layers import *
from upstride.type2.tf.keras.test_layers import Type2Conv2DTestSet, Type2PointwiseConv2DTestSet, Type2DepthwiseConv2DTestSet, Type2DenseTestSet
from upstride.type3.tf.keras.test_layers import *

# FIXME: importing old test code. Consider removing.
from upstride.type0.tf.keras.test_layers import TestDepthwiseConv2D, TestConv2D, TestConv2DGrad, TestDense
from upstride.type2.tf.keras.test_layers import TestType2Conv2D, TestType2Conv2DBasic, TestType2Dense, TestType2DepthwiseConv2D
from upstride.type2.tf.keras.test_layers import TestType2LayersTF2Upstride, TestType2Upstride2TF

if __name__ == "__main__":
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  unittest.main()
