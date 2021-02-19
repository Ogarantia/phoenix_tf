import sys, os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pytest
import tensorflow as tf

# import testcases
from src.python.upstride.internal.clifford_product import TestCliffordProduct
from src.python.upstride.type0.tf.keras.test_layers import Type0Conv2DTestSet, Type0PointwiseConv2DTestSet, Type0DepthwiseConv2DTestSet, Type0DenseTestSet, Type0InputGradientAndTypeTest
from src.python.upstride.type1.tf.keras.test_layers import *
from src.python.upstride.type2.tf.keras.test_layers import Type2Conv2DTestSet, Type2PointwiseConv2DTestSet, Type2DepthwiseConv2DTestSet, Type2DenseTestSet, Type2InputGradientAndTypeTest
from src.python.upstride.type3.tf.keras.test_layers import *

# import exhaustive testcases
from src.python.upstride.type2.tf.keras.test_layers import TestSetType2PointwiseConv2DExhaustive

# FIXME: importing old test code. Consider removing.
from src.python.upstride.type0.tf.keras.test_layers import TestDepthwiseConv2D, TestConv2D, TestConv2DGrad, TestDense
from src.python.upstride.type2.tf.keras.test_layers import TestType2Conv2D, TestType2Conv2DBasic, TestType2Dense, TestType2DepthwiseConv2D
from src.python.upstride.type2.tf.keras.test_layers import TestType2LayersTF2Upstride, TestType2Upstride2TF

if __name__ == "__main__":
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  pytest.main(sys.argv)   # pass main arguments, including this filename
  layers.clean_up()