import sys, os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pytest
import tensorflow as tf

# import testcases
from src.python.upstride.internal.clifford_product import TestCliffordProduct
from src.python.upstride.type0.tf.keras.test_layers import TestSetType0Conv2D, TestSetType0PointwiseConv2D, TestSetType0DepthwiseConv2D, TestSetType0Dense, TestSetType0InputGradientAndType
from src.python.upstride.type1.tf.keras.test_layers import TestSetType1Conv2D, TestSetType1PointwiseConv2D, TestSetType1DepthwiseConv2D, TestSetType1Dense, TestSetType1InputGradientAndType
from src.python.upstride.type2.tf.keras.test_layers import TestSetType2Conv2D, TestSetType2PointwiseConv2D, TestSetType2DepthwiseConv2D, TestSetType2Dense, TestSetType2InputGradientAndType
from src.python.upstride.type3.tf.keras.test_layers import TestSetType3Conv2D, TestSetType3PointwiseConv2D, TestSetType3DepthwiseConv2D, TestSetType3Dense, TestSetType3InputGradientAndType

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
  code = pytest.main(sys.argv)   # pass main arguments, including this filename
  # FIXME: clean_up throws errors when imported from src.python.upstride.internal.layers
  # or when layers are imported from src.python.upstride.typeX.tf.keras
  from src.python.upstride.type3.tf.keras.test_layers import layers
  layers.clean_up()

  exit(code)