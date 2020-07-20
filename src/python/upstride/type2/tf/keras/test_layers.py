import unittest
import tensorflow as tf
import numpy as np
from .layers import TF2Upstride

class TestType2LayersTF2Upstride(unittest.TestCase):
    def test_rgb_in_img(self):
        x = tf.convert_to_tensor(np.zeros((2, 640, 480, 3), dtype=np.float32))
        y = TF2Upstride(strategy='joint')(x)
        self.assertEqual(y.shape, (8, 640, 480, 1))



