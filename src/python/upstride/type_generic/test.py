import unittest
import tensorflow as tf

class TestAssert(unittest.TestCase):
  def assert_and_print(self, test, ref, function="", variable=""):
    err = tf.math.reduce_max(tf.math.abs(test - ref))
    self.assertLess(err, 1e-4, f"Absolute {variable} difference with the reference is too big: {err}")
    print(f'[{function}] Absolute {variable} difference:', err.numpy())
