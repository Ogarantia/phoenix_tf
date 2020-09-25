import unittest
import tensorflow as tf

class TestCase(unittest.TestCase):
  """ Base class for unittests containing handy stuff """
  DEFAULT_ERROR_THRESHOLD = 1e-4
  HALF_FLOAT_ERROR_THRESHOLD = 1e-2

  def setup():
    """ Prepares the test module to be executed. Called by unittest.
    Running this single test without preparation may cause a cuDNN crash (dunno why).
    """
    # allow memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    # call a dummy function to get cuDNN handle created
    from .custom_op import upstride_ops
    upstride_ops.wait()


  def assert_and_print(self, test, ref, function="", variable=""):
    """ Prints stuff and raises exception if two tensors are too different
    """
    err = tf.math.reduce_max(tf.math.abs(test - ref))
    threshold=self.HALF_FLOAT_ERROR_THRESHOLD if ref.dtype == tf.float16 else self.DEFAULT_ERROR_THRESHOLD
    self.assertLess(err, threshold, f"Absolute {variable} difference with the reference is too big: {err}")
    print(f'[{function}] Absolute {variable} difference:', err.numpy())
