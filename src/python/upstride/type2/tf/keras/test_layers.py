import unittest
import tensorflow as tf
import numpy as np
from .layers import TF2Upstride, Upstride2TF, Conv2D


def quaternion_mult_naive(tf_op, inputs, kernels):
  c1 = tf_op(inputs[0], kernels[0]) - tf_op(inputs[1], kernels[1]) - tf_op(inputs[2], kernels[2]) - tf_op(inputs[3], kernels[3])
  c2 = tf_op(inputs[0], kernels[1]) + tf_op(inputs[1], kernels[0]) + tf_op(inputs[2], kernels[3]) - tf_op(inputs[3], kernels[2])
  c3 = tf_op(inputs[0], kernels[2]) + tf_op(inputs[2], kernels[0]) + tf_op(inputs[3], kernels[1]) - tf_op(inputs[1], kernels[3])
  c4 = tf_op(inputs[0], kernels[3]) + tf_op(inputs[3], kernels[0]) + tf_op(inputs[1], kernels[2]) - tf_op(inputs[2], kernels[1])
  return [c1, c2, c3, c4]


class TestType2LayersTF2Upstride(unittest.TestCase):
  def test_rgb_in_img(self):
    x = tf.convert_to_tensor(np.zeros((2, 640, 480, 3), dtype=np.float32))
    y = TF2Upstride(strategy='joint')(x)
    self.assertEqual(y.shape, (8, 640, 480, 1))

  def test_gray_in_real_rgb_in_img(self):
    x = tf.convert_to_tensor(np.zeros((2, 640, 480, 3), dtype=np.float32))
    y = TF2Upstride(strategy='grayscale')(x)
    self.assertEqual(y.shape, (8, 640, 480, 1))

  def test_learn_multivector(self):
    x = tf.convert_to_tensor(np.zeros((2, 640, 480, 3), dtype=np.float32))
    y = TF2Upstride(strategy='learned')(x)
    self.assertEqual(y.shape, (8, 640, 480, 3))

  def test_default(self):
    x = tf.convert_to_tensor(np.zeros((2, 640, 480, 3), dtype=np.float32))
    y = TF2Upstride(strategy='')(x)
    self.assertEqual(y.shape, (8, 640, 480, 3))


class TestType2Upstride2TF(unittest.TestCase):
  def test_concat(self):
    x = tf.random.uniform((8, 2, 2, 1), dtype=tf.float32)
    y = Upstride2TF('concat')(x)
    self.assertEqual(y.shape, (2, 2, 2, 4))

  def test_default(self):
    x = tf.random.uniform((8, 2, 2, 1), dtype=tf.float32)
    y = Upstride2TF('default')(x)
    self.assertEqual(y.shape, (2, 2, 2, 1))


class TestType2Conv2D(unittest.TestCase):
  def test_conv2d_tf(self):
    # Run a convolution in tensorflow and in upstride with random inputs and compare the results
    upstride_conv = Conv2D(1, (1, 1), use_bias=False)
    upstride_conv(tf.random.uniform((4, 1, 1, 1)))  # run a first time to init the kernel
    kernels = upstride_conv.kernel  # take the quaternion kernel

    inputs = tf.random.uniform((4, 1, 1, 1))

    # upstride conv
    upstride_output = upstride_conv(inputs)

    def tf_op(i, k):
      # upstride kernel is (O, I, H, W). TF expect (H, W, I, O)
      k = tf.transpose(k, [2, 3, 1, 0])
      output = tf.nn.conv2d(i, k, 1, "SAME")
      return output

    inputs = tf.reshape(inputs, [4, 1, 1, 1, 1])

    tf_output = quaternion_mult_naive(tf_op, inputs, kernels)

    for i in range(4):
      self.assertAlmostEqual(upstride_output.numpy().flatten()[i], [i.numpy().flatten()[0] for i in tf_output][i])

  def test_conv2d_fixed_value(self):
    """ in this function, we test 5 quaternions multiplications
    """
    kernels_factors = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 2, 0, 0], [0, 2, 2, 0], [5, 6, 7, 8]]
    inputs_factors = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 2, 2, 0], [1, 2, 0, 3], [1, 2, 3, 4]]
    expected_outputs = [[1, 0, 0, 0],
                        [0, 2, 0, 0],
                        [-4, 0, 0, -4],
                        [-4, -4, 8, 4],
                        [-60, 12, 30, 24]]

    for i in range(len(kernels_factors)):
      input_r = tf.ones((1, 1, 1, 1)) * inputs_factors[i][0]
      input_i = tf.ones((1, 1, 1, 1)) * inputs_factors[i][1]
      input_j = tf.ones((1, 1, 1, 1)) * inputs_factors[i][2]
      input_k = tf.ones((1, 1, 1, 1)) * inputs_factors[i][3]
      inputs = tf.concat([input_r, input_i, input_j, input_k], axis=0)

      kernel_r = tf.ones((1, 1, 1, 1, 1)) * kernels_factors[i][0]
      kernel_i = tf.ones((1, 1, 1, 1, 1)) * kernels_factors[i][1]
      kernel_j = tf.ones((1, 1, 1, 1, 1)) * kernels_factors[i][2]
      kernel_k = tf.ones((1, 1, 1, 1, 1)) * kernels_factors[i][3]
      kernels = tf.concat([kernel_r, kernel_i, kernel_j, kernel_k], axis=0)

      # define the keras operation, hijack the kernel and run it
      conv_op = Conv2D(1, (1, 1), use_bias=False)
      conv_op(inputs)  # run a first time to init the kernel
      conv_op.kernel = kernels
      outputs = conv_op(inputs)
      self.assertEqual(list(outputs.numpy().flatten()), expected_outputs[i])
