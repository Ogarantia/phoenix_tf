import unittest
import tensorflow as tf
import numpy as np
from .layers import TF2Upstride, Upstride2TF, Conv2D


def quaternion_mult_naive(tf_op, inputs, kernels, bias=(0, 0, 0, 0)):
  c1 = tf_op(inputs[0], kernels[0]) - tf_op(inputs[1], kernels[1]) - tf_op(inputs[2], kernels[2]) - tf_op(inputs[3], kernels[3]) + bias[0]
  c2 = tf_op(inputs[0], kernels[1]) + tf_op(inputs[1], kernels[0]) + tf_op(inputs[2], kernels[3]) - tf_op(inputs[3], kernels[2]) + bias[1]
  c3 = tf_op(inputs[0], kernels[2]) + tf_op(inputs[2], kernels[0]) + tf_op(inputs[3], kernels[1]) - tf_op(inputs[1], kernels[3]) + bias[2]
  c4 = tf_op(inputs[0], kernels[3]) + tf_op(inputs[3], kernels[0]) + tf_op(inputs[1], kernels[2]) - tf_op(inputs[2], kernels[1]) + bias[3]
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


class TestType2Conv2DBasic(unittest.TestCase):
  """ Basic quaternion convolution sanity check """

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
      self.assertAlmostEqual(upstride_output.numpy().flatten()[i], [i.numpy().flatten()[0] for i in tf_output][i], 6)

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


class TestType2Conv2D(unittest.TestCase):
  """ Exhaustive quaternion convolution testing """

  def run_test(self, img_size=224, filter_size=3, in_channels=3, out_channels=64, padding='SAME', strides=[1, 1], dilations=[1, 1], use_bias=False):
    # initialize inputs
    tf.random.set_seed(45)
    inputs = [tf.random.uniform((1, img_size, img_size, in_channels), dtype=tf.float32, minval=-5, maxval=5) for _ in range(4)]
    upstride_inputs = tf.concat([tf.transpose(_, [0, 3, 1, 2]) for _ in inputs], axis=0)

    # set up upstride convolution operation
    upstride_conv = Conv2D(filters=out_channels, kernel_size=filter_size, strides=strides, padding=padding, dilation_rate=dilations,
                           use_bias=use_bias, bias_initializer=tf.keras.initializers.RandomUniform())

    upstride_conv(upstride_inputs)  # runs a first time to initialize the kernel
    kernels = upstride_conv.kernel  # copies the quaternion kernel
    bias = upstride_conv.bias

    # compute ground truth (reference) output
    def tf_op(i, k):
      # upstride kernel is (O, I, H, W). TF expects (H, W, I, O)
      k = tf.transpose(k, [2, 3, 1, 0])
      output = tf.nn.conv2d(i, k, strides=strides, padding=padding, dilations=dilations)
      return output
    
    reference_outputs = quaternion_mult_naive(tf_op, inputs, kernels, bias if use_bias else (0, 0, 0, 0))
    reference_outputs = tf.concat([tf.transpose(_, [0, 3, 1, 2]) for _ in reference_outputs], axis=0)

    # compute test outputs
    upstride_outputs = upstride_conv(upstride_inputs)

    # compare things
    error = tf.math.reduce_max(tf.math.abs(reference_outputs - upstride_outputs))
    self.assertLess(error, 1e-4, f"Absolute d_input difference compared to the reference is too big: {error}")

  def test_upstride_inputs_backprop(self):
    try:
      tf.keras.backend.set_image_data_format('channels_first')  # FIXME We should find a proper way to pass 'channels_first'
      self.run_test(img_size=1, filter_size=1, in_channels=1, out_channels=1, padding='SAME', strides=[1, 1], dilations=[1, 1])
      self.run_test(img_size=1, filter_size=1, in_channels=1, out_channels=1, padding='SAME', strides=[1, 1], dilations=[1, 1], use_bias=True)
      self.run_test(img_size=4, filter_size=2, in_channels=1, out_channels=1, padding='SAME', strides=[1, 1], dilations=[1, 1])
      self.run_test(img_size=4, filter_size=2, in_channels=1, out_channels=1, padding='SAME', strides=[1, 1], dilations=[1, 1], use_bias=True)
      self.run_test(img_size=8, filter_size=3, in_channels=2, out_channels=2, padding='VALID')
      self.run_test(img_size=8, filter_size=3, in_channels=2, out_channels=2, padding='VALID', use_bias=True)
      self.run_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='VALID')
      self.run_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='VALID', use_bias=True)
      self.run_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='SAME')
      self.run_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='SAME', use_bias=True)
      self.run_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, strides=[2, 2])
      self.run_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, strides=[2, 2], use_bias=True)
      self.run_test(img_size=32, filter_size=3, in_channels=3, out_channels=8, padding='VALID')
      self.run_test(img_size=32, filter_size=3, in_channels=3, out_channels=8, padding='VALID', use_bias=True)
      self.run_test(img_size=32, filter_size=4, in_channels=3, out_channels=8, padding='SAME')
      self.run_test(img_size=32, filter_size=4, in_channels=3, out_channels=8, padding='SAME', use_bias=True)
      self.run_test(img_size=32, filter_size=4, in_channels=3, out_channels=8, strides=[2, 2])
      self.run_test(img_size=32, filter_size=4, in_channels=3, out_channels=8, strides=[2, 2], use_bias=True)
      self.run_test(img_size=224, filter_size=3, in_channels=3, out_channels=48, strides=[2, 2], padding='VALID')
      self.run_test(img_size=224, filter_size=3, in_channels=3, out_channels=48, strides=[2, 2], padding='VALID', use_bias=True)
    finally:
      tf.keras.backend.set_image_data_format('channels_last')  # FIXME We should find a proper way to pass 'channels_last'
