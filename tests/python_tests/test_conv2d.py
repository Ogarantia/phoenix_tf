import unittest
import tensorflow as tf
from src.python.upstride.type2.tf.keras.layers import upstride_ops

class TestConv2D(unittest.TestCase):
    def run_conv2d_test(self, img_size=224, filter_size=3, in_channels=3, out_channels=64, padding='VALID', strides=[1, 1], dilations=[1, 1]):
        """ Runs a single convolution and compares the result with TensorFlow output """
        filter = tf.random.uniform((out_channels, in_channels, filter_size, filter_size), dtype=tf.float32, minval=-0.5, maxval=0.5)
        input = tf.random.uniform((1, in_channels, img_size, img_size), dtype=tf.float32, minval=-0.5, maxval=0.5)

        # run upstride convolution
        output_test = upstride_ops.upstride_conv2d(
          input, filter,
          strides=strides,
          padding=padding,
          dilations=dilations,
          data_format='NCHW'
        )

        # run TF convolution on a properly transposed input
        input = tf.transpose(input, [0, 2, 3, 1])
        filter = tf.transpose(filter, [2, 3, 1, 0])
        output_ref = tf.nn.conv2d(
          input, filter,
          strides=strides,
          padding=padding,
          dilations=dilations
        )
        # compare the outputs
        output_ref = tf.transpose(output_ref, [0, 3, 1, 2])
        err = tf.math.reduce_max(tf.math.abs(output_test - output_ref))
        self.assertLess(err, 1e-4, f"Absolute difference with the reference is too big: {err}")
        print('[Conv2DFwd] Absolute difference:', err.numpy())


    def test_conv2d(self):
        self.run_conv2d_test(img_size=224, filter_size=3, in_channels=3, out_channels=64, padding='VALID')
        self.run_conv2d_test(img_size=224, filter_size=4, in_channels=3, out_channels=64, padding='SAME')
        self.run_conv2d_test(img_size=224, filter_size=5, in_channels=3, out_channels=16, strides=[2, 2])
        self.run_conv2d_test(img_size=112, filter_size=6, in_channels=16, out_channels=32, dilations=[2, 2])
        self.run_conv2d_test(img_size=112, filter_size=3, in_channels=32, out_channels=48, padding='SAME', strides=[1, 2], dilations=[3, 4])

class TestConv2DGrad(unittest.TestCase):
    def run_conv2dgrad_test(self, img_size=128, filter_size=3, in_channels=2, out_channels=1, padding='SAME', strides=[1,1], dilations=[1,1]):
      """ Runs a single convolution forward and backward and compares the result with TensorFlow output """
      input = tf.random.uniform((1, in_channels, img_size, img_size), dtype=tf.float32, minval=-0.5, maxval=0.5)
      filter = tf.random.uniform((out_channels, in_channels, filter_size, filter_size), dtype=tf.float32, minval=-0.5, maxval=0.5)

      ## UPSTRIDE
      with tf.GradientTape() as gt:
        gt.watch(filter)
        output_test = upstride_ops.upstride_conv2d(
                  input, filter,
                  strides=strides,
                  padding=padding,
                  dilations=dilations,
                  data_format='NCHW')
      grad_test = gt.gradient(output_test, filter)

      ## TENSORFLOW
      # #                            N  H  W  C
      input_t = tf.transpose(input, [0, 2, 3, 1])
      # #                              H  W  I  O
      filter_t = tf.transpose(filter, [2, 3, 1, 0])
      input_t = tf.identity(input_t)
      filter_t = tf.identity(filter_t)
      with tf.GradientTape(persistent=True) as gt:
        gt.watch(filter_t)
        output_ref_TF = tf.nn.conv2d(
                  input_t, filter_t,
                  strides=strides,
                  padding=padding,
                  dilations=dilations) 

      grad_reference_TF = gt.gradient(output_ref_TF, filter_t)
      #                                              O  I  H  W 
      grad_reference_TF = tf.transpose(grad_reference_TF, [3, 2, 0, 1])

      ## COMPARISONS
      err = tf.math.reduce_max(tf.math.abs(grad_test - grad_reference_TF))
      self.assertLess(err, 1e-4, f"Absolute difference with the reference is too big: {err}")
      print('[Conv2DBwd] Absolute difference:', err.numpy())

    def test_conv2dgrad(self):
      self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='VALID')
      self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='SAME')
      self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, strides=[2, 2])
      self.run_conv2dgrad_test(img_size=32, filter_size=3, in_channels=3, out_channels=8, padding='VALID')
      self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, out_channels=8, padding='SAME')
      self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, out_channels=8, strides=[2, 2])
      self.run_conv2dgrad_test(img_size=224, filter_size=3, in_channels=3, out_channels=48, strides=[2, 2], padding='VALID')
