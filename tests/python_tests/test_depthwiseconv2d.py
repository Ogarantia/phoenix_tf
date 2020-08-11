import unittest
import tensorflow as tf
from src.python.upstride.type2.tf.keras.layers import upstride_ops, DepthwiseConv2D

class TestDeptwiseConv2D(unittest.TestCase):
    def run_conv2d_test(self, img_size=9, filter_size=3, in_channels=64, padding='VALID', strides=[1,1, 1, 1], dilations=[1, 1]):
        """ Runs a single convolution and compares the result with TensorFlow output """
        #                          N      C            H          W
        input = tf.random.uniform((1, in_channels, img_size, img_size), dtype=tf.float32)
        #                               H           W            
        filter = tf.random.uniform((filter_size, filter_size, in_channels, 1), dtype=tf.float32)

        # run TF convolution on a properly transposed input
        #                            N  H  W  C
        input = tf.transpose(input, [0, 2, 3, 1])
        # filter = tf.transpose(filter, [2, 3, 1, 0])
        output_ref = tf.nn.depthwise_conv2d(
          input, filter,
          strides=strides,
          padding=padding,
          dilations=dilations
        )

        #re-transpose for OneDNN to  N  C  H  W 
        input = tf.transpose(input, [0, 3, 1, 2])
        #                              1  G   H  W
        filter = tf.transpose(filter, [3, 2, 0, 1])
        # DepthwiseConv2D using conv2D with groups == input channels
        output_test = upstride_ops.upstride_conv2d(
          input, filter,
          strides=strides,
          padding=padding,
          dilations=dilations,
          data_format='NCHW',
          groups=in_channels
        )
        output_ref = tf.transpose(output_ref, [0,3,1,2])
        
        ## COMPARISONS
        err = tf.math.reduce_max(tf.math.abs(output_test - output_ref))
        self.assertLess(err, 1e-4, f"Absolute difference with the reference is too big: {err}")
        print('[DepthwiseConv2D] Absolute difference:', err.numpy())

    def test_conv2d(self):
        self.run_conv2d_test(img_size=224, filter_size=3, in_channels=64, padding='VALID')
        self.run_conv2d_test(img_size=224, filter_size=4, in_channels=64, padding='SAME')
        self.run_conv2d_test(img_size=224, filter_size=3, in_channels=64, strides=[1, 2, 2, 1])
        self.run_conv2d_test(img_size=224, filter_size=3, in_channels=32, padding='VALID')
        self.run_conv2d_test(img_size=224, filter_size=4, in_channels=32, padding='SAME')
        self.run_conv2d_test(img_size=112, filter_size=6, in_channels=32, dilations=[2, 2])
        