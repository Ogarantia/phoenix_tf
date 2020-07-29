import unittest
import tensorflow as tf
from src.python.upstride.type2.tf.keras.layers import *

class TestConv2D(unittest.TestCase):
    def runConv2DTest(self, imgSize=224, filterSize=3, inChannels=3, outChannels=64, padding='VALID', strides=[1, 1], dilations=[1, 1]):
        """ Runs a single convolution and compares the result with TensorFlow output """
        filter = tf.random.uniform((outChannels, inChannels, filterSize, filterSize), dtype=tf.float32)
        input = tf.random.uniform((1, inChannels, imgSize, imgSize), dtype=tf.float32)

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
        print('Absolute difference:', err.numpy())


    def testConv2D(self):
        self.runConv2DTest(imgSize=224, filterSize=3, inChannels=3, outChannels=64, padding='VALID')
        self.runConv2DTest(imgSize=224, filterSize=4, inChannels=3, outChannels=64, padding='SAME')
        self.runConv2DTest(imgSize=224, filterSize=5, inChannels=3, outChannels=16, strides=[2, 2])
        self.runConv2DTest(imgSize=112, filterSize=6, inChannels=16, outChannels=32, dilations=[2, 2])
        self.runConv2DTest(imgSize=112, filterSize=3, inChannels=32, outChannels=48, padding='SAME', strides=[1, 2], dilations=[3, 4])
