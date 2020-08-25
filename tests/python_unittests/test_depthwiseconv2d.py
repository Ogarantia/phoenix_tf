import unittest
import tensorflow as tf
from upstride.type_generic.custom_op import upstride_ops

def get_inputs_and_filters(in_channels, img_size, filter_size):
  input_upstride = tf.random.normal((1, # N
                                   in_channels, # C
                                   img_size, # H
                                   img_size), # W
                                   dtype=tf.float32)
  filter_upstride = tf.random.normal((1, # channel multiplier or channels per groups. for depthwise conv, 1 channel per groups
                                      in_channels, # C number of channel or number of groups
                                      filter_size, # H
                                      filter_size), # W
                                      stddev = 1/(filter_size ** 2 * in_channels),
                                      dtype=tf.float32)
  input_tf = tf.transpose(input_upstride, [0, 2, 3, 1]) # input is now [N  H  W  C]
  filter_tf = tf.transpose(filter_upstride, [2, 3, 1, 0])
  return input_upstride, filter_upstride, input_tf, filter_tf

class TestDepthwiseConv2D(unittest.TestCase):
    def run_conv2d_test(self, img_size=9, filter_size=3, in_channels=64, padding='VALID', strides=[1,1, 1, 1], dilations=[1, 1]):
        """ Runs a single convolution and compares the result with TensorFlow output """
        input_upstride, filter_upstride, input_tf, filter_tf = get_inputs_and_filters(in_channels, img_size, filter_size)

        # run TF convolution on a properly transposed input
        output_tf = tf.nn.depthwise_conv2d(
          input_tf, filter_tf,
          strides=strides,
          padding=padding,
          dilations=dilations
        )

        # DepthwiseConv2D using conv2D with groups == input channels
        output_upstride = upstride_ops.upstride_conv2d(
          input_upstride, filter_upstride,
          strides=strides,
          padding=padding,
          dilations=dilations,
          data_format='NCHW',
          groups=in_channels
        )
        output_tf = tf.transpose(output_tf, [0,3,1,2])
        
        ## COMPARISONS
        err = tf.math.reduce_max(tf.math.abs(output_upstride - output_tf))
        self.assertLess(err, 1e-4, f"Absolute difference with the reference is too big: {err}")
        print('[DepthwiseConv2D] Absolute difference:', err.numpy())

    def test_conv2d(self):
        self.run_conv2d_test(img_size=224, filter_size=3, in_channels=64, padding='VALID')
        self.run_conv2d_test(img_size=224, filter_size=4, in_channels=64, padding='SAME')
        self.run_conv2d_test(img_size=224, filter_size=3, in_channels=64, strides=[1, 2, 2, 1])
        self.run_conv2d_test(img_size=224, filter_size=3, in_channels=32, padding='VALID')
        self.run_conv2d_test(img_size=224, filter_size=4, in_channels=32, padding='SAME')
        self.run_conv2d_test(img_size=112, filter_size=6, in_channels=32, dilations=[2, 2])

    

    # def test_conv2d_depthwise(self, img_size=5, filter_size=3, in_channels=4, out_channels=6, padding='VALID', strides=[1, 1], dilations=[1, 1], groups=2):
    #     """ Runs a single convolution and compares the result with its expected output """
    #     # If GPU is available, then it computes the output_ref. Otherwise, use the hard-coded version previously computed from a seed
    #     device_name = tf.test.gpu_device_name()
    #     # if device_name != '/device:GPU:0':
    #     if False:
    #       tf.random.set_seed(42)
    #       inputs_channels_first = tf.cast(tf.random.uniform((1, in_channels, img_size, img_size), dtype=tf.int32, minval=-5, maxval=5), dtype=tf.float32)
    #       filters_upstride = tf.cast(tf.random.uniform((groups, in_channels, filter_size, filter_size), dtype=tf.int32, minval=-5, maxval=5), dtype=tf.float32)

    #       output_ref = tf.constant(
    #               [[[[  7.,  53.,   7.],
    #                 [  1.,  16.,   7.],
    #                 [ -9., -30., -36.]],

    #                 [[ 27., -16.,  23.],
    #                 [-11.,  22.,  34.],
    #                 [ 26.,  48.,  11.]],

    #                 [[ 13.,  -6.,   4.],
    #                 [ 36.,   7.,  32.],
    #                 [-20.,  13.,  -5.]],

    #                 [[ 44.,  36.,  -8.],
    #                 [  4.,  54.,  -1.],
    #                 [-21.,  17., -37.]],

    #                 [[  3.,   2.,  31.],
    #                 [-19.,  -6.,  40.],
    #                 [-28.,  20., -19.]],

    #                 [[ -1.,  20.,  57.],
    #                 [-21.,  46., -15.],
    #                 [ 53.,  24., -53.]]]])
    #     else:
    #       inputs_channels_first = tf.cast(tf.random.uniform((1, in_channels, img_size, img_size), dtype=tf.int32, minval=-5, maxval=5), dtype=tf.float32)
    #       filters_upstride = tf.cast(tf.random.uniform((out_channels, in_channels // groups, filter_size, filter_size), dtype=tf.int32, minval=-5, maxval=5), dtype=tf.float32)
          
    #       inputs_channels_last = tf.transpose(inputs_channels_first, [0, 2, 3, 1])
    #       filters_keras = tf.transpose(filters_upstride, [2, 3, 1, 0])

    #       input_shape = inputs_channels_last.shape
    #       # model = tf.keras.layers.Conv2D(out_channels, filter_size, input_shape=input_shape[1:], groups=groups, use_bias=False)
    #       model = tf.keras.layers.DepthwiseConv2D(filter_size, use_bias=False, depth_multiplier=groups)
    #       _ = model(inputs_channels_last)
    #       # self.assertTrue(model.get_weights()[0].shape == filters_keras.shape, f"ref-model-weights' shape and upstride-model-weights' shape mismatch")
    #       model.set_weights([filters_keras])
    #       output_ref = tf.transpose(model(inputs_channels_last), [0, 3, 1, 2])

    #     # run upstride convolution
    #     output_test = upstride_ops.upstride_conv2d(
    #       inputs_channels_first, filters_upstride,
    #       strides=strides,
    #       padding=padding,
    #       dilations=dilations,
    #       data_format='NCHW',
    #       groups=groups
    #     )

    #     err = tf.math.reduce_max(tf.math.abs(output_test - output_ref))
    #     self.assertLess(err, 1e-4, f"Absolute difference with the reference is too big: {err}")
    #     print('[Conv2DFwd] Absolute difference:', err.numpy())
        
class TestDepthwiseConv2DGrad(unittest.TestCase):
    def run_conv2dgrad_test(self, img_size=128, filter_size=3, in_channels=2, padding='SAME', strides=[1,1,1,1], dilations=[1,1]):
      """ Runs a single convolution forward and backward and compares the result with TensorFlow output
      """
      input_upstride, filter_upstride, input_tf, filter_tf = get_inputs_and_filters(in_channels, img_size, filter_size)

      with tf.GradientTape() as gt:
        gt.watch(filter_tf)
        output_tf = tf.nn.conv2d(
                 input_tf, filter_tf,
                 strides=strides,
                 padding=padding,
                 dilations=dilations)
      grad_reference_TF = gt.gradient(output_tf, filter_tf)
      #                                                    O  I  H  W
      grad_reference_TF = tf.transpose(grad_reference_TF, [3, 2, 0, 1])

      with tf.GradientTape() as gt:
        gt.watch(filter_upstride)
        output_upstride = upstride_ops.upstride_conv2d(
                 input_upstride, filter_upstride,
                 strides=strides,
                 padding=padding,
                 dilations=dilations,
                 data_format='NCHW',
                 groups=in_channels)


      grad_test = gt.gradient(output_upstride, filter_upstride)

      ## COMPARISONS
      err = tf.math.reduce_max(tf.math.abs(grad_test - grad_reference_TF))
      self.assertLess(err, 1e-4, f"Absolute difference with the reference is too big: {err}")
      print('[Conv2DBwd] Absolute difference:', err.numpy())

    def test_conv2dgrad(self):
      self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, padding='VALID')
      self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, padding='SAME')
      self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, strides=[1, 2, 2, 1])
      self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, padding='VALID')
      self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, padding='SAME')
      self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, strides=[1, 2, 2, 1])
