import unittest
import tensorflow as tf
from upstride.type_generic.custom_op import upstride_ops


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

  def test_conv2d_grouped(self, img_size=5, filter_size=3, in_channels=4, out_channels=6, padding='VALID', strides=[1, 1], dilations=[1, 1], groups=2):
    """ Runs a single grouped convolution and compares the result with its expected output """
    # If GPU is available, then it computes the output_ref. Otherwise, use the hard-coded version previously computed from a seed
    if not tf.test.gpu_device_name():
      tf.random.set_seed(42)
      inputs_channels_first = tf.cast(tf.random.uniform((1, in_channels, img_size, img_size), dtype=tf.int32, minval=-5, maxval=5), dtype=tf.float32)
      filters_upstride = tf.cast(tf.random.uniform((out_channels, in_channels // groups, filter_size, filter_size), dtype=tf.int32, minval=-5, maxval=5), dtype=tf.float32)

      output_ref = tf.constant(
          [[[[51.,  89.,  -1.],
             [5.,  70.,   6.],
             [-30., -13., -73.]],

            [[30., -14.,  54.],
             [-30.,  16.,  74.],
             [-2.,  68.,  -8.]],

            [[12.,  14.,  61.],
             [15.,  53.,  17.],
             [33.,  37., -58.]],

            [[24.,  85.,  16.],
             [10., -43.,  36.],
             [-44.,  10.,  15.]],

            [[25., -58., -20.],
             [-53., -15., -61.],
             [19.,   2.,  40.]],

            [[-29.,   2.,  -9.],
             [34.,  -5.,  15.],
             [3.,  33.,   3.]]]])

      grad_ref_filters_keras_transposed = None  # Tensorflow does not compute the gradient w.r.t the filter, even if kindly asked to do so with GradientTape
      grad_ref_inputs_channels_last = tf.constant(
          [[[[4.,   6.,  -3., -13.],
             [5.,  -1.,  -6., -10.],
             [0.,  -4., -15., -12.],
             [-4., -10., -12.,   1.],
             [-5.,  -3.,  -9.,  -2.]],

            [[5.,   1.,   2., -13.],
             [-3., -17.,  -2., -10.],
             [-9., -12., -16., -18.],
             [-14., -13., -18.,  -5.],
             [-6.,   5., -14.,  -8.]],

            [[1.,   0.,   3., -16.],
             [-7., -26., -10., -16.],
             [-9., -21., -19., -28.],
             [-10., -21., -22., -12.],
             [-2.,   5.,  -9., -12.]],

            [[-3.,  -6.,   6.,  -3.],
             [-12., -25.,  -4.,  -6.],
             [-9., -17.,  -4., -16.],
             [-6., -11., -10., -13.],
             [3.,   8.,   0., -10.]],

            [[-4.,  -1.,   1.,  -3.],
             [-4.,  -9.,  -8.,  -6.],
             [0.,  -9.,  -3., -10.],
             [4.,  -8.,  -4.,  -7.],
             [4.,   0.,   5.,  -4.]]]])
    else:
      inputs_channels_first = tf.cast(tf.random.uniform((1, in_channels, img_size, img_size), dtype=tf.int32, minval=-5, maxval=5), dtype=tf.float32)
      filters_upstride = tf.cast(tf.random.uniform((out_channels, in_channels // groups, filter_size, filter_size), dtype=tf.int32, minval=-5, maxval=5), dtype=tf.float32)

      inputs_channels_last = tf.transpose(inputs_channels_first, [0, 2, 3, 1])  # (N, H, H, W) -> (N, H, W, C)
      filters_keras = tf.transpose(filters_upstride, [2, 3, 1, 0])  # (H, W, I, O) -> (O, I, H, W)

      input_shape = inputs_channels_last.shape
      model = tf.keras.layers.Conv2D(out_channels, filter_size, input_shape=input_shape[1:], groups=groups, use_bias=False)
      model(inputs_channels_last)
      self.assertTrue(model.get_weights()[0].shape == filters_keras.shape, f"ref-model-weights' shape and upstride-model-weights' shape mismatch")
      model.set_weights([filters_keras])

      with tf.GradientTape(persistent=True) as gt:
        gt.watch(filters_keras)
        gt.watch(inputs_channels_last)
        output_ref = tf.transpose(model(inputs_channels_last), [0, 3, 1, 2])

      grad_ref_filters_keras_transposed = gt.gradient(output_ref, filters_keras)
      grad_ref_inputs_channels_last = gt.gradient(output_ref, inputs_channels_last)

    grad_ref_inputs_channels_first = tf.transpose(grad_ref_inputs_channels_last, [0, 3, 1, 2])
    # run upstride convolution
    with tf.GradientTape(persistent=True) as gt:
      gt.watch(filters_upstride)
      gt.watch(inputs_channels_first)
      output_test = upstride_ops.upstride_conv2d(
          inputs_channels_first, filters_upstride,
          strides=strides,
          padding=padding,
          dilations=dilations,
          data_format='NCHW',
          groups=groups
      )
    grad_test_filters_upstride = gt.gradient(output_test, filters_upstride)
    grad_test_inputs_channels_first = gt.gradient(output_test, inputs_channels_first)

    err = tf.math.reduce_max(tf.math.abs(output_test - output_ref))
    self.assertLess(err, 1e-4, f"Absolute difference with the reference is too big: {err}")
    print('[Grouped Conv2DFwd] Absolute difference:', err.numpy())

    err = tf.math.reduce_max(tf.math.abs(grad_test_inputs_channels_first - grad_ref_inputs_channels_first))
    self.assertLess(err, 1e-4, f"Absolute difference with the reference is too big: {err}")
    print('[Grouped Conv2DBwd] Absolute difference:', err.numpy())


class TestConv2DGrad(unittest.TestCase):
  def run_conv2dgrad_test(self, img_size=128, filter_size=3, in_channels=2, out_channels=1, padding='SAME', strides=[1, 1], dilations=[1, 1]):
    """ Runs a single convolution forward and backward and compares the result with TensorFlow output """
    input = tf.random.uniform((1, in_channels, img_size, img_size), dtype=tf.float32, minval=-0.5, maxval=0.5)
    filter = tf.random.uniform((out_channels, in_channels, filter_size, filter_size), dtype=tf.float32, minval=-0.5, maxval=0.5)

    # UPSTRIDE
    with tf.GradientTape(persistent=True) as gt:
      gt.watch(filter)
      gt.watch(input)
      output_test = upstride_ops.upstride_conv2d(
          input, filter,
          strides=strides,
          padding=padding,
          dilations=dilations,
          data_format='NCHW')
    grad_test_filter = gt.gradient(output_test, filter)
    grad_test_input = gt.gradient(output_test, input)

    # TENSORFLOW
    # #                            N  H  W  C
    input_t = tf.transpose(input, [0, 2, 3, 1])
    # #                              H  W  I  O
    filter_t = tf.transpose(filter, [2, 3, 1, 0])
    input_t = tf.identity(input_t)
    filter_t = tf.identity(filter_t)
    with tf.GradientTape(persistent=True) as gt:
      gt.watch(filter_t)
      gt.watch(input_t)
      output_ref_TF = tf.nn.conv2d(
          input_t, filter_t,
          strides=strides,
          padding=padding,
          dilations=dilations)

    grad_reference_filter_TF = gt.gradient(output_ref_TF, filter_t)
    grad_reference_input_TF = gt.gradient(output_ref_TF, input_t)
    #                                                                  O  I  H  W
    grad_reference_filter_TF = tf.transpose(grad_reference_filter_TF, [3, 2, 0, 1])
    grad_reference_input_TF = tf.transpose(grad_reference_input_TF, [0, 3, 1, 2])

    # COMPARISONS
    err_filter = tf.math.reduce_max(tf.math.abs(grad_test_filter - grad_reference_filter_TF))
    self.assertLess(err_filter, 1e-4, f"Absolute filter difference compared to the reference is too big: {err_filter}")
    print('[Conv2DBwd] Absolute filter difference:', err_filter.numpy())

    err_input = tf.math.reduce_max(tf.math.abs(grad_test_input - grad_reference_input_TF))
    self.assertLess(err_input, 1e-4, f"Absolute input difference compared to the reference is too big: {err_input}")
    print('[Conv2DBwd] Absolute input difference:', err_input.numpy())

  def test_conv2dgrad(self):
    self.run_conv2dgrad_test(img_size=8, filter_size=3, in_channels=2, out_channels=2, padding='VALID')
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='VALID')
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='SAME')
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, strides=[2, 2])
    self.run_conv2dgrad_test(img_size=32, filter_size=3, in_channels=3, out_channels=8, padding='VALID')
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, out_channels=8, padding='SAME')
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, out_channels=8, strides=[2, 2])
    self.run_conv2dgrad_test(img_size=224, filter_size=3, in_channels=3, out_channels=48, strides=[2, 2], padding='VALID')


def get_inputs_and_filters(in_channels, img_size, filter_size):
  input_upstride = tf.random.normal((1,  # N
                                     in_channels,  # C
                                     img_size,  # H
                                     img_size),  # W
                                    dtype=tf.float32)
  filter_upstride = tf.random.normal((in_channels,  # C number of channel or number of groups
                                      1,  # channel multiplier or channels per groups. for depthwise conv, 1 channel per groups
                                      filter_size,  # H
                                      filter_size),  # W
                                     stddev=1/(filter_size ** 2 * in_channels),
                                     dtype=tf.float32)
  input_tf = tf.transpose(input_upstride, [0, 2, 3, 1])  # now input is [N  H  W  C]
  filter_tf = tf.transpose(filter_upstride, [2, 3, 0, 1])  # now filter is [H  W  I  O]
  return input_upstride, filter_upstride, input_tf, filter_tf


class TestDepthwiseConv2D(unittest.TestCase):
  def run_conv2d_test(self, img_size=9, filter_size=3, in_channels=64, padding='VALID', strides=[1, 1, 1, 1], dilations=[1, 1]):
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
    output_tf = tf.transpose(output_tf, [0, 3, 1, 2])

    # COMPARISONS
    err = tf.math.reduce_max(tf.math.abs(output_upstride - output_tf))
    self.assertLess(err, 1e-4, f"Absolute difference with the reference is too big: {err}")
    print('[DepthwiseConv2DFwd] Absolute difference:', err.numpy())

  def test_conv2d(self):
    self.run_conv2d_test(img_size=5, filter_size=3, in_channels=4, padding='VALID')
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=64, padding='VALID')
    self.run_conv2d_test(img_size=224, filter_size=4, in_channels=64, padding='SAME')
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=64, strides=[1, 2, 2, 1])
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=32, padding='VALID')
    self.run_conv2d_test(img_size=224, filter_size=4, in_channels=32, padding='SAME')
    self.run_conv2d_test(img_size=112, filter_size=6, in_channels=32, dilations=[2, 2])


class TestDepthwiseConv2DGrad(unittest.TestCase):
  def run_conv2dgrad_test(self, img_size=128, filter_size=3, in_channels=2, padding='SAME', strides=[1, 1, 1, 1], dilations=[1, 1]):
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
    grad_reference_TF = tf.transpose(grad_reference_TF, [2, 3, 0, 1])

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

    # COMPARISONS
    err = tf.math.reduce_max(tf.math.abs(grad_test - grad_reference_TF))
    self.assertLess(err, 1e-4, f"Absolute difference with the reference is too big: {err}")
    print('[DepthwiseConv2DBwd] Absolute difference:', err.numpy())

  def test_conv2dgrad(self):
    self.run_conv2dgrad_test(img_size=5, filter_size=3, in_channels=4, padding='VALID')
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, padding='VALID')
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, padding='SAME')
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, strides=[1, 2, 2, 1])
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, padding='VALID')
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, padding='SAME')
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, strides=[1, 2, 2, 1])
