import unittest
import tensorflow as tf
from upstride.type_generic.custom_op import upstride_ops
from upstride.type_generic.test import TestCase
from upstride import tf_version
from .layers import DepthwiseConv2D

def setUpModule():
  TestCase.setup()


def get_inputs_and_filters(in_channels, img_size, filter_size, out_channels, use_bias, dtype=tf.float32):
  input_upstride = tf.random.uniform((2,  # N (batch size)
                                      in_channels,  # C
                                      img_size,  # H
                                      img_size),  # W
                                     dtype=dtype, minval=-0.5, maxval=0.5)
  filter_upstride = tf.random.uniform((out_channels,  # channel multiplier or channels per groups. for depthwise conv, 1 channel per groups
                                       in_channels,  # C number of channel or number of groups
                                       filter_size,  # H
                                       filter_size),  # W
                                      dtype=dtype, minval=-0.5, maxval=0.5)
  input_tf = tf.transpose(input_upstride, [0, 2, 3, 1])  # input is now [N  H  W  C]
  filter_tf = tf.transpose(filter_upstride, [2, 3, 1, 0])

  if (use_bias):
    bias = tf.random.uniform((out_channels,), dtype=dtype, minval=-0.5, maxval=0.5)
  else:
    bias = []
  return input_upstride, filter_upstride, input_tf, filter_tf, bias


def get_inputs_and_filters_depthwise(in_channels, img_size, filter_size, use_bias):
  input_upstride = tf.random.normal((2,  # N (batch size)
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
  input_tf = tf.transpose(input_upstride, [0, 2, 3, 1])  # input is now [N  H  W  C]
  filter_tf = tf.transpose(filter_upstride, [2, 3, 0, 1])  # now filter is [H  W  I  O]

  if (use_bias):
    bias = tf.random.uniform((in_channels,), dtype=tf.float32, minval=-0.5, maxval=0.5)
  else:
    bias = []
  return input_upstride, filter_upstride, input_tf, filter_tf, bias


class TestConv2D(TestCase):
  def run_conv2d_test(self, img_size=224, filter_size=3, in_channels=3, out_channels=64, padding='VALID', strides=[1, 1], dilations=[1, 1], use_bias=False, dtype=tf.float32):
    """ Runs a single convolution and compares the result with TensorFlow output """
    input_upstride, filter_upstride, input_tf, filter_tf, bias = get_inputs_and_filters(in_channels, img_size, filter_size, out_channels, use_bias, dtype)

    # run upstride convolution
    output_upstride = upstride_ops.upstride_conv2d(
        input_upstride, filter_upstride, bias,
        use_bias=use_bias,
        strides=strides,
        padding=padding,
        dilations=dilations,
        data_format='NCHW')

    # run TF convolution on a properly transposed input
    output_tf = tf.nn.conv2d(
        input_tf, filter_tf,
        strides=strides,
        padding=padding,
        dilations=dilations)
    if (use_bias):
      output_tf = tf.nn.bias_add(output_tf, bias)

    # compare the outputs
    output_tf = tf.transpose(output_tf, [0, 3, 1, 2])
    err = tf.math.reduce_max(tf.math.abs(output_upstride - output_tf))
    self.assertLess(err, 1e-4, f"Absolute difference with the reference is too big: {err}")
    print('[Conv2DFwd] Absolute difference:', err.numpy())


  def test_conv2d(self):
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=3, out_channels=64, padding='VALID')
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=3, out_channels=64, padding='VALID', use_bias=True)
    self.run_conv2d_test(img_size=224, filter_size=4, in_channels=3, out_channels=64, padding='SAME')
    self.run_conv2d_test(img_size=224, filter_size=4, in_channels=3, out_channels=64, padding='SAME', use_bias=True)
    self.run_conv2d_test(img_size=224, filter_size=5, in_channels=3, out_channels=16, strides=[2, 2])
    self.run_conv2d_test(img_size=224, filter_size=5, in_channels=3, out_channels=16, strides=[2, 2], use_bias=True)
    self.run_conv2d_test(img_size=112, filter_size=6, in_channels=16, out_channels=32, dilations=[2, 2])
    self.run_conv2d_test(img_size=112, filter_size=6, in_channels=16, out_channels=32, dilations=[2, 2], use_bias=True)
    self.run_conv2d_test(img_size=112, filter_size=3, in_channels=32, out_channels=48, padding='SAME', strides=[1, 2], dilations=[3, 4])
    self.run_conv2d_test(img_size=112, filter_size=3, in_channels=32, out_channels=48, padding='SAME', strides=[1, 2], dilations=[3, 4], use_bias=True)
    # few float16 tests (only on GPU)
    if tf.test.gpu_device_name():
      # FIXME: these tests fail on a GPU with CUDA CC < 5.3. Need a reliable way to detect if we run such a GPU
      self.run_conv2d_test(img_size=224, filter_size=3, in_channels=5, out_channels=32, padding='VALID', use_bias=True, dtype=tf.float16)
      self.run_conv2d_test(img_size=56, filter_size=3, in_channels=16, out_channels=16, padding='SAME', use_bias=False, dtype=tf.float16)

  def test_conv2d_grouped(self, img_size=5, filter_size=3, in_channels=4, out_channels=6, padding='VALID', strides=[1, 1], dilations=[1, 1], groups=2):
    """ Runs a single grouped convolution and compares the result with its expected output """
    # If GPU is available and tf_version is at least 2.3, then it computes the output_ref. Otherwise, use the hard-coded version previously computed from a seed
    if not tf.test.gpu_device_name() or tf_version < 2.3:
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

      inputs_channels_last = tf.transpose(inputs_channels_first, [0, 2, 3, 1])  # (N, C, H, W) -> (N, H, W, C)
      filters_keras = tf.transpose(filters_upstride, [2, 3, 1, 0])  # (O, I, H, W) -> (H, W, I, O)

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
          inputs_channels_first, filters_upstride, [],
          strides=strides,
          padding=padding,
          dilations=dilations,
          data_format='NCHW',
          groups=groups,
          use_bias=False
      )
    grad_test_filters_upstride = gt.gradient(output_test, filters_upstride)
    grad_test_inputs_channels_first = gt.gradient(output_test, inputs_channels_first)

    self.assert_and_print(output_test, output_ref, "Grouped Conv2DFwd", "output")
    self.assert_and_print(grad_test_inputs_channels_first, grad_ref_inputs_channels_first, "Grouped Conv2DFwd", "dinputs")


class TestConv2DGrad(TestCase):
  def run_conv2dgrad_test(self, img_size=128, filter_size=3, in_channels=2, out_channels=1, padding='SAME', strides=[1, 1], dilations=[1, 1], use_bias=False, dtype=tf.float32):
    """ Runs a single convolution forward and backward and compares the result with TensorFlow output """
    input_upstride, filter_upstride, input_tf, filter_tf, bias = get_inputs_and_filters(in_channels, img_size, filter_size, out_channels, use_bias, dtype)

    # UPSTRIDE
    with tf.GradientTape(persistent=True) as gt:
      gt.watch([filter_upstride, input_upstride])
      if use_bias:
        gt.watch(bias)
      output_upstride = upstride_ops.upstride_conv2d(input_upstride, filter_upstride, bias,
                                                     strides=strides,
                                                     padding=padding,
                                                     dilations=dilations,
                                                     data_format='NCHW',
                                                     use_bias=use_bias)
    grad_test_filter = gt.gradient(output_upstride, filter_upstride)
    grad_test_input = gt.gradient(output_upstride, input_upstride)
    if use_bias:
      grad_test_bias = gt.gradient(output_upstride, bias)

    # TENSORFLOW
    input_tf = tf.identity(input_tf)
    filter_tf = tf.identity(filter_tf)
    with tf.GradientTape(persistent=True) as gt:
      gt.watch([filter_tf, input_tf])
      if use_bias:
        gt.watch(bias)
      output_tf = tf.nn.conv2d(input_tf, filter_tf,
                               strides=strides,
                               padding=padding,
                               dilations=dilations)
      if use_bias:
        output_tf = tf.nn.bias_add(output_tf, bias)

    grad_reference_filter_tf = gt.gradient(output_tf, filter_tf)
    grad_reference_input_tf = gt.gradient(output_tf, input_tf)
    if use_bias:
      grad_reference_bias_tf = gt.gradient(output_tf, bias)
    #                                                                  O  I  H  W
    grad_reference_filter_tf = tf.transpose(grad_reference_filter_tf, [3, 2, 0, 1])
    grad_reference_input_tf = tf.transpose(grad_reference_input_tf, [0, 3, 1, 2])

    # COMPARISONS
    self.assert_and_print(output_upstride, tf.transpose(output_tf, [0, 3, 1, 2]), "TestConv2DGrad", "output")
    self.assert_and_print(grad_test_input, grad_reference_input_tf, "TestConv2DGrad", "dinput")
    self.assert_and_print(grad_test_filter, grad_reference_filter_tf, "TestConv2DGrad", "dweights")
    if use_bias:
      self.assert_and_print(grad_test_bias, grad_reference_bias_tf, "TestConv2DGrad", "dbias")

  def test_conv2dgrad(self):
    self.run_conv2dgrad_test(img_size=8, filter_size=3, in_channels=2, out_channels=2, padding='VALID')
    self.run_conv2dgrad_test(img_size=8, filter_size=3, in_channels=2, out_channels=2, padding='VALID', use_bias=True)
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='VALID')
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='VALID', use_bias=True)
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='SAME')
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='SAME', use_bias=True)
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, strides=[2, 2])
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, strides=[2, 2], use_bias=True)
    self.run_conv2dgrad_test(img_size=32, filter_size=3, in_channels=3, out_channels=8, padding='VALID')
    self.run_conv2dgrad_test(img_size=32, filter_size=3, in_channels=3, out_channels=8, padding='VALID', use_bias=True)
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, out_channels=8, padding='SAME')
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, out_channels=8, padding='SAME', use_bias=True)
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, out_channels=8, strides=[2, 2])
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, out_channels=8, strides=[2, 2], use_bias=True)
    self.run_conv2dgrad_test(img_size=224, filter_size=3, in_channels=3, out_channels=48, strides=[2, 2], padding='VALID')
    self.run_conv2dgrad_test(img_size=224, filter_size=3, in_channels=3, out_channels=48, strides=[2, 2], padding='VALID', use_bias=True)
    # few float16 tests (only on GPU)
    if tf.test.gpu_device_name():
      # FIXME: these tests fail on a GPU with CUDA CC < 5.3. Need a reliable way to detect if we run such a GPU
      self.run_conv2dgrad_test(img_size=5, filter_size=2, in_channels=3, out_channels=4, padding='VALID', use_bias=False, dtype=tf.float16)
      self.run_conv2dgrad_test(img_size=5, filter_size=1, in_channels=8, out_channels=16, padding='VALID', use_bias=True, dtype=tf.float16)
      self.run_conv2dgrad_test(img_size=7, filter_size=1, in_channels=8, out_channels=16, padding='VALID', use_bias=True, dtype=tf.float16)


class TestDepthwiseConv2D(TestCase):
  def run_conv2d_test(self, img_size=9, filter_size=3, in_channels=64, use_bias=False, padding='VALID', strides=[1, 1, 1, 1], dilations=[1, 1]):
    """ Runs a single convolution and compares the result with TensorFlow output """
    input_upstride, filter_upstride, input_tf, filter_tf, bias = get_inputs_and_filters_depthwise(in_channels, img_size, filter_size, use_bias)

    # run TF convolution on a properly transposed input
    output_tf = tf.nn.depthwise_conv2d(
        input_tf, filter_tf,
        strides=strides,
        padding=padding,
        dilations=dilations)
    if (use_bias):
      output_tf = tf.nn.bias_add(output_tf, bias)

    # DepthwiseConv2D using conv2D with groups == input channels
    output_upstride = upstride_ops.upstride_conv2d(
        input_upstride, filter_upstride, bias,
        strides=strides,
        padding=padding,
        dilations=dilations,
        data_format='NCHW',
        groups=in_channels,
        use_bias=use_bias)
    output_tf = tf.transpose(output_tf, [0, 3, 1, 2])

    # COMPARISONS
    self.assert_and_print(output_upstride, output_tf, "DepthwiseConv2DFwd", "output")

  def test_conv2d(self):
    self.run_conv2d_test(img_size=5, filter_size=3, in_channels=4, padding='VALID')
    self.run_conv2d_test(img_size=5, filter_size=3, in_channels=4, padding='VALID', use_bias=True)
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=64, padding='VALID')
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=64, padding='VALID', use_bias=True)
    self.run_conv2d_test(img_size=224, filter_size=4, in_channels=64, padding='SAME')
    self.run_conv2d_test(img_size=224, filter_size=4, in_channels=64, padding='SAME', use_bias=True)
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=64, strides=[1, 2, 2, 1])
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=64, strides=[1, 2, 2, 1], use_bias=True)
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=32, padding='VALID')
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=32, padding='VALID', use_bias=True)
    self.run_conv2d_test(img_size=224, filter_size=4, in_channels=32, padding='SAME')
    self.run_conv2d_test(img_size=224, filter_size=4, in_channels=32, padding='SAME', use_bias=True)
    self.run_conv2d_test(img_size=112, filter_size=6, in_channels=32, dilations=[2, 2])
    self.run_conv2d_test(img_size=112, filter_size=6, in_channels=32, dilations=[2, 2], use_bias=True)

  def run_depthwise_conv2d_python_constructor(self, img_size=9, filter_size=3, in_channels=64, use_bias=False, padding='VALID', strides=(1, 1), dilations=[1, 1]):
    """ Test to verify that we are able to call depthwise convolution through the python interface.
    """
    cpp_inputs = tf.random.normal((2,           # N (batch size)
                                   in_channels, # C
                                   img_size,    # H
                                   img_size),   # W
                                   dtype=tf.float32)
    upstride_depthwise_conv = DepthwiseConv2D(kernel_size=filter_size, strides=strides, padding=padding, dilation_rate=dilations, use_bias=use_bias)
    upstride_depthwise_conv(cpp_inputs) # runs a first time to initialize the kernel

  def test_depthwise_conv2d_python_constructor(self, img_size=224, filter_size=3, in_channels=64, padding='VALID', use_bias=True):
    try:
      tf.keras.backend.set_image_data_format('channels_first')  # FIXME We should find a proper way to pass 'channels_first'
      self.run_depthwise_conv2d_python_constructor(img_size=224, filter_size=3, in_channels=64, padding='VALID', use_bias=True)
      self.run_depthwise_conv2d_python_constructor(img_size=224, filter_size=3, in_channels=64, padding='VALID', use_bias=False)
    finally:
      tf.keras.backend.set_image_data_format('channels_last')  # FIXME We should find a proper way to pass 'channels_last'


class TestDepthwiseConv2DGrad(TestCase):
  def run_conv2dgrad_test(self, img_size=128, filter_size=3, in_channels=2, use_bias=False, padding='SAME', strides=[1, 1, 1, 1], dilations=[1, 1]):
    """ Runs a single convolution forward and backward and compares the result with TensorFlow output
    """
    input_upstride, filter_upstride, input_tf, filter_tf, bias = get_inputs_and_filters_depthwise(in_channels, img_size, filter_size, use_bias)

    with tf.GradientTape(persistent=True) as gt:
      gt.watch([filter_tf, input_tf])
      output_tf = tf.nn.conv2d(
          input_tf, filter_tf,
          strides=strides,
          padding=padding,
          dilations=dilations)
    grad_reference_filter_tf = gt.gradient(output_tf, filter_tf)
    grad_reference_input_tf = gt.gradient(output_tf, input_tf)
    # transpose to match UpStride layout                               O  I  H  W
    grad_reference_filter_tf = tf.transpose(grad_reference_filter_tf, [2, 3, 0, 1])
    grad_reference_input_tf = tf.transpose(grad_reference_input_tf, [0, 3, 1, 2])

    with tf.GradientTape(persistent=True) as gt:
      gt.watch([filter_upstride, input_upstride])
      output_upstride = upstride_ops.upstride_conv2d(
          input_upstride, filter_upstride, bias,
          strides=strides,
          padding=padding,
          dilations=dilations,
          data_format='NCHW',
          groups=in_channels,
          use_bias=use_bias)
    grad_test_filter = gt.gradient(output_upstride, filter_upstride)
    grad_test_input = gt.gradient(output_upstride, input_upstride)

    # COMPARISONS
    self.assert_and_print(grad_test_filter, grad_reference_filter_tf, "Conv2DBwd", "dfilter")
    self.assert_and_print(grad_test_input, grad_reference_input_tf, "Conv2DBwd", "dinput")

  def test_conv2dgrad(self):
    self.run_conv2dgrad_test(img_size=5, filter_size=3, in_channels=4, padding='VALID')
    self.run_conv2dgrad_test(img_size=5, filter_size=3, in_channels=4, padding='VALID', use_bias=True)
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, padding='VALID')
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, padding='VALID', use_bias=True)
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, padding='SAME')
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, padding='SAME', use_bias=True)
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, strides=[1, 2, 2, 1])
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, strides=[1, 2, 2, 1], use_bias=True)
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, padding='VALID')
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, padding='VALID', use_bias=True)
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, padding='SAME')
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, padding='SAME', use_bias=True)
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, strides=[1, 2, 2, 1])
    self.run_conv2dgrad_test(img_size=32, filter_size=4, in_channels=3, strides=[1, 2, 2, 1], use_bias=True)

class TestDense(TestCase):
  def get_inputs_and_filters_dense(self, batch_size, in_features, out_features, dtype=tf.float32):
    inputs = tf.random.uniform([batch_size, in_features], minval=-1, maxval=1, dtype=dtype)
    weights = tf.random.uniform([in_features, out_features], minval=-1, maxval=1, dtype=dtype)
    bias = tf.random.uniform([out_features,], minval=-1, maxval=1, dtype=dtype)
    return inputs, weights, bias

  def run_test(self, batch_size, in_features, out_features, use_bias=False, dtype=tf.float32):
    """ Runs a single dense and compares the result with TensorFlow output """
    from . import layers
    inputs, weights, bias = self.get_inputs_and_filters_dense(batch_size, in_features, out_features, dtype)
    # run upstride dense
    model_upstride = layers.Dense(out_features, use_bias=use_bias)
    model_upstride(inputs)
    if use_bias:
      model_upstride.set_weights([weights, tf.expand_dims(bias, 0)])
    else:
      model_upstride.set_weights([weights])

    with tf.GradientTape(persistent=True) as gt:
      gt.watch([model_upstride.kernel, inputs])
      if use_bias:
        gt.watch(model_upstride.bias)
      output_upstride = model_upstride(inputs)
      if use_bias:
        dinputs_upstride, dweights_upstride, dbias_upstride = gt.gradient(output_upstride, [inputs, model_upstride.kernel, model_upstride.bias])
      else:
        dinputs_upstride, dweights_upstride = gt.gradient(output_upstride, [inputs, model_upstride.kernel])

    # run TF dense
    model_tf = tf.keras.layers.Dense(out_features, use_bias=False)
    output_tf = model_tf(inputs)
    model_tf.set_weights([weights])
    with tf.GradientTape(persistent=True) as gt:
      gt.watch([model_tf.kernel, inputs])
      output_tf = model_tf(inputs)
      if use_bias:
        gt.watch(bias)
        output_tf = tf.nn.bias_add(output_tf, bias)
        dbias_tf = gt.gradient(output_tf, bias)
      dinputs_tf, dweights_tf = gt.gradient(output_tf, [inputs, model_tf.kernel])

    # compare the outputs
    self.assert_and_print(output_upstride, output_tf, "DenseFwd")
    self.assert_and_print(dinputs_upstride, dinputs_tf, "DenseFwd", "dinput")
    self.assert_and_print(dweights_upstride, dweights_tf, "DenseFwd", "dweights")
    if use_bias:
      self.assert_and_print(dbias_upstride, dbias_tf, "DenseFwd", "dbias")

  def test_dense(self):
    self.run_test(batch_size=1, in_features=1, out_features=1)
    self.run_test(batch_size=2, in_features=3, out_features=4)
    self.run_test(batch_size=64, in_features=64, out_features=10)
    self.run_test(batch_size=64, in_features=64, out_features=10, use_bias=True)
    self.run_test(batch_size=5, in_features=4, out_features=3, use_bias=True)
    self.run_test(batch_size=128, in_features=100, out_features=10)
    self.run_test(batch_size=128, in_features=100, out_features=10, use_bias=True)
    # few float16 tests (only on GPU)
    if tf.test.gpu_device_name():
      # FIXME: these tests fail on a GPU with CUDA CC < 5.3. Need a reliable way to detect if we run such a GPU
      try:
        tf.keras.backend.set_floatx('float16')
        self.run_test(batch_size=8, in_features=4, out_features=4, use_bias=True, dtype=tf.float16)
        self.run_test(batch_size=1, in_features=2, out_features=16, use_bias=False, dtype=tf.float16)
      finally:
        tf.keras.backend.set_floatx('float32')