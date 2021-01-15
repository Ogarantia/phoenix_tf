import unittest
import tensorflow as tf
from packaging import version
from upstride.internal.custom_ops import upstride_ops
from src.python.upstride.internal.test import setUpModule, apply_some_non_linearity, Conv2DTestSet, PointwiseConv2DTestSet, DepthwiseConv2DTestSet, DenseTestSet, InputGradientAndTypeTest, TestCase
from upstride.internal.clifford_product import CliffordProduct
from upstride.type0.tf.keras.layers import DepthwiseConv2D, Conv2D, Dense

clifford_product = CliffordProduct((0, 0, 0), [""])
setUpModule()


class Type0Conv2DTestSet(Conv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, Conv2D)


class Type0PointwiseConv2DTestSet(PointwiseConv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, Conv2D)


class Type0DepthwiseConv2DTestSet(DepthwiseConv2DTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, DepthwiseConv2D)


class Type0DenseTestSet(DenseTestSet, unittest.TestCase):
  def setUp(self):
    self.setup(clifford_product, Dense)


class Type0InputGradientAndTypeTest(InputGradientAndTypeTest, unittest.TestCase):
  def setUp(self):
    from upstride.type0.tf.keras import layers
    self.setup(layers)


def get_inputs_and_filters(in_channels, img_size, filter_size, out_channels, use_bias, dtype=tf.float32, batch_size=2, val=0.5):
  #                                   N (batch size),   C,        H,        W
  input_upstride = tf.random.uniform((batch_size, in_channels, img_size, img_size), dtype=dtype, minval=-val, maxval=val)
  filter_upstride = tf.random.uniform((out_channels,  # channel multiplier or channels per groups. for depthwise conv, 1 channel per groups
                                       in_channels,  # C number of channel or number of groups
                                       filter_size,  # H
                                       filter_size),  # W
                                      dtype=dtype, minval=-val, maxval=val)
  input_tf = tf.transpose(input_upstride, [0, 2, 3, 1])  # input is now [N  H  W  C]
  filter_tf = tf.transpose(filter_upstride, [2, 3, 1, 0])

  if (use_bias):
    bias = tf.random.uniform((out_channels,), dtype=dtype, minval=-val, maxval=val)
  else:
    bias = []
  return input_upstride, filter_upstride, input_tf, filter_tf, bias

def get_output_and_gradients(model, kernel, inputs):
  with tf.GradientTape(persistent=True) as gt:
    gt.watch([kernel, inputs])
    if model.bias is not None:
      gt.watch(model.bias)
    output = model(inputs)
    output = apply_some_non_linearity(output)
    if model.bias is not None:
      dinputs, dweights, dbias = gt.gradient(output, [inputs, kernel, model.bias])
    else:
      dinputs, dweights = gt.gradient(output, [inputs, kernel])
      dbias = None
  return output, dinputs, dweights, dbias

class TestConv2D(TestCase):
  def run_conv2d_test(self, img_size=224, filter_size=3, in_channels=3, out_channels=64, padding='VALID', strides=[1, 1], dilations=[1, 1], use_bias=False, dtype=tf.float32, batch_size=2):
    """ Runs a single convolution and compares the result with TensorFlow output """
    input_upstride, filter_upstride, input_tf, filter_tf, bias = get_inputs_and_filters(in_channels, img_size, filter_size, out_channels, use_bias, dtype, batch_size)

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
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=3, out_channels=64, padding='VALID', batch_size=3)
    self.run_conv2d_test(img_size=224, filter_size=3, in_channels=3, out_channels=64, padding='VALID', batch_size=4, use_bias=True)
    self.run_conv2d_test(img_size=224, filter_size=4, in_channels=3, out_channels=64, padding='SAME', batch_size=5)
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
    if not tf.test.gpu_device_name() or version.parse(tf.__version__) < version.parse("2.3"):
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
      # TODO call grouped convolution through the python interface
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
  def run_conv2dgrad_test(self, img_size=128, filter_size=3, in_channels=2, out_channels=1, padding='SAME', strides=[1, 1], dilations=[1, 1], use_bias=False, dtype=tf.float32, batch_size=2):
    """ Runs a single convolution forward and backward and compares the result with TensorFlow output """
    input_upstride, filter_upstride, input_tf, filter_tf, bias = get_inputs_and_filters(in_channels, img_size, filter_size, out_channels, use_bias, dtype, batch_size)

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
    self.run_conv2dgrad_test(img_size=8, filter_size=3, in_channels=2, out_channels=2, padding='VALID', batch_size=3)
    self.run_conv2dgrad_test(img_size=8, filter_size=3, in_channels=2, out_channels=2, padding='VALID', batch_size=4, use_bias=True)
    self.run_conv2dgrad_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='VALID', batch_size=5)
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
  def run_dw_conv2d_test(self, img_size=128, filter_size=3, channels=2, use_bias=False, padding='SAME', strides=[1, 1], dilations=[1, 1], batch_size=2):
    """ Runs a single convolution forward and backward and compares the result with TensorFlow output
    """
    #                               N    , Channels,  Height ,   Width
    inputs = tf.random.normal((batch_size, channels, img_size, img_size), dtype=tf.float32)
    # Defines models
    model_up = DepthwiseConv2D(filter_size, strides, padding, data_format='channels_first', bias_initializer='glorot_uniform', dilation_rate=dilations, use_bias=use_bias)
    model_tf = tf.keras.layers.DepthwiseConv2D(filter_size, strides, padding, data_format='channels_first', dilation_rate=dilations, use_bias=use_bias)
    # Call the models for the first time so that the weights get initialized and can be modified later on
    model_up(inputs)
    model_up.require_input_grad = True
    model_tf(inputs)
    # Sets weights from the TF model to be equal to the ones on the Phoenix model, up to a transposition
    ref_params = model_up.get_weights()
    if use_bias:
      model_tf.set_weights([tf.transpose(ref_params[0], [2, 3, 0, 1]), ref_params[1]])
    else:
      model_tf.set_weights([tf.transpose(ref_params[0], [2, 3, 0, 1])])

    output_upstride, dinputs_upstride, dweights_upstride, dbias_upstride = get_output_and_gradients(model_up, model_up.kernel, inputs)
    output_tf, dinputs_tf, dweights_tf, dbias_tf = get_output_and_gradients(model_tf, model_tf.depthwise_kernel, inputs)

    self.assert_and_print(output_upstride, output_tf, "DepthwiseConv2D", "output")
    self.assert_and_print(dweights_upstride, tf.transpose(dweights_tf, [2, 3, 0, 1]), "DepthwiseConv2D", "dfilter")
    self.assert_and_print(dinputs_upstride, dinputs_tf, "DepthwiseConv2D", "dinput")
    if use_bias:
      self.assert_and_print(dbias_upstride, dbias_tf, "DepthwiseConv2D", "dbias")

  def test_dw_conv2d(self):
    self.run_dw_conv2d_test(img_size=5, filter_size=3, channels=4, padding='VALID', batch_size=3)
    self.run_dw_conv2d_test(img_size=5, filter_size=3, channels=4, padding='VALID', batch_size=4, use_bias=True)
    self.run_dw_conv2d_test(img_size=9, filter_size=3, channels=3, padding='VALID', batch_size=5)
    self.run_dw_conv2d_test(img_size=9, filter_size=3, channels=3, padding='VALID', use_bias=True)
    self.run_dw_conv2d_test(img_size=9, filter_size=3, channels=3, padding='SAME')
    self.run_dw_conv2d_test(img_size=9, filter_size=3, channels=3, padding='SAME', use_bias=True)
    self.run_dw_conv2d_test(img_size=9, filter_size=3, channels=3, strides=[2, 2])
    self.run_dw_conv2d_test(img_size=9, filter_size=3, channels=3, strides=[2, 2], use_bias=True)
    self.run_dw_conv2d_test(img_size=32, filter_size=4, channels=3, padding='VALID')
    self.run_dw_conv2d_test(img_size=32, filter_size=4, channels=3, padding='VALID', use_bias=True)
    self.run_dw_conv2d_test(img_size=32, filter_size=4, channels=3, padding='SAME')
    self.run_dw_conv2d_test(img_size=32, filter_size=4, channels=3, padding='SAME', use_bias=True)
    self.run_dw_conv2d_test(img_size=32, filter_size=4, channels=3, strides=[2, 2])
    self.run_dw_conv2d_test(img_size=32, filter_size=4, channels=3, strides=[2, 2], use_bias=True)

class TestDense(TestCase):
  def get_inputs_and_filters_dense(self, batch_size, in_features, out_features, dtype=tf.float32):
    inputs = tf.random.uniform([batch_size, in_features], minval=-1, maxval=1, dtype=dtype)
    weights = tf.random.uniform([in_features, out_features], minval=-1, maxval=1, dtype=dtype)
    bias = tf.random.uniform([out_features,], minval=-1, maxval=1, dtype=dtype)
    return inputs, weights, bias

  def run_test(self, batch_size, in_features, out_features, use_bias=False, dtype=tf.float32):
    """ Runs a single dense and compares the result with TensorFlow output """
    inputs, weights, bias = self.get_inputs_and_filters_dense(batch_size, in_features, out_features, dtype)
    # Declare models
    model_up = Dense(out_features, use_bias=use_bias)
    model_up.require_input_grad = True
    model_tf = tf.keras.layers.Dense(out_features, use_bias=use_bias)

    # Initialize weights so that they exist at the moment that they are going to be modified
    model_up(inputs)
    model_tf(inputs)

    if use_bias:
      model_up.set_weights([weights, bias])
      model_tf.set_weights([weights, bias])
    else:
      model_up.set_weights([weights])
      model_tf.set_weights([weights])

    output_upstride, dinputs_upstride, dweights_upstride, dbias_upstride = get_output_and_gradients(model_up, model_up.kernel, inputs)
    output_tf, dinputs_tf, dweights_tf, dbias_tf = get_output_and_gradients(model_tf, model_tf.kernel, inputs)

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
