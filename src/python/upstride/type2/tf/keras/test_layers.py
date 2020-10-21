import unittest
import tensorflow as tf
import numpy as np
from .layers import TF2Upstride, Upstride2TF, Conv2D, Dense, DepthwiseConv2D
from upstride.type_generic.test import TestCase, apply_some_non_linearity


def setUpModule():
  """ Called by unittest to prepare the module
  """
  TestCase.setup()


def quaternion_mult_naive(tf_op, inputs, kernels, bias=(0, 0, 0, 0)):
  c1 = tf_op(inputs[0], kernels[0]) - tf_op(inputs[1], kernels[1]) - tf_op(inputs[2], kernels[2]) - tf_op(inputs[3], kernels[3]) + bias[0]
  c2 = tf_op(inputs[0], kernels[1]) + tf_op(inputs[1], kernels[0]) + tf_op(inputs[2], kernels[3]) - tf_op(inputs[3], kernels[2]) + bias[1]
  c3 = tf_op(inputs[0], kernels[2]) + tf_op(inputs[2], kernels[0]) + tf_op(inputs[3], kernels[1]) - tf_op(inputs[1], kernels[3]) + bias[2]
  c4 = tf_op(inputs[0], kernels[3]) + tf_op(inputs[3], kernels[0]) + tf_op(inputs[1], kernels[2]) - tf_op(inputs[2], kernels[1]) + bias[3]
  return [c1, c2, c3, c4]


def get_gradient_and_output_tf(inputs, function, kernels, bias=None):
  """ Computes the output and the gradients for a given TF-based quaternionic function.
      Transposes the returned arguments so that it corresponds to channels_first
  """
  dbias = None
  biases = []
  if bias is not None:
    for i in range(bias.shape[0]):
      biases.append(bias[i, :])
  with tf.GradientTape(persistent=True) as gt:
    gt.watch(kernels)
    for e in inputs:
      gt.watch(e)
    if bias is not None:
      for b in biases:
        gt.watch(b)
    outputs = function(inputs, kernels, biases)
    for i in range(len(outputs)):
      outputs[i] = apply_some_non_linearity(outputs[i])
    if bias is not None:
      dbias = [gt.gradient(outputs, b) for b in biases]
    outputs = [tf.transpose(outputs[i], [0, 3, 1, 2]) for i in range(len(outputs))]
  dinputs = [gt.gradient(outputs, e) for e in inputs]
  dinputs = [tf.transpose(dinputs[i], [0, 3, 1, 2]) for i in range(len(dinputs))]
  dkernels = gt.gradient(outputs, kernels)
  return dinputs, dkernels, dbias, outputs


def get_gradient_and_output_upstride(inputs, op):
  """ Computes the output and the gradients for a given upstride-based quaternionic op.
  """
  dbias = None
  with tf.GradientTape(persistent=True) as gt:
    gt.watch(op.kernel)
    gt.watch(inputs)
    if op.bias is not None:
      gt.watch(op.bias)
    outputs = op(inputs)
    outputs = apply_some_non_linearity(outputs)
  dinputs = gt.gradient(outputs, inputs)
  if op.bias is not None:
    dbias = gt.gradient(outputs, op.bias)
  dkernels = gt.gradient(outputs, op.kernel)
  return dinputs, dkernels, dbias, outputs


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
    tf_op = lambda i,k : tf.nn.conv2d(i, tf.transpose(k, [2, 3, 1, 0]), 1, "SAME") # upstride kernel is (O, I, H, W). TF expects (H, W, I, O)
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


class TestType2Conv2D(TestCase):
  """ Implements quaternion convolution unitary testing varying img_size, filter_size, 
      in_channels, out_channels, padding, strides, dilations and use_bias.
  """
  def run_test(self, img_size=224, filter_size=3, in_channels=3, out_channels=64, padding='SAME', strides=[1, 1], dilations=[1, 1], use_bias=False, batch_size=3):
    # initialize inputs
    py_inputs = [tf.cast(tf.random.uniform((batch_size, img_size, img_size, in_channels), dtype=tf.int32, minval=-5, maxval=5), dtype=tf.float32) for _ in range(4)]
    py_inputs_channels_first = [tf.transpose(_, [0, 3, 1, 2]) for _ in py_inputs]
    cpp_inputs = tf.concat(py_inputs_channels_first, axis=0)

    upstride_conv = Conv2D(filters=out_channels, kernel_size=filter_size, strides=strides, padding=padding, dilation_rate=dilations, use_bias=use_bias)

    upstride_conv(cpp_inputs) # runs a first time to initialize the kernel
    weights = tf.cast(tf.random.uniform(upstride_conv.kernel.shape, dtype=tf.int32, minval=-5, maxval=5), dtype=tf.float32)
    if use_bias:
      bias = tf.cast(tf.random.uniform(upstride_conv.bias.shape, dtype=tf.int32, minval=-5, maxval=5), dtype=tf.float32)
      upstride_conv.set_weights([weights, bias])
    else:
      bias = None
      upstride_conv.set_weights([weights])
    kernels = upstride_conv.kernel  # copies the quaternion kernel

    def py_conv(inputs, kernels, biases):
      # upstride kernel is (O, I, H, W). TF expects (H, W, I, O)
      tf_op = lambda i,k : tf.nn.conv2d(i, tf.transpose(k, [2, 3, 1, 0]), strides=strides, padding=padding, dilations=dilations)
      outputs = quaternion_mult_naive(tf_op, inputs, kernels)
      if biases != []:
        outputs = [tf.nn.bias_add(outputs[i], biases[i]) for i in range(len(outputs))]
      return outputs 

    dinput_test, dkernels_test, dbias_test, output_test = get_gradient_and_output_upstride(cpp_inputs, upstride_conv)
    dinput_ref, dkernels_ref, dbias_ref, output_ref = get_gradient_and_output_tf(py_inputs, py_conv, kernels, bias)

    output_ref_concat = tf.concat(output_ref, axis=0)
    dinput_ref_concat = tf.concat(dinput_ref, axis=0)
    dkernels_ref_concat = tf.concat(dkernels_ref, axis=0)

    # COMPARISONS
    self.assert_and_print(output_test, output_ref_concat, "TestType2Conv2D", "output")
    self.assert_and_print(dinput_test, dinput_ref_concat, "TestType2Conv2D", "dinput")
    self.assert_and_print(dkernels_test, dkernels_ref_concat, "TestType2Conv2D", "dweights")
    if use_bias:
      dbias_ref_concat = tf.convert_to_tensor(dbias_ref)
      self.assert_and_print(dbias_test, dbias_ref_concat, "TestType2Conv2D", "dbias")

  def test_type2_conv2d(self):
    try:
      tf.keras.backend.set_image_data_format('channels_first')  # FIXME We should find a proper way to pass 'channels_first'
      self.run_test(img_size=1, filter_size=1, in_channels=1, out_channels=1, padding='SAME', strides=[1, 1], dilations=[1, 1], batch_size=4)
      self.run_test(img_size=1, filter_size=1, in_channels=1, out_channels=1, padding='SAME', strides=[1, 1], dilations=[1, 1], use_bias=True, batch_size=4)
      self.run_test(img_size=4, filter_size=2, in_channels=1, out_channels=1, padding='SAME', strides=[1, 1], dilations=[1, 1], batch_size=5)
      self.run_test(img_size=4, filter_size=2, in_channels=1, out_channels=1, padding='SAME', strides=[1, 1], dilations=[1, 1], use_bias=True, batch_size=5)
      self.run_test(img_size=7, filter_size=3, in_channels=2, out_channels=2, padding='VALID')
      self.run_test(img_size=7, filter_size=3, in_channels=2, out_channels=2, padding='VALID', use_bias=True)
      self.run_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='VALID')
      self.run_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='VALID', use_bias=True)
      self.run_test(img_size=9, filter_size=6, in_channels=10, out_channels=5, padding='VALID', use_bias=True)
      self.run_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='SAME')
      self.run_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, padding='SAME', use_bias=True)
      self.run_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, strides=[2, 2])
      self.run_test(img_size=9, filter_size=3, in_channels=3, out_channels=16, strides=[2, 2], use_bias=True)
      self.run_test(img_size=56, filter_size=3, in_channels=3, out_channels=8, padding='VALID')
      self.run_test(img_size=56, filter_size=3, in_channels=3, out_channels=8, padding='VALID', use_bias=True)
      self.run_test(img_size=56, filter_size=4, in_channels=3, out_channels=8, padding='SAME')
      self.run_test(img_size=56, filter_size=4, in_channels=3, out_channels=8, padding='SAME', use_bias=True)
      self.run_test(img_size=56, filter_size=4, in_channels=3, out_channels=8, strides=[2, 2])
      self.run_test(img_size=56, filter_size=4, in_channels=3, out_channels=8, strides=[2, 2], use_bias=True)
      self.run_test(img_size=224, filter_size=3, in_channels=3, out_channels=48, strides=[2, 2], padding='VALID')
      self.run_test(img_size=224, filter_size=3, in_channels=3, out_channels=48, strides=[2, 2], padding='VALID', use_bias=True)
    finally:
      tf.keras.backend.set_image_data_format('channels_last')  # FIXME We should find a proper way to pass 'channels_last'


class TestType2DepthwiseConv2D(TestCase):
  def run_depthwise_conv2d_test(self, img_size=128, filter_size=3, channels=2, use_bias=False, padding='SAME', strides=[1, 1], dilations=[1, 1], batch_size=3):
    """ Runs a single depthwise convolution forward and backward and compares the result with TensorFlow output
    """
    # Initializes inputs: For tensorflow/python-based engine, list of 4 tensors. For phoenix, tf.concat(py_inputs)
    py_inputs = [tf.cast(tf.random.uniform((1, img_size, img_size, channels), dtype=tf.int32, minval=-5, maxval=5), dtype=tf.float32) for _ in range(4)]
    py_inputs_channels_first = [tf.transpose(_, [0, 3, 1, 2]) for _ in py_inputs]
    cpp_inputs = tf.concat(py_inputs_channels_first, axis=0)
    # Defines upstride model and initializes weights.
    model_up = DepthwiseConv2D(filter_size, strides, padding, data_format='channels_first', dilation_rate=dilations, bias_initializer='glorot_uniform', use_bias=use_bias)
    model_up(cpp_inputs)
    # Defines model_up.kernel as being equal to depthwise_kernel. It is necessary to use get_gradient_and_output_upstride()
    model_up.kernel = model_up.depthwise_kernel

    dinputs_upstride, dweights_upstride, dbias_upstride, output_upstride = get_gradient_and_output_upstride(cpp_inputs, model_up)

    # Defines model_tf as being the quaternion multiplication of depthwise_conv2d + bias
    def model_tf(inputs, depthwise_kernel_tf, biases):
      tf_op = lambda i,k : tf.keras.backend.depthwise_conv2d(i, k, strides=tuple(strides), padding=padding.lower(), dilation_rate=tuple(dilations), data_format='channels_last')
      outputs = quaternion_mult_naive(tf_op, inputs, depthwise_kernel_tf)
      if biases != []:
        outputs = [tf.nn.bias_add(outputs[i], biases[i]) for i in range(len(outputs))]
      return outputs

    # Defines depthwise_kernel_tf from model_up.depthwise_kernel as being a list and transposing it elements from iohw to hwio
    depthwise_kernel_tf = []
    for i in range(4):
      depthwise_kernel_tf.append(tf.transpose(model_up.depthwise_kernel[i, :], [2, 3, 0, 1]))

    dinputs_tf_list, dweights_tf_list, dbias_tf_list, output_tf_list = get_gradient_and_output_tf(py_inputs, model_tf, depthwise_kernel_tf, bias=model_up.bias if use_bias else None)

    # Transforms the list of tensors from python/tensorflow-based engine into tensors, for comparing them later on
    output_tf = tf.concat(output_tf_list, axis=0)
    dinputs_tf = tf.concat(dinputs_tf_list, axis=0)
    dweights_tf = tf.transpose(tf.stack(dweights_tf_list, axis=0), [0, 3, 4, 1, 2])
    if use_bias:
      dbias_tf = tf.stack(dbias_tf_list, axis=0)

    self.assert_and_print(output_upstride, output_tf, "Type2DepthwiseConv2DBwd", "output")
    self.assert_and_print(dweights_upstride, dweights_tf, "Type2DepthwiseConv2DBwd", "dfilter")
    self.assert_and_print(dinputs_upstride, dinputs_tf, "Type2DepthwiseConv2DBwd", "dinput")
    if use_bias:
      self.assert_and_print(dbias_upstride, dbias_tf, "Type2DepthwiseConv2DBwd", "dbias")

  def test_conv2dgrad(self):
    self.run_depthwise_conv2d_test(img_size=5, filter_size=3, channels=4, padding='VALID', batch_size=2)
    self.run_depthwise_conv2d_test(img_size=5, filter_size=3, channels=4, padding='VALID', batch_size=4, use_bias=True)
    self.run_depthwise_conv2d_test(img_size=9, filter_size=3, channels=3, padding='VALID', batch_size=5)
    self.run_depthwise_conv2d_test(img_size=9, filter_size=3, channels=3, padding='VALID', use_bias=True)
    self.run_depthwise_conv2d_test(img_size=9, filter_size=3, channels=3, padding='SAME')
    self.run_depthwise_conv2d_test(img_size=9, filter_size=3, channels=3, padding='SAME', use_bias=True)
    self.run_depthwise_conv2d_test(img_size=9, filter_size=10, channels=12, padding='SAME', use_bias=True)
    self.run_depthwise_conv2d_test(img_size=9, filter_size=3, channels=3, strides=[2, 2])
    self.run_depthwise_conv2d_test(img_size=9, filter_size=3, channels=3, strides=[2, 2], use_bias=True)
    self.run_depthwise_conv2d_test(img_size=32, filter_size=4, channels=3, padding='VALID')
    self.run_depthwise_conv2d_test(img_size=32, filter_size=4, channels=3, padding='VALID', use_bias=True)
    self.run_depthwise_conv2d_test(img_size=32, filter_size=4, channels=3, padding='SAME')
    self.run_depthwise_conv2d_test(img_size=32, filter_size=4, channels=3, padding='SAME', use_bias=True)
    self.run_depthwise_conv2d_test(img_size=32, filter_size=4, channels=3, strides=[2, 2])
    self.run_depthwise_conv2d_test(img_size=32, filter_size=4, channels=3, strides=[2, 2], use_bias=True)

class TestType2Dense(TestCase):
  def get_gradient_and_output(self, inputs, function, kernels, bias):
    if type(inputs) == list: # TENSORFLOW
      with tf.GradientTape(persistent=True) as gt:
        gt.watch(kernels)
        if bias is not None:
          gt.watch(bias)
        for e in inputs:
          gt.watch(e)
        outputs = tf.concat(function(inputs, kernels), axis=0)
        outputs = apply_some_non_linearity(outputs)
      dinputs = tf.concat([gt.gradient(outputs, e) for e in inputs], axis=0)
      dkernels = tf.concat(gt.gradient(outputs, kernels), axis=0)
    else: # UPSTRIDE
      with tf.GradientTape(persistent=True) as gt:
        gt.watch([inputs, kernels])
        if bias is not None:
          gt.watch(bias)
        outputs = function(inputs, kernels)
        outputs = apply_some_non_linearity(outputs)
      dinputs, dkernels = gt.gradient(outputs, [inputs, kernels])
    if bias is not None:
      dbias = gt.gradient(outputs, bias)
    else:
      dbias = None
    return dinputs, dkernels, dbias, outputs

  def run_test(self, batch_size, in_features, out_features, use_bias=False):
    """ Runs a single dense and compares the result with TensorFlow output """
    py_inputs = [tf.random.uniform((batch_size, in_features), dtype=tf.float32, minval=-1, maxval=1) for _ in range(4)]
    cpp_inputs = tf.concat(py_inputs, axis=0)
    model = Dense(out_features, bias_initializer='glorot_uniform', use_bias=use_bias)
    model(cpp_inputs) # runs a first time to initialize the kernel and the bias, according to use_bias

    def cpp_dense(inputs, kernels):
      return model(inputs)

    def py_dense(inputs, kernels):
      tf_op = lambda i,k : tf.linalg.matmul(i, k)
      output = quaternion_mult_naive(tf_op, inputs, kernels)
      if use_bias:
        for i in range(4):
          output[i] = tf.nn.bias_add(output[i], model.bias[i, :])
      return output

    dinput_test, dkernels_test, dbias_test, output_test = self.get_gradient_and_output(cpp_inputs, cpp_dense, model.kernel, model.bias if use_bias else None)
    dinput_ref, dkernels_ref, dbias_ref, output_ref = self.get_gradient_and_output(py_inputs, py_dense, model.kernel, model.bias if use_bias else None)

    self.assert_and_print(output_test, output_ref, "Type2Dense")
    self.assert_and_print(dinput_test, dinput_ref, "Type2Dense", "dinput")
    self.assert_and_print(dkernels_test, dkernels_ref, "Type2Dense", "dkernels")
    if dbias_ref is not None:
      self.assert_and_print(dbias_test, dbias_ref, "Type2Dense", "dbias")

  def test_dense(self):
    self.run_test(batch_size=1, in_features=1, out_features=1)
    self.run_test(batch_size=2, in_features=3, out_features=4, use_bias=True)
    self.run_test(batch_size=64, in_features=64, out_features=10)
    self.run_test(batch_size=64, in_features=64, out_features=10, use_bias=True)
    self.run_test(batch_size=2, in_features=4, out_features=3, use_bias=True)
    self.run_test(batch_size=128, in_features=100, out_features=10)
    self.run_test(batch_size=128, in_features=100, out_features=10, use_bias=True)