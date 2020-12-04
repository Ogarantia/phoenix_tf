import unittest
import tensorflow as tf
import numpy as np
from .. import utils

def setUpModule():
  """ Prepares the test module to be executed.
  Running tests without preparation may cause a cuDNN crash (dunno why).
  """
  # allow memory growth
  gpus = tf.config.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  # call a dummy function to get cuDNN handle created
  from .custom_op import upstride_ops
  upstride_ops.wait()

def transpose_to_channel_first(tensor):
  """ Transposes a tensor from channel last to channel first format
  """
  rank = len(tensor.shape)
  permutation = [0, rank - 1] + list(range(1, rank - 1))
  return tf.transpose(tensor, permutation)

def random_integer_tensor(shape, dtype=None):
  """ Generates a random tensor containing integer values
  """
  tensor = tf.random.uniform(shape, -4, +4, dtype=tf.int32)
  return tf.cast(tensor, dtype or tf.float32)

def apply_some_non_linearity(x):
  """ Applies some non linearity to the input x so that the unitary tests are robust wrt to the gradient.
  """
  return tf.where(tf.math.abs(x) > 1, tf.math.abs(x) - 0.5, 0.5*x**2)

class CliffordProductLayer(tf.keras.layers.Layer):
  """ Wraps a keras multiplicative operation to be applied within Clifford product
  """
  def __init__(self, clifford_product, layer_class, **kwargs):
    """ Creates a wrapper for a keras class
    :clifford_product: a Clifford product specification
    :layer_class:      scalar keras layer class to wrap, e.g. tf.keras.layers.Conv2D
    Keyword arguments are forwarded to the wrapped layer class constructor.
    """
    super().__init__()
    # make sure there is no use_bias
    if 'use_bias' in kwargs and kwargs['use_bias'] == True:
      raise ValueError("Bias addition cannot be used with CliffordProductLayer")
    # keep clifford_product
    self.clifford_product = clifford_product
    # create layers
    name = kwargs.pop("name", None)
    self.layers = [
      layer_class(name=name + f"_{i}" if name else None, **kwargs)
      for i in range(clifford_product.dim)
    ]

  def build(self, input_shape):
    # build layers
    shape = input_shape.as_list()
    shape[0] //= self.clifford_product.dim
    for layer in self.layers:
      layer.build(shape)

  def call(self, inputs):
    # split the input tensor along the batch dimension onto blades
    inputs = tf.split(inputs, self.clifford_product.dim, axis=0)
    # define the operation to apply: j-th operation is applied to i-th layer
    op = lambda i, j: self.layers[j](inputs[i])
    # run Clifford product
    outputs = self.clifford_product.apply(op)
    # return outputs concatenated along the batch dimension
    return tf.concat(outputs, axis=0)


# FIXME: consider removing this class
class TestCase(unittest.TestCase):
  """ Base class for unittests containing handy stuff """
  DEFAULT_ERROR_THRESHOLD = 5e-3
  HALF_FLOAT_ERROR_THRESHOLD = 5e-2

  @staticmethod
  def setup():
    setUpModule()

  def assert_and_print(self, test, ref, function="", variable=""):
    """ Prints stuff and raises exception if two tensors are too different
    """
    err = tf.math.reduce_max(tf.math.abs(test - ref))
    threshold=self.HALF_FLOAT_ERROR_THRESHOLD if ref.dtype == tf.float16 else self.DEFAULT_ERROR_THRESHOLD
    self.assertLess(err, threshold, f"Absolute {variable} difference with the reference is too big: {err}")
    print(f'[{function}] Absolute {variable} difference:', err.numpy())

class TestBase:
  """ Basic class for tests containing handy utilities
  """
  def assert_positive_range(self, tensor, threshold):
    """ Asserts on a positive range of a tensor, i.e., its values have some meaningful diversity
    """
    rng = tf.reduce_max(tensor) - tf.reduce_min(tensor)
    self.assertGreater(rng.numpy(), threshold)

  def assert_zero_integer_difference(self, tensor1, tensor2):
    self.assert_positive_range(tensor1, 0.9)
    diff = tf.round(tensor1 - tensor2)
    diff = tf.reduce_sum(diff)
    self.assertEqual(diff.numpy(), 0)

  @staticmethod
  def uses_gpu():
    """ Returns True if TF uses GPU
    """
    # good old tf.test.gpu_device_name() segfaults when called in unittest decorator before fp16 tests. Yep.
    return tf.config.list_physical_devices('GPU') != []

class Conv2DTestBase(TestBase):
  def setup(self, clifford_product, ref_op_class, test_op_class, kernel_initializer):
    """ Type-agnostic convolution forward and backward pass test set
    :clifford_product:    Clifford product specification defining the algebra being tested
    :ref_op_class:        reference scalar operation class, e.g. tf.keras.layers.Conv2D
    :test_op_class:       test operation class implementing the convolution for the algebra being tested
    :kernel_initializer:  argument name to pass to the layer, 'kernel_initializer' or 'depthwise_initializer'
                          depending on which convolution operation is used
    """
    self.clifford_product = clifford_product
    self.ref_op_class = ref_op_class
    self.test_op_class = test_op_class
    self.kernel_initializer = kernel_initializer

  def run_conv2d_test_instance(self, test_shape, dtype=tf.float32, **kwargs):
    """ Runs a single instance of a forward and backward pass test for a given parameter set.
    The reference operation is run in channel-last format for compatibility with CPU backend.
    The test operation is run in channel-first.
    :test_shape:          test input shape in channel-last format (NHWC)
    :dtype:               scalar data type of the test batch
    """
    assert self.kernel_initializer not in kwargs, 'kernel_initializer option is not supported in this test'
    assert 'bias_initializer' not in kwargs, 'bias_initializer option is not supported in this test'
    assert 'data_format' not in kwargs, 'data_format option is not supported in this test'

    # generate channel-last random input
    input_tensor = random_integer_tensor((test_shape[0] * self.clifford_product.dim,) + test_shape[1:], dtype=dtype)

    # prepare bias
    use_bias = kwargs.pop('use_bias', False)
    if use_bias:
      bias_length = kwargs.get('filters', test_shape[-1])
      bias = random_integer_tensor((self.clifford_product.dim, bias_length), dtype=dtype)

    # construct reference operation
    kwargs[self.kernel_initializer] = random_integer_tensor
    ref_op = CliffordProductLayer(self.clifford_product, self.ref_op_class,
      data_format='channels_last',
      **kwargs)

    # run once to get it ready
    ref_op(tf.zeros_like(input_tensor))

    # get kernels TF variables
    kernels = [layer.weights[0] for layer in ref_op.layers]

    # run and watch on gradients
    with tf.GradientTape(persistent=True) as gt:
      gt.watch(input_tensor)
      gt.watch(kernels)
      # run reference op
      ref_output = ref_op(input_tensor)
      # add bias if any
      if use_bias:
        ref_output = tf.split(ref_output, self.clifford_product.dim, axis=0)
        for i in range(len(ref_output)):
          ref_output[i] = tf.nn.bias_add(ref_output[i], bias[i,:])
        ref_output = tf.concat(ref_output, axis=0)
      # apply a non-linearity
      ref_output = apply_some_non_linearity(ref_output)

    # get gradients
    ref_input_grad = gt.gradient(ref_output, input_tensor)
    ref_kernel_grad = [gt.gradient(ref_output, k) for k in kernels]

    # create test operation
    kwargs[self.kernel_initializer] = lambda shape, dtype: kernels.pop(0).numpy()
    test_op = self.test_op_class(
      use_bias=use_bias,
      data_format='channels_first',
      bias_initializer=lambda shape, dtype: tf.squeeze(bias) if use_bias else [],
      **kwargs)
    test_op.require_input_grad = True

    # transpose input to channel first
    input_tensor = transpose_to_channel_first(input_tensor)

    # run the test operation once to get it ready
    test_op(tf.zeros_like(input_tensor))

    # get its kernel
    kernel = test_op.weights[0]

    # run the test operation and watch on gradients
    with tf.GradientTape(persistent=True) as gt:
      gt.watch(input_tensor)
      gt.watch(kernel)
      test_output = test_op(input_tensor)
      test_output = apply_some_non_linearity(test_output)

    # get gradients
    test_input_grad = gt.gradient(test_output, input_tensor)
    test_kernel_grad = gt.gradient(test_output, kernel)

    # do transpositions
    ref_output = transpose_to_channel_first(ref_output)
    ref_input_grad = transpose_to_channel_first(ref_input_grad)
    ref_kernel_grad = tf.transpose(ref_kernel_grad, utils.permutation("nHWIO", "nOIHW"))

    # assert on the difference
    self.assert_zero_integer_difference(ref_output, test_output)
    self.assert_zero_integer_difference(ref_input_grad, test_input_grad)
    self.assert_zero_integer_difference(ref_kernel_grad, test_kernel_grad)

  def run_fp16_conv2d_test_instance(self, test_shape, **kwargs):
    """ Runs a half-precision floating-point forward and backward pass test
    """
    tf.keras.backend.set_floatx('float16')
    try:
      self.run_conv2d_test_instance(test_shape, dtype=tf.float16, **kwargs)
    finally:
      tf.keras.backend.set_floatx('float32')


class Conv2DTestSet(Conv2DTestBase):
  """ Test set for regular Conv2D operation
  """
  def setup(self, clifford_product, test_op_class):
    super().setup(clifford_product, tf.keras.layers.Conv2D, test_op_class, 'kernel_initializer')

  def test_basic(self):
    """ Basic Conv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 5, 5, 8), filters=32, kernel_size=3)

  def test_bigger_batch(self):
    """ Conv2D with bigger batch test """
    self.run_conv2d_test_instance(test_shape=(5, 3, 3, 8), filters=32, kernel_size=3)

  def test_pointwise(self):
    """ Pointwise Conv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 5, 5, 8), filters=32, kernel_size=1)

  def test_strided(self):
    """ Strided Conv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 5, 5, 64), filters=16, strides=(2, 3), kernel_size=2)

  def test_dilated(self):
    """ Dilated Conv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 7, 7, 16), filters=16, dilation_rate=(2, 3), kernel_size=3)

  def test_non_square(self):
    """ Non-square image Conv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 15, 3, 64), filters=16, kernel_size=2)

  def test_bias(self):
    """ Biased Conv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 3, 3, 64), filters=32, kernel_size=1, use_bias=True)

  def test_padded(self):
    """ Padded Conv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 5, 5, 32), filters=32, kernel_size=3, padding='same')

  def test_padded_strided(self):
    """ Padded strided Conv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 7, 7, 16), filters=32, kernel_size=3, strides=2, padding='same')

  def test_padded_dilated(self):
    """ Padded dilated Conv2D test """
    self.run_conv2d_test_instance(test_shape=(2, 7, 7, 16), filters=32, kernel_size=3, dilation_rate=(2, 2), padding='same')

  @unittest.skipIf(not TestBase.uses_gpu(), "grouped conv not supported on CPU")
  def test_grouped(self):
    """ Group Conv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 5, 5, 64), filters=48, groups=4, kernel_size=3)

  @unittest.skipIf(not TestBase.uses_gpu(), "fp16 not supported on CPU")
  def test_fp16(self):
    """ Half-precision floating point Conv2D test """
    self.run_fp16_conv2d_test_instance(test_shape=(2, 5, 5, 8), filters=16, kernel_size=3, use_bias=True)


class PointwiseConv2DTestSet(Conv2DTestBase):
  """ Test set for pointwise Conv2D operation
  """
  def setup(self, clifford_product, test_op_class):
    super().setup(clifford_product, tf.keras.layers.Conv2D, test_op_class, 'kernel_initializer')

  def test_basic(self):
    """ Basic PointwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 3, 3, 64), filters=32, kernel_size=1)

  def test_larger_batch(self):
    """ PointwiseConv2D with larger batch test """
    self.run_conv2d_test_instance(test_shape=(5, 3, 3, 32), filters=32, kernel_size=1)

  def test_batch_largest(self):
    """ PointwiseConv2D with batch size as the largest parameter test """
    self.run_conv2d_test_instance(test_shape=(11, 3, 3, 8), filters=8, kernel_size=1)

  def test_non_square_high(self):
    """ Non-square high image PointwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 14, 3, 32), filters=16, kernel_size=1)

  def test_non_square_wide(self):
    """ Non-square wide image PointwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 3, 12, 32), filters=16, kernel_size=1)

  def test_irregular_params(self):
    """ Irregular size parameters PointwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(7, 11, 5, 3), filters=13, kernel_size=1)

  def test_bias(self):
    """ Biased PointwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 3, 3, 64), filters=32, kernel_size=1, use_bias=True)

  def test_bias_larger_batch(self):
    """ Biased PointwiseConv2D with larger batch test """
    self.run_conv2d_test_instance(test_shape=(5, 3, 3, 16), filters=32, kernel_size=1, use_bias=True)

  def test_few_filters(self):
    """ Few filters PointwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(4, 7, 7, 64), filters=3, kernel_size=1)

  def test_few_input_channels(self):
    """ Few input channels PointwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(4, 7, 7, 3), filters=64, kernel_size=1)

  def test_few_channels(self):
    """ Few channels PointwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(6, 7, 7, 3), filters=3, kernel_size=1)

  def test_even_height(self):
    """ Image with even height PointwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(3, 4, 3, 16), filters=16, kernel_size=1)

  def test_even_width(self):
    """ Image with even width PointwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(3, 3, 4, 16), filters=16, kernel_size=1)

  def test_even_image(self):
    """ Image with even width and height PointwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(3, 4, 4, 16), filters=16, kernel_size=1)

  @unittest.skipIf(not TestBase.uses_gpu(), "fp16 not supported on CPU")
  def test_fp16(self):
    """ Half-precision floating point PointwiseConv2D test """
    self.run_fp16_conv2d_test_instance(test_shape=(3, 5, 5, 8), filters=16, kernel_size=1)

  @unittest.skipIf(not TestBase.uses_gpu(), "fp16 not supported on CPU")
  def test_fp16_bias(self):
    """ Half-precision floating point biased PointwiseConv2D test """
    self.run_fp16_conv2d_test_instance(test_shape=(3, 4, 4, 8), filters=16, kernel_size=1, use_bias=True)


class DepthwiseConv2DTestSet(Conv2DTestBase):
  """ Test set for depthwise Conv2D operation
  """
  def setup(self, clifford_product, test_op_class):
    super().setup(clifford_product, tf.keras.layers.DepthwiseConv2D, test_op_class, 'depthwise_initializer')

  def test_basic(self):
    """ Basic DepthwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 5, 5, 32), kernel_size=3)

  def test_bigger_batch(self):
    """ DepthwiseConv2D with bigger batch test """
    self.run_conv2d_test_instance(test_shape=(5, 3, 3, 16), kernel_size=3)

  def test_strided(self):
    """ Strided DepthwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 11, 11, 32), strides=3, kernel_size=4)

  def test_dilated(self):
    """ Dilated DepthwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 7, 7, 16), dilation_rate=(2, 3), kernel_size=3)

  def test_non_square(self):
    """ Non-square image DepthwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 15, 3, 64), kernel_size=2)

  def test_bias(self):
    """ Biased DepthwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 3, 3, 64), kernel_size=1, use_bias=True)

  def test_padded(self):
    """ Padded DepthwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 5, 5, 32), kernel_size=3, padding='same')

  def test_padded_strided(self):
    """ Padded strided DepthwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(1, 7, 7, 16), kernel_size=3, strides=2, padding='same')

  def test_padded_dilated(self):
    """ Padded dilated DepthwiseConv2D test """
    self.run_conv2d_test_instance(test_shape=(2, 7, 7, 16), kernel_size=3, dilation_rate=(2, 2), padding='same')

  @unittest.skipIf(not TestBase.uses_gpu(), "fp16 not supported on CPU")
  def test_fp16(self):
    """ Half-precision floating point DepthwiseConv2D test """
    self.run_fp16_conv2d_test_instance(test_shape=(2, 3, 3, 4), kernel_size=2, use_bias=True)


class DenseTestSet(TestBase):
  def setup(self, clifford_product, test_op_class):
    """ Type-agnostic Dense forward and backward pass test set
    :clifford_product:    Clifford product specification defining the algebra being tested
    :test_op_class:       test Dense operation class implementing the convolution for the algebra being tested
    """
    self.clifford_product = clifford_product
    self.test_op_class = test_op_class

  def run_dense_test_instance(self, test_shape, dtype=tf.float32, **kwargs):
    """ Runs a single instance of a forward and backward pass test for a given parameter set.
    :test_shape:          test input shape of rank 2 (in NC format)
    :dtype:               scalar data type of the test batch
    """
    assert 'kernel_initializer' not in kwargs, 'kernel_initializer option is not supported in this test'
    assert 'bias_initializer' not in kwargs, 'bias_initializer option is not supported in this test'

    # generate channel-last random input
    input_tensor = random_integer_tensor((test_shape[0] * self.clifford_product.dim,) + test_shape[1:], dtype=dtype)

    # prepare bias
    use_bias = kwargs.pop('use_bias', False)
    if use_bias:
      bias = random_integer_tensor((self.clifford_product.dim, kwargs.get('units')), dtype=dtype)

    # construct reference operation
    kwargs['kernel_initializer'] = random_integer_tensor
    ref_op = CliffordProductLayer(self.clifford_product, tf.keras.layers.Dense, **kwargs)

    # run once to get it ready
    ref_op(tf.zeros_like(input_tensor))

    # get kernels TF variables
    kernels = [layer.weights[0] for layer in ref_op.layers]

    # run and watch on gradients
    with tf.GradientTape(persistent=True) as gt:
      gt.watch(input_tensor)
      gt.watch(kernels)
      # run reference op
      ref_output = ref_op(input_tensor)
      # add bias if any
      if use_bias:
        ref_output = tf.split(ref_output, self.clifford_product.dim, axis=0)
        for i in range(len(ref_output)):
          ref_output[i] = tf.nn.bias_add(ref_output[i], bias[i,:])
        ref_output = tf.concat(ref_output, axis=0)
      # apply a non-linearity
      ref_output = apply_some_non_linearity(ref_output)

    # get gradients
    ref_input_grad = gt.gradient(ref_output, input_tensor)
    ref_kernel_grad = [gt.gradient(ref_output, k) for k in kernels]

    # prepare test kernel initializer: reference kernels are stacked along the outermost dimension
    kernels = tf.squeeze(tf.stack(kernels, axis=0))
    kwargs['kernel_initializer'] = lambda shape, dtype: kernels

    # create test operation
    test_op = self.test_op_class(use_bias=use_bias,
                                 bias_initializer=lambda shape, dtype: tf.squeeze(bias) if use_bias else [],
                                 **kwargs)
    test_op.require_input_grad = True

    # run the test operation once to get it ready
    test_op(tf.zeros_like(input_tensor))

    # get its kernel
    kernel = test_op.weights[0]

    # run the test operation and watch on gradients
    with tf.GradientTape(persistent=True) as gt:
      gt.watch(input_tensor)
      gt.watch(kernel)
      test_output = test_op(input_tensor)
      test_output = apply_some_non_linearity(test_output)

    # get gradients
    test_input_grad = gt.gradient(test_output, input_tensor)
    test_kernel_grad = gt.gradient(test_output, kernel)

    # assert on the difference
    self.assert_zero_integer_difference(ref_output, test_output)
    self.assert_zero_integer_difference(ref_input_grad, test_input_grad)
    self.assert_zero_integer_difference(ref_kernel_grad, test_kernel_grad)

  def run_fp16_dense_test_instance(self, test_shape, **kwargs):
    """ Runs a half-precision floating-point forward and backward pass test
    """
    tf.keras.backend.set_floatx('float16')
    try:
      self.run_dense_test_instance(test_shape, dtype=tf.float16, **kwargs)
    finally:
      tf.keras.backend.set_floatx('float32')

  def test_basic(self):
    """ Basic Dense test """
    self.run_dense_test_instance(test_shape=(1, 8), units=4)

  def test_bigger_batch(self):
    """ Bigger batch Dense test """
    self.run_dense_test_instance(test_shape=(2, 4), units=8)

  def test_biased(self):
    """ Biased Dense test """
    self.run_dense_test_instance(test_shape=(1, 12), units=8, use_bias=True)

  @unittest.skipIf(not TestBase.uses_gpu(), "fp16 not supported on CPU")
  def test_fp16(self):
    """ Half-precision floating point Dense test """
    self.run_fp16_dense_test_instance(test_shape=(2, 4), units=8, use_bias=True)