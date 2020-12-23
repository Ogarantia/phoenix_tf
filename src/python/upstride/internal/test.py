import unittest
import tensorflow as tf
from upstride import utils
from upstride.internal.layers import TYPE0


def setUpModule():
  """ Prepares the test module to be executed.

  Running tests without preparation may cause a cuDNN crash (dunno why).
  """
  # allow memory growth
  gpus = tf.config.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  # call a dummy function to get cuDNN handle created
  from upstride.internal.custom_ops import upstride_ops
  upstride_ops.wait()


def gpu_visible():
  """ Returns True if TF sees GPU
  """
  # good old tf.test.gpu_device_name() segfaults when called in unittest decorator before fp16 tests. Yep.
  return tf.config.list_physical_devices('GPU') != []


def transpose_to_channel_first(tensor):
  """ Transposes a tensor from channel last to channel first format
  """
  rank = len(tensor.shape)
  permutation = [0, rank - 1] + list(range(1, rank - 1))
  return tf.transpose(tensor, permutation)


def random_integer_tensor(shape, dtype=tf.float32):
  """ Generates a random tensor containing integer values
  """
  tensor = tf.random.uniform(shape, -4, +4, dtype=tf.int32)
  return tf.cast(tensor, dtype)


def random_float_tensor(shape, dtype=tf.float32):
  """ Generates a random tensor containing float values
  """
  tensor = tf.random.uniform(shape, -4, +4, dtype=tf.float32)
  return tf.cast(tensor, dtype)


def apply_some_non_linearity(x):
  """ Applies some non linearity to the input x so that the unitary tests are robust wrt to the gradient.

  >>> x = tf.constant([-2.5, -0.5, 0, 0.5, 2.5])
  >>> x_applied = apply_some_non_linearity(x)
  >>> x_applied_ref = tf.constant([2, 0.125, 0, 0.125, 2])
  >>> bool(tf.reduce_all(x_applied == x_applied_ref))
  True
  """
  return tf.where(tf.math.abs(x) > 1, tf.math.abs(x) - 0.5, 0.5*x**2)


def assert_positive_range(tensor, threshold=0.9):
  """ Asserts on a positive range of a tensor, i.e., its values have some meaningful diversity
  """
  if tf.size(tensor[0]) > 1:
    rng = tf.reduce_max(tensor) - tf.reduce_min(tensor)
    assert rng.numpy() > threshold


def assert_zero_integer_difference(tensor1, tensor2):
  """ Asserts integer tensors are equal after rounding
  """
  assert_positive_range(tensor1)
  diff = tf.round(tensor1 - tensor2)
  diff = tf.reduce_sum(diff)
  assert diff.numpy() == 0


def assert_small_float_difference(tensor1, tensor2, relative_error_threshold):
  """ Asserts float tensors differ by no more than threshold scaled by the values checked
  """
  abs_diff = tf.abs(tensor1 - tensor2)
  abs_max_tensors = tf.abs(tf.maximum(tensor1, tensor2))
  threshold = relative_error_threshold * (1 + abs_max_tensors)
  assert tf.reduce_all(abs_diff < threshold)


class CliffordProductLayer(tf.keras.layers.Layer):
  """ Wraps a keras multiplicative operation to be applied within Clifford product
  """
  def __init__(self, clifford_product, layer_class, bias, **kwargs):
    """ Creates a wrapper for a keras class
    :clifford_product:    a Clifford product specification
    :layer_class:         scalar keras layer class to wrap, e.g. tf.keras.layers.Conv2D
    :bias:                bias tensor (if kwargs['use_bias'] is True) or None
    Keyword arguments are forwarded to the wrapped layer class constructor.
    """
    super().__init__()
    # keep clifford_product
    self.clifford_product = clifford_product
    # apply bias outside of layers
    self.bias = bias
    self.use_bias = kwargs.pop('use_bias', False)
    # create layers
    name = kwargs.pop("name", None)
    self.layers = [
      layer_class(name=name + f"_{i}" if name else None, use_bias=False, **kwargs)
      for i in range(self.clifford_product.dim)
    ]

  def build(self, input_shape):
    # build layers
    shape = input_shape.as_list()
    shape[0] //= self.clifford_product.dim
    for layer in self.layers:
      layer.build(shape)

  def set_kernels(self, kernels):
    # set weights in each layer
    for i in range(self.clifford_product.dim):
      self.layers[i].set_weights([kernels[i]])

  def call(self, inputs):
    # split the input tensor along the batch dimension onto blades
    inputs = tf.split(inputs, self.clifford_product.dim, axis=0)
    # define the operation to apply: j-th operation is applied to i-th layer
    op = lambda i, j: self.layers[j](inputs[i])
    # run Clifford product
    outputs = self.clifford_product.apply(op)
    # apply bias if necessary
    if self.use_bias:
      for i in range(self.clifford_product.dim):
        outputs[i] = tf.nn.bias_add(outputs[i], self.bias[i])
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
  def setup(self, clifford_product, test_op_class, ref_op_class, kernel_initializer, op_requires_data_format, underlying_dtype=tf.int32):
    """ Type-agnostic test base
    :clifford_product:        Clifford product specification defining the algebra being tested
    :ref_op_class:            reference scalar operation class, e.g. tf.keras.layers.Conv2D
    :test_op_class:           tested operation class implementing the convolution for the algebra being tested
    :kernel_initializer:      argument name to pass to the layer, 'kernel_initializer' or 'depthwise_initializer'
                              depending on which convolution operation is used
    :op_requires_data_format: if 'true', the operations to be created require data_format as a parameter
    :underlying_dtype:        underlying tensor type used to generate values
    """
    self.clifford_product = clifford_product
    self.test_op_class = test_op_class
    self.ref_op_class = ref_op_class
    self.kernel_initializer = kernel_initializer
    self.underlying_dtype = underlying_dtype
    self.op_requires_data_format = op_requires_data_format
    self.random_tensor = random_integer_tensor if underlying_dtype is tf.int32 else random_float_tensor
    self.require_input_grad = True
    self.receives_type0_inputs = False

  def assert_equal_tensors(self, tensor1, tensor2):
    """ Assert that two tensors are numerically (almost) equal
    :tensor1:             first tensor to compare
    :tensor2:             second tensor to compare
    """
    if self.underlying_dtype is tf.int32:
      assert_zero_integer_difference(tensor1, tensor2)
    else:
      RELATIVE_ERROR_THRESHOLD = 2e-3
      assert_small_float_difference(tensor1, tensor2, RELATIVE_ERROR_THRESHOLD)

  def verify_kwargs(self, **kwargs):
    """ Verify that certain parameters do not appear in kwargs
    """
    assert self.kernel_initializer not in kwargs, 'kernel_initializer option is not supported in this test'
    assert 'bias_initializer' not in kwargs, 'bias_initializer option is not supported in this test'
    assert 'data_format' not in kwargs, 'data_format option is not supported in this test'

  def create_input_tensor(self, test_shape, dtype):
    """ Create a random input tensor suitable for the used algebra
    :test_shape:          test input data shape before expanding for the used algebra
    :dtype:               the actual type of the input data tensor
    """
    return self.random_tensor((test_shape[0] * self.clifford_product.dim,) + test_shape[1:], dtype=dtype)

  def construct_test_operation(self, **kwargs):
    """ Construct tested operation
    """
    kwargs[self.kernel_initializer] = self.random_tensor
    if self.op_requires_data_format:
      kwargs['data_format'] = self.test_data_format
    if kwargs['use_bias']:
      kwargs['bias_initializer'] = self.random_tensor

    test_op = self.test_op_class(**kwargs)
    test_op.require_input_grad = self.require_input_grad
    test_op.receives_type0_inputs = self.receives_type0_inputs
    return test_op

  def construct_reference_operation(self, bias, **kwargs):
    """ Construct reference operation
    :bias:                bias tensor
    """
    if self.op_requires_data_format:
      kwargs['data_format'] = self.ref_data_format
    ref_op = CliffordProductLayer(
      self.clifford_product,
      self.ref_op_class,
      bias,
      **kwargs)
    return ref_op

  def get_ref_compliant_kernels(self, test_op):
    """ Prepare kernels initialized by the tested operation so that they can be used with the reference operation
    :test_op:             tested operation, which was run already
    """
    kernels = test_op.weights[0]
    if test_op.upstride_datatype == TYPE0:
      kernels = tf.expand_dims(kernels, axis=0)
    return kernels

  def get_ref_compliant_bias(self, test_op):
    """ If necessary, prepare bias initialized by the tested operation so that it can be used with the reference operation
    :test_op:             tested operation, which was run already
    """
    bias = None
    if test_op.use_bias:
      bias = test_op.weights[1]
      if test_op.upstride_datatype == TYPE0:
        bias = tf.expand_dims(bias, axis=0)
    return bias

  def prepare_test_op(self, input_tensor, **kwargs):
    """ Create tested operation and run it once so that its weights are ready
    :input_tensor:        input data tensor
    """
    test_op = self.construct_test_operation(**kwargs)
    test_op(tf.zeros_like(input_tensor))
    return test_op

  def prepare_ref_op(self, input_tensor, test_op, **kwargs):
    """ Create reference operation and set its weights, using the tensors initialized with tested operation
    :input_tensor:        input data tensor
    :test_op:             tested operation, which was run already
    """
    kernels = self.get_ref_compliant_kernels(test_op)
    bias = self.get_ref_compliant_bias(test_op)
    ref_op = self.construct_reference_operation(bias, **kwargs)
    ref_op(tf.zeros_like(input_tensor))
    ref_op.set_kernels(kernels)
    return ref_op

  def compute_output_and_gradients(self, op, input_tensor, test):
    """ Compute the forward and backward pass tensors for the given operation
    :op:                  operation to run
    :input_tensor:        input data tensor
    :test:                if 'true', op is the tested operation, else the reference one
    """
    kernels = [op.weights[0]] if test else [layer.weights[0] for layer in op.layers]

    with tf.GradientTape(persistent=True) as gt:
      if self.require_input_grad:
        gt.watch(input_tensor)
      for kernel in kernels:
        gt.watch(kernel)
      if op.use_bias:
        gt.watch(op.bias)
      output = op(input_tensor)
      output = apply_some_non_linearity(output)

    input_grad = gt.gradient(output, input_tensor) if self.require_input_grad else None
    kernel_grad = [gt.gradient(output, kernel) for kernel in kernels]
    # stack kernel gradient tensors into a single tensor if there is more than one
    kernel_grad = tf.convert_to_tensor(kernel_grad) if len(kernel_grad) == 1 else tf.stack(kernel_grad, axis=0)
    bias_grad = None
    if op.use_bias:
      bias_grad = gt.gradient(output, op.bias)

    return output, input_grad, kernel_grad, bias_grad

  def run_fp16_test_instance(self, test_shape, **kwargs):
    """ Runs a half-precision floating-point forward and backward pass test
    """
    tf.keras.backend.set_floatx('float16')
    try:
      self.run_test_instance(test_shape, dtype=tf.float16, **kwargs)
    finally:
      tf.keras.backend.set_floatx('float32')


class Conv2DTestBase(TestBase):
  def setup(self, clifford_product, ref_op_class, test_op_class, kernel_initializer, underlying_dtype=tf.int32):
    """ Type-agnostic convolution forward and backward pass test base
    :clifford_product:    Clifford product specification defining the algebra being tested
    :ref_op_class:        reference scalar operation class, e.g. tf.keras.layers.Conv2D
    :test_op_class:       tested operation class implementing the convolution for the algebra being tested
    :kernel_initializer:  argument name to pass to the layer, 'kernel_initializer' or 'depthwise_initializer'
                          depending on which convolution operation is used
    :underlying_dtype:    underlying tensor type used to generate values
    """
    super().setup(clifford_product, test_op_class, ref_op_class, kernel_initializer, True, underlying_dtype)
    self.test_data_format = 'channels_first'
    self.ref_data_format = 'channels_last'

  def get_ref_compliant_kernels(self, test_op):
    """ Prepare kernels initialized by the tested operation so that they can be used with the reference operation
    :test_op:             tested operation, which was run already
    """
    return tf.transpose(super().get_ref_compliant_kernels(test_op), utils.permutation("nOIHW", "nHWIO"))

  def transpose_ref_to_channel_first(self, ref_output, ref_input_grad, ref_kernel_grad):
    """ Transposes reference tensors to channel first
    :ref_output:          reference output tensor
    :ref_input_grad:      reference tensor with gradient wrt input
    :ref_kernel_grad:      reference tensor with gradient wrt kernel
    """
    ref_output = transpose_to_channel_first(ref_output)
    ref_kernel_grad = tf.transpose(ref_kernel_grad, utils.permutation("nHWIO", "nOIHW"))
    if self.require_input_grad:
      ref_input_grad = transpose_to_channel_first(ref_input_grad)
    return ref_output, ref_input_grad, ref_kernel_grad

  def run_test_instance(self, test_shape, dtype=tf.float32, **kwargs):
    """ Runs a single instance of a forward and backward pass test for a given parameter set.
    The reference operation is run in channel-last format for compatibility with CPU backend.
    :test_shape:          test input shape in channel-last format (NHWC)
    :dtype:               scalar data type of the test batch
    """
    # verify the parameters passed in kwargs
    self.verify_kwargs(**kwargs)
    kwargs.setdefault('use_bias', False)

    # generate random channel-last input
    input_tensor = self.create_input_tensor(test_shape, dtype)
    input_tensor_channel_first = transpose_to_channel_first(input_tensor)

    test_op = self.prepare_test_op(input_tensor_channel_first, **kwargs)
    test_output, test_input_grad, test_kernel_grad, test_bias_grad = self.compute_output_and_gradients(test_op, input_tensor_channel_first, True)

    ref_op = self.prepare_ref_op(input_tensor, test_op, **kwargs)
    ref_output, ref_input_grad, ref_kernel_grad, ref_bias_grad = self.compute_output_and_gradients(ref_op, input_tensor, False)

    # do transpositions to channel-first to compare with test tensors
    ref_output, ref_input_grad, ref_kernel_grad = self.transpose_ref_to_channel_first(ref_output, ref_input_grad, ref_kernel_grad)

    # assert on the difference between reference and test tensors
    self.assert_equal_tensors(ref_output, test_output)
    self.assert_equal_tensors(ref_kernel_grad, test_kernel_grad)

    if self.require_input_grad:
      self.assert_equal_tensors(ref_input_grad, test_input_grad)

    if test_op.use_bias:
      # assert on the difference between reference and test bias tensors
      self.assert_equal_tensors(ref_bias_grad, test_bias_grad)

  def run_conv2d_real_valued_input_test_instance(self, test_shape, dtype=tf.float32, **kwargs):
    """ Runs a single instance of a forward and backward pass test for a given parameter set.
    The reference operation is run in channel-last format for compatibility with CPU backend.
    The test operation is run in channel-first.
    :test_shape:          test input shape in channel-last format (NHWC)
    :dtype:               scalar data type of the test batch
    """
    # verify the parameters passed in kwargs
    self.verify_kwargs(**kwargs)
    assert kwargs.get('use_bias', False) == False, 'use_bias is not supported in tests with a real-valued input'

    # set the parameters specific to convolutions with real valued input
    kwargs['use_bias'] = False
    self.require_input_grad = False
    self.receives_type0_inputs = True

    # generate channel-last random input
    input_tensor = random_integer_tensor(test_shape, dtype=dtype)
    input_tensor_channel_first = transpose_to_channel_first(input_tensor)
    # generate extended input for the reference operation
    input_tensor_with_zero_imag = tf.concat([
        input_tensor,
        tf.zeros((test_shape[0] * (self.clifford_product.dim - 1),) + test_shape[1:], dtype=dtype),
    ], axis=0)

    test_op = self.prepare_test_op(input_tensor_channel_first, **kwargs)
    test_output, _, test_kernel_grad, _ = self.compute_output_and_gradients(test_op, input_tensor_channel_first, True)

    ref_op = self.prepare_ref_op(input_tensor_with_zero_imag, test_op, **kwargs)
    ref_output, _, ref_kernel_grad, _ = self.compute_output_and_gradients(ref_op, input_tensor_with_zero_imag, False)

    # do transpositions to channel-first to compare with test tensors
    ref_output, _, ref_kernel_grad = self.transpose_ref_to_channel_first(ref_output, _, ref_kernel_grad)

    # assert on the difference between reference and test tensors
    self.assert_equal_tensors(ref_output, test_output)
    self.assert_equal_tensors(ref_kernel_grad, test_kernel_grad)


class DenseTestBase(TestBase):
  def setup(self, clifford_product, test_op_class, underlying_dtype=tf.int32):
    """ Type-agnostic Dense forward and backward pass test base
    :clifford_product:    Clifford product specification defining the algebra being tested
    :test_op_class:       test Dense operation class implementing the convolution for the algebra being tested
    :underlying_dtype:    underlying tensor type used to generate values
    """
    super().setup(clifford_product, test_op_class, tf.keras.layers.Dense, 'kernel_initializer', False, underlying_dtype)

  def run_test_instance(self, test_shape, dtype=tf.float32, **kwargs):
    """ Runs a single instance of a forward and backward pass test for a given parameter set.
    :test_shape:          test input shape of rank 2 (in NC format)
    :dtype:               scalar data type of the test batch
    """
    # verify the parameters passed in kwargs
    self.verify_kwargs(**kwargs)
    kwargs.setdefault('use_bias', False)

    # generate random input
    input_tensor = self.create_input_tensor(test_shape, dtype)

    test_op = self.prepare_test_op(input_tensor, **kwargs)
    test_output, test_input_grad, test_kernel_grad, test_bias_grad = self.compute_output_and_gradients(test_op, input_tensor, True)

    ref_op = self.prepare_ref_op(input_tensor, test_op, **kwargs)
    ref_output, ref_input_grad, ref_kernel_grad, ref_bias_grad = self.compute_output_and_gradients(ref_op, input_tensor, False)

    # assert on the difference between reference and test tensors
    self.assert_equal_tensors(ref_output, test_output)
    self.assert_equal_tensors(ref_input_grad, test_input_grad)
    self.assert_equal_tensors(ref_kernel_grad, test_kernel_grad)

    if test_op.use_bias:
      # assert on the difference between reference and test bias tensors
      self.assert_equal_tensors(ref_bias_grad, test_bias_grad)


class Conv2DTestSet(Conv2DTestBase):
  """ Test set for regular Conv2D operation
  """
  def setup(self, clifford_product, test_op_class):
    super().setup(clifford_product, tf.keras.layers.Conv2D, test_op_class, 'kernel_initializer')

  def test_basic(self):
    """ Basic Conv2D test """
    self.run_test_instance(test_shape=(1, 5, 5, 8), filters=32, kernel_size=3)

  def test_bigger_batch(self):
    """ Conv2D with bigger batch test """
    self.run_test_instance(test_shape=(5, 3, 3, 8), filters=32, kernel_size=3)

  def test_pointwise(self):
    """ Pointwise Conv2D test """
    self.run_test_instance(test_shape=(1, 5, 5, 8), filters=32, kernel_size=1)

  def test_strided(self):
    """ Strided Conv2D test """
    self.run_test_instance(test_shape=(1, 5, 5, 64), filters=16, strides=(2, 3), kernel_size=2)

  def test_dilated(self):
    """ Dilated Conv2D test """
    self.run_test_instance(test_shape=(1, 7, 7, 16), filters=16, dilation_rate=(2, 3), kernel_size=3)

  def test_non_square(self):
    """ Non-square image Conv2D test """
    self.run_test_instance(test_shape=(1, 15, 3, 64), filters=16, kernel_size=2)

  def test_bias(self):
    """ Biased Conv2D test """
    self.run_test_instance(test_shape=(1, 3, 3, 64), filters=32, kernel_size=1, use_bias=True)

  def test_padded(self):
    """ Padded Conv2D test """
    self.run_test_instance(test_shape=(1, 5, 5, 32), filters=32, kernel_size=3, padding='same')

  def test_padded_strided(self):
    """ Padded strided Conv2D test """
    self.run_test_instance(test_shape=(1, 7, 7, 16), filters=32, kernel_size=3, strides=2, padding='same')

  def test_padded_dilated(self):
    """ Padded dilated Conv2D test """
    self.run_test_instance(test_shape=(2, 7, 7, 16), filters=32, kernel_size=3, dilation_rate=(2, 2), padding='same')

  @unittest.skipIf(not gpu_visible(), "grouped conv not supported on CPU")
  @unittest.skipIf(tf.version.VERSION < '2.3.0', "tensorflow version needs to be at least 2.3.0")
  def test_grouped(self):
    """ Group Conv2D test """
    self.run_test_instance(test_shape=(1, 5, 5, 64), filters=48, groups=4, kernel_size=3)

  def test_real_valued_input(self):
    """ Conv2D with real-valued input test """
    self.run_conv2d_real_valued_input_test_instance(test_shape=(2, 3, 4, 8), filters=16, kernel_size=3, strides=2, padding='same')

  @unittest.skipIf(not gpu_visible(), "fp16 not supported on CPU")
  def test_fp16(self):
    """ Half-precision floating point Conv2D test """
    self.run_fp16_test_instance(test_shape=(2, 5, 5, 8), filters=16, kernel_size=3, use_bias=True)


class PointwiseConv2DTestSet(Conv2DTestBase):
  """ Test set for pointwise Conv2D operation
  """
  def setup(self, clifford_product, test_op_class):
    super().setup(clifford_product, tf.keras.layers.Conv2D, test_op_class, 'kernel_initializer')

  def test_minimal(self):
    """ Minimal PointwiseConv2D test """
    self.run_test_instance(test_shape=(1, 1, 1, 1), filters=1, kernel_size=1)

  def test_basic(self):
    """ Basic PointwiseConv2D test """
    self.run_test_instance(test_shape=(1, 3, 3, 64), filters=32, kernel_size=1)

  def test_larger_batch(self):
    """ PointwiseConv2D with larger batch test """
    self.run_test_instance(test_shape=(5, 3, 3, 32), filters=32, kernel_size=1)

  def test_batch_largest(self):
    """ PointwiseConv2D with batch size as the largest parameter test """
    self.run_test_instance(test_shape=(11, 3, 3, 8), filters=8, kernel_size=1)

  def test_non_square_high(self):
    """ Non-square high image PointwiseConv2D test """
    self.run_test_instance(test_shape=(1, 14, 3, 32), filters=16, kernel_size=1)

  def test_non_square_wide(self):
    """ Non-square wide image PointwiseConv2D test """
    self.run_test_instance(test_shape=(1, 3, 12, 32), filters=16, kernel_size=1)

  def test_irregular_params(self):
    """ Irregular size parameters PointwiseConv2D test """
    self.run_test_instance(test_shape=(7, 11, 5, 3), filters=13, kernel_size=1)

  def test_bias(self):
    """ Biased PointwiseConv2D test """
    self.run_test_instance(test_shape=(1, 3, 3, 64), filters=32, kernel_size=1, use_bias=True)

  def test_bias_larger_batch(self):
    """ Biased PointwiseConv2D with larger batch test """
    self.run_test_instance(test_shape=(5, 3, 3, 16), filters=32, kernel_size=1, use_bias=True)

  def test_few_filters(self):
    """ Few filters PointwiseConv2D test """
    self.run_test_instance(test_shape=(4, 7, 7, 64), filters=3, kernel_size=1)

  def test_few_input_channels(self):
    """ Few input channels PointwiseConv2D test """
    self.run_test_instance(test_shape=(4, 7, 7, 3), filters=64, kernel_size=1)

  def test_few_channels(self):
    """ Few channels PointwiseConv2D test """
    self.run_test_instance(test_shape=(6, 7, 7, 3), filters=3, kernel_size=1)

  def test_even_height(self):
    """ Image with even height PointwiseConv2D test """
    self.run_test_instance(test_shape=(3, 4, 3, 16), filters=16, kernel_size=1)

  def test_even_width(self):
    """ Image with even width PointwiseConv2D test """
    self.run_test_instance(test_shape=(3, 3, 4, 16), filters=16, kernel_size=1)

  def test_even_image(self):
    """ Image with even width and height PointwiseConv2D test """
    self.run_test_instance(test_shape=(3, 4, 4, 16), filters=16, kernel_size=1)

  @unittest.skipIf(not gpu_visible(), "fp16 not supported on CPU")
  def test_fp16(self):
    """ Half-precision floating point PointwiseConv2D test """
    self.run_fp16_test_instance(test_shape=(3, 5, 5, 8), filters=16, kernel_size=1)

  @unittest.skipIf(not gpu_visible(), "fp16 not supported on CPU")
  def test_fp16_bias(self):
    """ Half-precision floating point biased PointwiseConv2D test """
    self.run_fp16_test_instance(test_shape=(3, 4, 4, 8), filters=16, kernel_size=1, use_bias=True)


class DepthwiseConv2DTestSet(Conv2DTestBase):
  """ Test set for depthwise Conv2D operation
  """
  def setup(self, clifford_product, test_op_class):
    super().setup(clifford_product, tf.keras.layers.DepthwiseConv2D, test_op_class, 'depthwise_initializer')

  def get_ref_compliant_kernels(self, test_op):
    """ Prepare kernels initialized by the tested operation so that they can be used with the reference operation
    :test_op:             tested operation, which was run already
    """
    return tf.transpose(TestBase.get_ref_compliant_kernels(self, test_op), utils.permutation("nOIHW", "nHWOI"))

  def test_basic(self):
    """ Basic DepthwiseConv2D test """
    self.run_test_instance(test_shape=(1, 5, 5, 32), kernel_size=3)

  def test_bigger_batch(self):
    """ DepthwiseConv2D with bigger batch test """
    self.run_test_instance(test_shape=(5, 3, 3, 16), kernel_size=3)

  def test_strided(self):
    """ Strided DepthwiseConv2D test """
    self.run_test_instance(test_shape=(1, 11, 11, 32), strides=3, kernel_size=4)

  def test_dilated(self):
    """ Dilated DepthwiseConv2D test """
    self.run_test_instance(test_shape=(1, 7, 7, 16), dilation_rate=(2, 3), kernel_size=3)

  def test_non_square(self):
    """ Non-square image DepthwiseConv2D test """
    self.run_test_instance(test_shape=(1, 15, 3, 64), kernel_size=2)

  def test_bias(self):
    """ Biased DepthwiseConv2D test """
    self.run_test_instance(test_shape=(1, 3, 3, 64), kernel_size=1, use_bias=True)

  def test_padded(self):
    """ Padded DepthwiseConv2D test """
    self.run_test_instance(test_shape=(1, 5, 5, 32), kernel_size=3, padding='same')

  def test_padded_strided(self):
    """ Padded strided DepthwiseConv2D test """
    self.run_test_instance(test_shape=(1, 7, 7, 16), kernel_size=3, strides=2, padding='same')

  def test_padded_dilated(self):
    """ Padded dilated DepthwiseConv2D test """
    self.run_test_instance(test_shape=(2, 7, 7, 16), kernel_size=3, dilation_rate=(2, 2), padding='same')

  @unittest.skipIf(not gpu_visible(), "fp16 not supported on CPU")
  def test_fp16(self):
    """ Half-precision floating point DepthwiseConv2D test """
    self.run_fp16_test_instance(test_shape=(2, 3, 3, 4), kernel_size=2, use_bias=True)


class DenseTestSet(DenseTestBase):
  """ Test set for Dense operation
  """
  def setup(self, clifford_product, test_op_class):
    super().setup(clifford_product, test_op_class)

  def test_basic(self):
    """ Basic Dense test """
    self.run_test_instance(test_shape=(1, 8), units=4)

  def test_bigger_batch(self):
    """ Bigger batch Dense test """
    self.run_test_instance(test_shape=(2, 4), units=8)

  def test_biased(self):
    """ Biased Dense test """
    self.run_test_instance(test_shape=(1, 12), units=8, use_bias=True)

  @unittest.skipIf(not gpu_visible(), "fp16 not supported on CPU")
  def test_fp16(self):
    """ Half-precision floating point Dense test """
    self.run_fp16_test_instance(test_shape=(2, 4), units=8, use_bias=True)


class InputGradientAndTypeTest:
  """ Tests "require_input_grad" and real-valued TF2Upstride output flags settings
  """
  def setup(self, layers):
    self.layers = layers

  def test(self):
    # build a model: 2 blocks of (conv2d + batchnorm + ReLU) and a dense layer
    model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(3, 32, 32)),
      self.layers.TF2Upstride(),
      self.layers.Conv2D(32, kernel_size=3, strides=2, use_bias=False, data_format='channels_first', name='conv1'),
      tf.keras.layers.BatchNormalization(),   #FIXME: Sequential does not accept layers that do not subclass tf.keras.Layer
      tf.keras.layers.Activation('relu'),
      self.layers.Conv2D(16, kernel_size=3, strides=2, use_bias=False, data_format='channels_first', name='conv2'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.Flatten(),
      self.layers.Dense(units=10, name='dense'),
      self.layers.Upstride2TF()
    ])

    conv1 = model.get_layer('conv1')
    conv2 = model.get_layer('conv2')
    dense = model.get_layer('dense')

    # assert on the input gradient requiredness
    assert conv1.require_input_grad == False,  "Input gradient for the first convolution layer should not be required"
    assert conv2.require_input_grad == True,   "Input gradient for the first convolution layer should be required"
    assert dense.require_input_grad == True,   "Input gradient for the dense layer should be required"

    # make a pass through the model to initialize type0-related flags
    model.compile(optimizer='sgd', loss='mse')
    model.predict(tf.zeros((1, 3, 32, 32)))

    # assert on type0-related flags
    if conv1.upstride_datatype != TYPE0:
      assert conv1.receives_type0_inputs == True,  "The first convolution layer should receive type0 input"
    assert conv2.receives_type0_inputs == False,   "The second convolution layer should not receive type0 input"
    assert dense.accepts_type0_inputs == False,    "The dense layer should not accept type0 input"