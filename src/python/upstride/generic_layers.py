"""users shouldn't import this package directly. instead import upstride.typeX.tf.keras.layers
"""
import functools
import inspect
from typing import List, Tuple
import tensorflow as tf


# Definition of the GA, setup when upstride.type{1/2/3}.calling tf.keras.layers
upstride_type = 3
blade_indexes = ["", "1", "2", "3", "12", "13", "23", "123"]
geometrical_def = (3, 0, 0)


def change_upstride_type(new_upstride_type, new_blade_indexes,  new_geometrical_def):
  global upstride_type, blade_indexes, geometrical_def
  upstride_type = new_upstride_type
  blade_indexes = new_blade_indexes
  geometrical_def = new_geometrical_def


def multivector_length() -> int:
  """map the upstride type to the number of dimensions in our GA
  """
  return len(blade_indexes)


def blade_index_to_position(index: str) -> int:
  @functools.lru_cache(maxsize=1)
  def get_dict():
    """return a dictionary that map the blade index to the position in the list encoding the multivector
    """
    d = {}
    for i, e in enumerate(blade_indexes):
      d[e] = i
    return d
  return get_dict()[index]


def square_vector(index: int) -> int:
  @functools.lru_cache(maxsize=1)
  def get_list():
    """return a list that map the indice to the sqare
    """
    l = [0]
    possible_squares = [1, -1, 0]
    for i in range(3):
      l += [possible_squares[i]] * geometrical_def[i]
    return l
  return get_list()[index]


def _ga_multiply_get_index(index_1: str, index_2: str) -> Tuple[int, str]:
  """given e_{index_1}, e_{index_2} return (s, index) such as e_{index_1} * e_{index_2} = s * e_{index}
  """
  l1 = [int(i) for i in index_1]
  l2 = [int(i) for i in index_2]
  s = 1

  # as l1 and l2 are already sorted, we can just merge them and count the number of permutation needed
  i1, i2, length_l1 = 0, 0, len(l1)
  out_l = []
  while i1 < len(l1) and i2 < len(l2):
    if l1[i1] == l2[i2]:
      # then move the element of l2 near the element of l1 and remove them
      if (length_l1 - 1) % 2 != 0:
        s *= -1
      # check the sign of the square
      s *= square_vector(l1[i1])
      length_l1 -= 1
      i1 += 1
      i2 += 1
    elif l1[i1] > l2[i2]:
      # then move the element of l2 before the element of l1
      if length_l1 % 2 != 0:
        s *= -1
      out_l.append(l2[i2])
      i2 += 1
    elif l1[i1] < l2[i2]:
      out_l.append(l1[i1])
      length_l1 -= 1
      i1 += 1
  out_l += l1[i1:] + l2[i2:]

  return s, "".join([str(i) for i in out_l])


def unit_multiplier(i: int, j: int) -> Tuple[int, int]:
  """given e_i and e_j, return (k,s) such as : e_i * e_j = s * e_k

  with:
      e_0 = 1, e_1 = i if upstride_type == 1
      e_0 = 1, e_1 = i, e_2 = j, e_3 = k if upstride_type == 2
      s in {-1, 1}

  for instance, upstride_type == 1,
  (0, 0) -> (0, 1) because e_0 * e_0 = 1 * 1 = 1 * e_0
  (0, 1) -> (1, 1) because e_0 * e_1 = 1 * e_1
  (1, 1) -> (0, -1) because e_1 * e_1 = i**2 = -1 = -1 * e_0
  """
  index1 = blade_indexes[i]
  index2 = blade_indexes[j]
  s, index = _ga_multiply_get_index(index1, index2)
  return blade_index_to_position(index), s


def get_layers(layer: tf.keras.layers.Layer, *argv, **kwargs) -> Tuple[List[tf.keras.layers.Layer], bool, dict]:
  """instantiate layer several times to match the number needed by the GA definition

  Any parameter analysis need to be done here. For instance, we can't define several times 
  a layer with the same name, so we need to edit the name manually

  Args:
      layer (tf.keras.layers.Layer): a keras layer that we need to instantiate several times

  Returns:
      List[tf.keras.layers.Layer]: the list of keras layers
  """
  # convert all arguments from argv to kwargs
  parameters = inspect.getfullargspec(layer.__init__).args
  for i, arg in enumerate(argv):
    kwargs[parameters[i + 1]] = arg  # + 1 because the first element of parameters is 'self'
  # add all default parameters to kwargs
  for key, value in inspect.signature(layer.__init__).parameters.items():
    if key in ['self', 'kwargs']:
      continue
    if key not in kwargs:
      kwargs[key] = value.default

  # If we define some bias, we don't want to put it in the linear layer but after, as a non-linear layer
  add_bias = False
  if "use_bias" in kwargs:
    add_bias = kwargs["use_bias"]
    kwargs["use_bias"] = False
  bias_parameters = {}
  if add_bias:
    for param in ["bias_initializer", "bias_regularizer", "bias_constraint"]:
      bias_parameters[param] = kwargs[param]

  # special case for the name of the layer : if defined, then we need to change it to create different operations
  if 'name' not in kwargs or kwargs['name'] is None:
    layers = [layer(**kwargs) for _ in range(multivector_length())]
  else:
    layers = []
    base_name = kwargs['name']
    for i in range(multivector_length()):
      kwargs['name'] = f'{base_name}_{i}'
      layers.append(layer(**kwargs))

  return layers, add_bias, bias_parameters


def compute_all_cross_product(layers, inputs):
  layers_outputs = []
  for i in range(multivector_length()):
    layers_outputs.append([])
    for j in range(multivector_length()):
      layers_outputs[i].append(layers[i](inputs[j]))
  return layers_outputs


def geometric_multiplication(cross_product_matrix, inverse=False):
  output = [None] * multivector_length()
  for i in range(multivector_length()):
    for j in range(multivector_length()):
      if not inverse:
        k, s = unit_multiplier(i, j)
      else:
        k, s = unit_multiplier(j, i)

      # same as output[k] += s*self.layers[i](inputs[j]), but cleaner graph
      if s == 1:
        if output[k] is None:
          output[k] = cross_product_matrix[i][j]
        else:
          output[k] += cross_product_matrix[i][j]
      elif s == -1:
        if output[k] is None:
          output[k] = -cross_product_matrix[i][j]
        else:
          output[k] -= cross_product_matrix[i][j]
  return output


class BiasLayer(tf.keras.layers.Layer):
  """Keras layer that only adds a bias to the input.

  code from https://github.com/tensorflow/agents/blob/v0.4.0/tf_agents/networks/bias_layer.py#L24-L81
  with some modifications when initializing the weight to use the same conf as other layers

  `BiasLayer` implements the operation:
  `output = input + bias`
  Arguments:
      bias_initializer: Initializer for the bias vector.
  Input shape:
      nD tensor with shape: `(batch_size, ..., input_dim)`. The most common
        situation would be a 2D input with shape `(batch_size, input_dim)`. Note
        a rank of at least 2 is required.
  Output shape:
      nD tensor with shape: `(batch_size, ..., input_dim)`. For instance, for a
        2D input with shape `(batch_size, input_dim)`, the output would have
        shape `(batch_size, input_dim)`.
  """

  def __init__(self, bias_initializer='zeros', bias_regularizer=None, bias_constraint=None, **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(BiasLayer, self).__init__(**kwargs)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])

    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `BiasLayer` '
                       'should be defined. Found `None`.')

    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})
    self.bias = self.add_weight(
        name='bias',
        shape=[input_shape[-1]],
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True,
        dtype=self.dtype)

    self.built = True

  def call(self, inputs):
    return tf.nn.bias_add(inputs, self.bias)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'bias_initializer':
            tf.keras.initializers.serialize(self.bias_initializer),
    }
    base_config = super(BiasLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GenericLinear:
  """
  this operation will perform linear operation for every GA. 
  Please note that this operation is not very efficient (need to split the tensor, do the computation then concat the results)
  """

  def __init__(self, layer, *argv, **kwargs):
    self.layers, self.add_bias, self.bias_parameters = get_layers(layer, *argv, **kwargs)

  def __call__(self, inputs):
    # split the input tensor into a list containing [real, complex1, ....]
    inputs = tf.split(inputs, len(blade_indexes))

    # R^{multivector_length()} input
    layers_outputs = compute_all_cross_product(self.layers, inputs)
    outputs = geometric_multiplication(layers_outputs)

    if self.add_bias:
      for i in range(multivector_length()):
        outputs[i] = BiasLayer(self.bias_parameters['bias_initializer'], self.bias_parameters['bias_regularizer'], self.bias_parameters['bias_constraint'])(outputs[i])
    outputs = tf.concat(outputs, 0)
    return outputs


class GenericNonLinear:
  """
  with the current mathematical formulation of GoM, non linear layers become easy to do : 
  for most of them, we simply need to call the tensorflow function. As real part and imaginary parts are stacked
  on first component of the tensor (usually the batch size), it is transparent for tensorflow
  """

  def __init__(self, layer, *argv, **kwargs):
    self.layers = layer(*argv, **kwargs)

  def __call__(self, inputs):
    return self.layers(inputs)


class Conv2D(GenericLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Conv2D, *argv, **kwargs)


class Dense(GenericLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Dense, *argv, **kwargs)


class Conv2DTranspose(GenericLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Conv2DTranspose, *argv, **kwargs)


class UpSampling2D(GenericLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.UpSampling2D, *argv, **kwargs)


class DepthwiseConv2D(GenericLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.DepthwiseConv2D, *argv, **kwargs)


class DepthwiseConv2DTranspose(GenericLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.DepthwiseConv2DTranspose, *argv, **kwargs)


class SeparableConv2D(GenericLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.SeparableConv2D, *argv, **kwargs)


class MaxPooling2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.MaxPooling2D, *argv, **kwargs)


class AveragePooling2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.AveragePooling2D, *argv, **kwargs)


class MaxPool2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.MaxPool2D, *argv, **kwargs)


class AveragePool2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.AveragePool2D, *argv, **kwargs)


class GlobalMaxPooling2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.GlobalMaxPooling2D, *argv, **kwargs)


class GlobalAveragePooling2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.GlobalAveragePooling2D, *argv, **kwargs)


class Reshape(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Reshape, *argv, **kwargs)


class Activation(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Activation, *argv, **kwargs)


class Flatten(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Flatten, *argv, **kwargs)


class ZeroPadding2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.ZeroPadding2D, *argv, **kwargs)


class Cropping2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Cropping2D, *argv, **kwargs)


class ReLU(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.ReLU, *argv, **kwargs)


class LeakyReLU(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.LeakyReLU, *argv, **kwargs)


class Add(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Add, *argv, **kwargs)
    self.list_as_input = True


class Multiply(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Multiply, *argv, **kwargs)
    self.list_as_input = True


class Concatenate(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Concatenate, *argv, **kwargs)
    self.list_as_input = True


# with the formulation of saving the blades of the multivector in the Batch dim, we need to handle
# Dropout and Batch normalization differently than other layers

class SplittedNonLinear:
  def __init__(self, layer, *argv, **kwargs):
    self.layers = [layer(*argv, **kwargs) for _ in range(len(blade_indexes))]

  def __call__(self, inputs):
    inputs = tf.split(inputs, len(blade_indexes))
    outputs = [self.layers[i](inputs[i]) for i in range(len(blade_indexes))]
    outputs = tf.concat(outputs, 0)
    return outputs


class Dropout(SplittedNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Dropout, *argv, **kwargs)


class BatchNormalization(SplittedNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.BatchNormalization, *argv, **kwargs)
