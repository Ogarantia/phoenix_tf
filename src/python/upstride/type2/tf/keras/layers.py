from typing import Dict

from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import load_library
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import tf_utils
from upstride.generic_convolution import GenericConv2D
from upstride.type_generic.tf.keras.layers import TYPE2
from .... import generic_layers
from ....generic_layers import *
from .dense import Dense

generic_layers.upstride_type = 2
generic_layers.blade_indexes = ["", "12", "23", "13"]
generic_layers.geometrical_def = (3, 0, 0)

# If you wish to overwrite some layers, please implements them here


class TF2Upstride(Layer):
  """assume this function is called at the begining of the network. 
  Select between several strategies, like putting colors to imaginary parts and grayscale in real, ...
  """

  @staticmethod
  def learn_vector_component(x, channels=3):
    """
    Learning module taken from this paper (https://arxiv.org/pdf/1712.04604.pdf)
    BN --> ReLU --> Conv --> BN --> ReLU --> Conv

    Args:
        x: input x
        channels: number of channels
    Returns:
        leaned  multi - vector (could have multiple channels)
    """
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(channels, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(channels, (3, 3), padding='same')(x)
    return x

  @staticmethod
  def rgb_in_img(x: tf.Tensor):

    red = tf.expand_dims(x[:, :, :, 0], -1)
    green = tf.expand_dims(x[:, :, :, 1], -1)
    blue = tf.expand_dims(x[:, :, :, 2], -1)
    zeros = tf.zeros_like(red)
    return tf.concat([zeros, red, green, blue], axis=0)

  @staticmethod
  def gray_in_real_rgb_in_img(x: tf.Tensor):
    red = tf.expand_dims(x[:, :, :, 0], -1)
    green = tf.expand_dims(x[:, :, :, 1], -1)
    blue = tf.expand_dims(x[:, :, :, 2], -1)
    grayscale = tf.image.rgb_to_grayscale(x)
    return tf.concat([grayscale, red, green, blue], axis=0)

  @staticmethod
  def learn_multivector(x: tf.Tensor):
    r = TF2Upstride.learn_vector_component(x, 3)
    i = TF2Upstride.learn_vector_component(x, 3)
    j = TF2Upstride.learn_vector_component(x, 3)
    k = TF2Upstride.learn_vector_component(x, 3)
    return tf.concat([r, i, j, k], axis=0)

  @staticmethod
  def default_convert(x: tf.Tensor):
    zeros = tf.zeros_like(x)
    return tf.concat([x, zeros, zeros, zeros], axis=0)

  def __init__(self, strategy=''):
    STRATEGIES = {
        'joint': TF2Upstride.rgb_in_img,
        'grayscale': TF2Upstride.gray_in_real_rgb_in_img,
        'learned': TF2Upstride.learn_multivector,
        '': TF2Upstride.default_convert,
    }

    try:
      self.strategy = STRATEGIES[strategy]
    except KeyError:
      raise ValueError(f"unknown strategy: {strategy}")

  def __call__(self, x):
    return self.strategy(x)


class Upstride2TF(Layer):
  """convert multivector back to real values.
  """

  def __init__(self, strategy='default'):
    self.concat = True if strategy == "concat" else False

  def __call__(self, x):
    components = tf.split(x, 4)
    if self.concat:
      return tf.concat(components, -1)
    else:
      return components[0]


class Conv2D(GenericConv2D):
  def __init__(self, filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               require_input_grad=True,
               **kwargs):
    super().__init__(filters,
                     kernel_size,
                     strides,
                     padding,
                     data_format,
                     dilation_rate,
                     groups,
                     activation,
                     use_bias,
                     kernel_initializer,
                     bias_initializer,
                     kernel_regularizer,
                     bias_regularizer,
                     activity_regularizer,
                     kernel_constraint,
                     bias_constraint,
                     require_input_grad
                     **kwargs)
    self.upstride_datatype = TYPE2


class DepthwiseConv2D(Conv2D):
  def __init__(self,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    super().__init__(
        filters=None,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = initializers.get(depthwise_initializer)
    self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
    self.depthwise_constraint = constraints.get(depthwise_constraint)
    self.bias_initializer = initializers.get(bias_initializer)

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                       'Received input shape:', str(input_shape))
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs to '
                       '`DepthwiseConv2D` '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    # 1 is the size of the group
    # in pure tensorflow, code is like this :
    # depthwise_kernel_shape = (self.kernel_size[0],
    #                           self.kernel_size[1],
    #                           input_dim,
    #                           self.depth_multiplier)
    # in upstride, the order need to be (o, g, h, w)
    depthwise_kernel_shape = (1, # number of output channels per group
                              self.depth_multiplier, # number of groups, so number of channels for depth wise conv 
                              self.kernel_size[0],
                              self.kernel_size[1])

    self.depthwise_kernel = self.add_weight(
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        name='depthwise_kernel',
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint)

    if self.use_bias:
      self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                  initializer=self.bias_initializer,
                                  name='bias',
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs):
    outputs = self.upstride_conv_op(
        inputs,
        self.depthwise_kernel,
        uptype=self.upstride_datatype,
        strides=self.strides,
        padding=self.padding,
        dilation_rate=self.dilation_rate,
        data_format=self.data_format,
        groups=inputs.shape[1])

    if self.use_bias:
      outputs = backend.bias_add(
          outputs,
          self.bias,
          data_format=self.data_format)

    return outputs

class MaxNormPooling2D(Layer):
  """ Max Pooling layer for quaternions which considers the norm of quaternions to choose the quaternions
  which exhibits maximum norm within a small window of pool size.
  Arguments:
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    norm_type: A string or integer. Order of the norm.
      Supported values are 'fro', 'euclidean', 1, 2, np.inf and
      any positive real number yielding the corresponding p-norm.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides=None, padding='valid', norm_type='euclidean',
               data_format=None, name=None, **kwargs):
    super(MaxNormPooling2D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = backend.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.norm_type = norm_type

  def call(self, inputs):
    inputs = [tf.expand_dims(inputs[i], -1) for i in range(len(inputs))]
    inputs = tf.keras.layers.concatenate(inputs, axis=-1)

    if self.data_format == 'channels_last':
      pool_shape = (1,) + self.pool_size + (1,)
      strides = (1,) + self.strides + (1,)
    else:
      pool_shape = (1, 1) + self.pool_size
      strides = (1, 1) + self.strides
    padding = self.padding.upper()

    norm = tf.norm(inputs, ord=self.norm_type, axis=-1)

    _, indices = tf.nn.max_pool_with_argmax(norm, ksize=pool_shape, strides=strides, padding=padding,
                                            include_batch_in_index=True)

    input_shape = tf.shape(norm)
    output_shape = tf.shape(indices)

    flat_input_size = tf.reduce_prod(input_shape)
    flat_indices_size = tf.reduce_prod(output_shape)

    indices = tf.reshape(indices, [flat_indices_size, 1])

    inputs = tf.reshape(inputs, [flat_input_size, 4])

    pooled = [tf.reshape(tf.gather_nd(inputs[:, i], indices), output_shape) for i in range(4)]

    return pooled

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format,
        'norm_type': self.norm_type
    }
    base_config = super(MaxNormPooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def sqrt_init(shape, dtype=None):
  value = (1 / tf.sqrt(4.0)) * tf.ones(shape)
  return value


def quaternion_standardization(input_centred, v: Dict, axis=-1):
  """compute the multiplication of the Cholesky decomposition of the var by the centered input

  Args:
      input_centred (List): list of 4 tensors
      v (Dict): variance. Labels are 'rr', 'ri', 'rj', ....
      axis (int, optional): Defaults to -1.

  Returns:
      List: list of 4 tensors
  """

  # Chokesky decomposition of 4x4 symmetric matrix
  w = {}
  w['rr'] = tf.sqrt(v['rr'])
  w['ri'] = (1.0 / w['rr']) * (v['ri'])
  w['ii'] = tf.sqrt((v['ii'] - (w['ri']*w['ri'])))
  w['rj'] = (1.0 / w['rr']) * (v['rj'])
  w['ij'] = (1.0 / w['ii']) * (v['ij'] - (w['ri']*w['rj']))
  w['jj'] = tf.sqrt((v['jj'] - (w['ij']*w['ij'] + w['rj']*w['rj'])))
  w['rk'] = (1.0 / w['rr']) * (v['rk'])
  w['ik'] = (1.0 / w['ii']) * (v['ik'] - (w['ri']*w['rk']))
  w['jk'] = (1.0 / w['jj']) * (v['jk'] - (w['ij']*w['ik'] + w['rj']*w['rk']))
  w['kk'] = tf.sqrt((v['kk'] - (w['jk']*w['jk'] + w['ik']*w['ik'] + w['rk']*w['rk'])))

  # compute the oposite
  o = {}
  o['rr'] = 1 / w['rr']
  o['ii'] = 1 / w['ii']
  o['jj'] = 1 / w['jj']
  o['kk'] = 1 / w['kk']
  o['ri'] = -(w['ri'] * o['rr']) / w['ii']
  o['rj'] = -(w['rj'] * o['rr'] + w['ij'] * o['ri']) / w['jj']
  o['rk'] = -(w['rk'] * o['rr'] + w['ik'] * o['ri'] + w['jk'] * o['rj'])/w['kk']
  o['ij'] = -(w['ij'] * o['ii']) / w['jj']
  o['ik'] = -(w['ik'] * o['ii'] + w['jk'] * o['ij']) / w['kk']
  o['jk'] = -(w['jk'] * o['jj']) / w['kk']

  w = o

  # Normalization. We multiply, x_normalized = W.x.
  # The returned result will be a quaternion standardized input
  # where the r, i, j, and k parts are obtained as follows:
  # x_r_normed = Wrr * x_r_cent + Wri * x_i_cent + Wrj * x_j_cent + Wrk * x_k_cent
  # x_i_normed = Wri * x_r_cent + Wii * x_i_cent + Wij * x_j_cent + Wik * x_k_cent
  # x_j_normed = Wrj * x_r_cent + Wij * x_i_cent + Wjj * x_j_cent + Wjk * x_k_cent
  # x_k_normed = Wrk * x_r_cent + Wik * x_i_cent + Wjk * x_j_cent + Wkk * x_k_cent
  output = []
  dim_names = "rijk"
  for p1 in range(4):
    tmp_out = 0
    for p2 in range(4):
      i1 = min(p1, p2)
      i2 = max(p1, p2)
      tmp_out += w[dim_names[i1]+dim_names[i2]] * input_centred[p2]
    output.append(tmp_out)
  return output


def quaternion_bn(input_centred, v: Dict, beta, gamma: Dict, axis=-1):
  """take a centred input and compute the BN

  Args:
      input_centred (List): list of 4 tensors
      v (Dict): variance. Labels are 'rr', 'ri', 'rj', ....
      beta (Dict): Labels are 'r' 'i' 'j' 'k'
      gamma (Dict): Labels are 'rr', 'ri', 'rj', ....
      axis (int, optional): Defaults to -1.

  Returns:
      List: list of 4 tensors
  """
  standardized_output = quaternion_standardization(input_centred, v, axis=axis)  # shape (BS, H, W, C) * 4

  # Now we perform the scaling and shifting of the normalized x using
  # the scaling parameter
  #           [  gamma_rr gamma_ri gamma_rj gamma_rk  ]
  #   Gamma = [  gamma_ri gamma_ii gamma_ij gamma_ik  ]
  #           [  gamma_rj gamma_ij gamma_jj gamma_jk  ]
  #           [  gamma_rk gamma_ik gamma_jk gamma_kk  ]
  # and the shifting parameter
  #    Beta = [beta_r beta_i beta_j beta_k].T
  # where:
  # x_r_BN = gamma_rr * x_r + gamma_ri * x_i + gamma_rj * x_j + gamma_rk * x_k + beta_r
  # x_i_BN = gamma_ri * x_r + gamma_ii * x_i + gamma_ij * x_j + gamma_ik * x_k + beta_i
  # x_j_BN = gamma_rj * x_r + gamma_ij * x_i + gamma_jj * x_j + gamma_jk * x_k + beta_j
  # x_k_BN = gamma_rk * x_r + gamma_ik * x_i + gamma_jk * x_j + gamma_kk * x_k + beta_k

  output = []
  broadcast_beta_shape = [1] * len(input_centred[0].shape)
  broadcast_beta_shape[axis] = input_centred[0].shape[axis]  # unittest [1, 1, 1, 5]

  dim_names = "rijk"
  for p1 in range(4):
    tmp_out = 0
    for p2 in range(4):
      i1 = min(p1, p2)
      i2 = max(p1, p2)
      tmp_out += gamma[dim_names[i1]+dim_names[i2]] * standardized_output[p2]
    output.append(tmp_out + tf.reshape(beta[dim_names[p1]], broadcast_beta_shape))

  return output


class BatchNormalizationQ(Layer):
  """
  quaternion implementation : https://github.com/gaudetcj/DeepQuaternionNetworks/blob/43b321e1701287ce9cf9af1eb16457bdd2c85175/quaternion_layers/bn.py
  tf implementation : https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/layers/normalization.py#L46
  paper : https://arxiv.org/pdf/1712.04604.pdf

  this version perform the same operations as the deep quaternion network version, but has been rewritten to be cleaner
  """

  def __init__(self, axis=-1, momentum=0.9, epsilon=1e-4, center=True, scale=True, beta_initializer='zeros',
               gamma_diag_initializer='sqrt_init',
               gamma_off_initializer='zeros',
               moving_mean_initializer='zeros',
               moving_variance_initializer='sqrt_init',
               moving_covariance_initializer='zeros',
               beta_regularizer=None,
               gamma_diag_regularizer=None,
               gamma_off_regularizer=None,
               beta_constraint=None,
               gamma_diag_constraint=None,
               gamma_off_constraint=None,
               **kwargs):
    """
    Args:
        axis: Integer, the axis that should be normalized (typically the features axis). For instance, after a `Conv2D` layer with
            `data_format="channels_first"`, set `axis=2` in `QuaternionBatchNormalization`.
    """
    super().__init__(**kwargs)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center  # if true then use beta and gamma to add a bit of variance
    self.scale = scale

    # gamma parameter, diagonale (trainable)
    if gamma_diag_initializer != 'sqrt_init':
      self.gamma_diag_initializer = tf.keras.initializers.get(gamma_diag_initializer)
    else:
      self.gamma_diag_initializer = sqrt_init
    self.gamma_diag_regularizer = tf.keras.regularizers.get(gamma_diag_regularizer)
    self.gamma_diag_constraint = tf.keras.constraints.get(gamma_diag_constraint)

    # gamma parameter, outside of diagonale (trainable)
    self.gamma_off_initializer = tf.keras.initializers.get(gamma_off_initializer)
    self.gamma_off_regularizer = tf.keras.regularizers.get(gamma_off_regularizer)
    self.gamma_off_constraint = tf.keras.constraints.get(gamma_off_constraint)

    # beta parameter (trainable)
    self.beta_initializer = tf.keras.initializers.get(beta_initializer)
    self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
    self.beta_constraint = tf.keras.constraints.get(beta_constraint)

    # moving_V parameter (not trainable)
    if moving_variance_initializer != 'sqrt_init':
      self.moving_variance_initializer = tf.keras.initializers.get(moving_variance_initializer)
    else:
      self.moving_variance_initializer = sqrt_init

    # moving covariance
    self.moving_covariance_initializer = tf.keras.initializers.get(moving_covariance_initializer)

    # moving_mean (not trainable)
    self.moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)

  def build(self, input_shape):
    ndim = len(input_shape[0])  # usually 4
    # update self.axis
    if self.axis < 0:
      self.axis = ndim + self.axis  # usually 3
    param_shape = input_shape[0][self.axis]  # 5 for unit-tests
    if param_shape is None:
      raise ValueError(f'Axis {self.axis} of input tensor should have a defined dimension '
                       f'but the layer received an input with shape {input_shape}.')
    self.gamma = {}
    self.moving_V = {}
    dim_names = "rijk"
    for p1 in range(4):
      for p2 in range(p1, 4):
        postfix = dim_names[p1]+dim_names[p2]
        self.gamma[postfix] = self.add_weight(shape=param_shape,
                                              name=f'gamma_{postfix}',
                                              initializer=self.gamma_diag_initializer,
                                              regularizer=self.gamma_diag_regularizer,
                                              constraint=self.gamma_diag_constraint)
        self.moving_V[postfix] = self.add_weight(shape=param_shape,
                                                 initializer=self.moving_variance_initializer,
                                                 name=f'moving_V{postfix}',
                                                 trainable=False)
    self.beta = {}
    self.moving_mean = []
    for p1 in range(4):
      postfix = dim_names[p1]
      self.beta[postfix] = self.add_weight(shape=(input_shape[0][self.axis],),
                                           name=f'beta{[postfix]}',
                                           initializer=self.beta_initializer,
                                           regularizer=self.beta_regularizer,
                                           constraint=self.beta_constraint)
      self.moving_mean.append(self.add_weight(shape=(input_shape[0][self.axis],),
                                              initializer=self.moving_mean_initializer,
                                              name=f'moving_mean{[postfix]}',
                                              trainable=False))
    self.built = True

  def call(self, inputs, training=None):
    # inputs is an array of 4 tensors
    input_shape = inputs[0].shape  # typically [BS, H, W, C]. For unittest (1,2,3,5)
    ndims = len(input_shape)  # typically 4
    reduction_axes = [i for i in range(ndims) if i != self.axis]  # [0, 1, 2]

    # substract by mean
    mu = []
    broadcast_mu = []
    broadcast_mu_shape = [1] * ndims
    broadcast_mu_shape[self.axis] = input_shape[self.axis]  # unittest [1, 1, 1, 5]
    for i in range(4):
      mu.append(tf.math.reduce_mean(inputs[i], axis=reduction_axes))  # compute mean for all blades. Unittest gives tf.Tensor([1 3 4 5 6], shape=(5,), dtype=int32)
      broadcast_mu.append(tf.reshape(mu[i], broadcast_mu_shape))
    input_centred = [inputs[i] - broadcast_mu[i] for i in range(4)]

    # compute covariance matrix
    v = {}
    dim_names = "rijk"
    for p1 in range(4):
      for p2 in range(p1, 4):
        postfix = dim_names[p1]+dim_names[p2]
        if self.scale:
          v[postfix] = tf.reduce_mean(input_centred[p1] * input_centred[p2], axis=reduction_axes)
          if p1 == p2:
            v[postfix] += self.epsilon

        elif self.center:
          v[postfix] = None
        else:
          raise ValueError('Error. Both scale and center in batchnorm are set to False.')

    input_bn = quaternion_bn(
        input_centred, v,
        self.beta,
        self.gamma,
        axis=self.axis
    )  # unittest shape : 4* shape=(1, 2, 3, 5)
    training_value = tf_utils.constant_value(training)
    if training_value == False:  # not the same as "not training_value" because of possible none
      return input_bn

    update_list = []
    for i in range(4):
      update_list.append(tf.keras.backend.moving_average_update(self.moving_mean[i], mu[i], self.momentum))
    for p1 in range(4):
      for p2 in range(p1, 4):
        postfix = dim_names[p1]+dim_names[p2]
        update_list.append(tf.keras.backend.moving_average_update(self.moving_V[postfix], v[postfix], self.momentum))
    self.add_update(update_list, inputs)

    def normalize_inference():
      if self.center:
        inference_centred = [inputs[i] - tf.reshape(self.moving_mean[i], broadcast_mu_shape) for i in range(4)]
      else:
        inference_centred = inputs
      return quaternion_bn(
          inference_centred, self.moving_V,
          self.beta,
          self.gamma,
          axis=self.axis
      )

    # Pick the normalized form corresponding to the training phase.
    return tf.keras.backend.in_train_phase(input_bn, normalize_inference, training=training)

  def get_config(self):
    config = {
        'axis': self.axis,
        'momentum': self.momentum,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'beta_initializer': initializers.serialize(self.beta_initializer),
        'gamma_diag_initializer': initializers.serialize(self.gamma_diag_initializer) if self.gamma_diag_initializer != sqrt_init else 'sqrt_init',
        'gamma_off_initializer': initializers.serialize(self.gamma_off_initializer),
        'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
        'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer) if self.moving_variance_initializer != sqrt_init else 'sqrt_init',
        'moving_covariance_initializer': initializers.serialize(self.moving_covariance_initializer),
        'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
        'gamma_diag_regularizer': tf.keras.regularizers.serialize(self.gamma_diag_regularizer),
        'gamma_off_regularizer': tf.keras.regularizers.serialize(self.gamma_off_regularizer),
        'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
        'gamma_diag_constraint': tf.keras.constraints.serialize(self.gamma_diag_constraint),
        'gamma_off_constraint': tf.keras.constraints.serialize(self.gamma_off_constraint),
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
