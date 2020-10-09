import tensorflow as tf
from packaging import version

def Model(framework, upstride, datatype_int=0, factor=1, dataformat='channels_first', input_shape=(3, 32, 32), nclasses=10):
  inputs = tf.keras.layers.Input(input_shape)
  x = inputs

  if datatype_int != 0:
    x = framework.TF2Upstride()(x)

  kwargs = {'require_input_grad': False} if upstride else {}
  if tf.config.experimental.list_physical_devices('GPU') != [] and version.parse(tf.__version__) < version.parse("2.3"):
    kwargs['groups'] = 3
  x = framework.Conv2D(12 // factor, 3, data_format=dataformat, use_bias=False, strides=[1, 2], activation=tf.nn.relu, **kwargs)(x)
  x = framework.Conv2D(4 // factor, 3, data_format=dataformat, use_bias=True, padding='VALID', dilation_rate=[2, 3], activation=tf.nn.relu)(x)
  x = framework.Conv2D(8 // factor, 3, data_format=dataformat, use_bias=False, padding='SAME', activation=tf.nn.relu)(x)

  if datatype_int != 0:
    x = framework.Upstride2TF()(x)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(nclasses)(x)
  logits = tf.keras.layers.Softmax()(x)
  model = tf.keras.models.Model(inputs, logits)
  return model