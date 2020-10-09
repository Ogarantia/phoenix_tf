import tensorflow as tf

def Model(framework, upstride, datatype_int=0, factor=1, dataformat='channels_first', input_shape=(3, 32, 32), nclasses=10):
  inputs = tf.keras.layers.Input(input_shape)
  x = inputs

  if datatype_int != 0:
    x = framework.TF2Upstride()(x)

  x = framework.Flatten()(x)
  kwargs = {'require_input_grad': False} if upstride else {}
  x = framework.Dense(64 // factor, use_bias=False, activation=tf.nn.relu, **kwargs)(x)
  x = framework.Dense(256 // factor, use_bias=True, activation=tf.nn.relu)(x)
  x = framework.Dense(nclasses)(x)

  if datatype_int != 0:
    x = framework.Upstride2TF()(x)

  logits = tf.keras.layers.Softmax()(x)
  model = tf.keras.models.Model(inputs, logits)
  return model