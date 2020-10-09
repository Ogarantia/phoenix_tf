import tensorflow as tf

def Model(framework, upstride, datatype_int=0, factor=1, dataformat='channels_first', input_shape=(3, 32, 32), nclasses=10):
  # set up the model layer by layer
  inputs = tf.keras.layers.Input(input_shape)
  x = inputs
 
  if datatype_int != 0:
    x = framework.TF2Upstride()(x)

  kwargs = {'require_input_grad': False} if upstride else {}
  
  # Unit 1 - Regular conv
  x = framework.Conv2D(16//factor, 1,
                       padding='same',
                       use_bias=False,
                       name='conv_1', **kwargs)(x)
  x = framework.BatchNormalization()(x)
  x = framework.Activation('relu')(x)

  # Unit 2 - Separable conv
  x = framework.DepthwiseConv2D((3, 3),
                                strides=2,
                                padding='same',
                                use_bias=True,
                                name='conv_2_dw')(x)
  x = framework.Conv2D(32//factor, 1,
                       padding='same',
                       use_bias=False,
                       name='conv_2_pw')(x)
  x = framework.BatchNormalization()(x)
  x = framework.Activation('relu')(x)

  # Unit 3 - Separable conv
  x = framework.DepthwiseConv2D((3, 3),
                                strides=2,
                                padding='same',
                                use_bias=True,
                                name='conv_3_dw')(x)
  x = framework.Conv2D(64//factor, 1,
                       padding='same',
                       use_bias=False,
                       name='conv_3_pw')(x)
  x = framework.BatchNormalization()(x)
  x = framework.Activation('relu')(x)

  # Unit 4 - Separable conv
  x = framework.DepthwiseConv2D((3, 3),
                                strides=2,
                                padding='same',
                                use_bias=True,
                                name='conv_4_dw')(x)
  x = framework.Conv2D(128//factor, 1,
                       padding='same',
                       use_bias=False,
                       name='conv_4_pw')(x)
  x = framework.BatchNormalization()(x)
  x = framework.Activation('relu')(x)

  # Unit 5
  x = framework.GlobalAveragePooling2D()(x)
  x = framework.Flatten()(x)
  x = framework.Dense(nclasses,
                      use_bias=True,
                      name='dense')(x)

  if datatype_int != 0:
    x = framework.Upstride2TF()(x)

  # construct the model and write out its summary
  logits = tf.keras.layers.Softmax()(x)
  model = tf.keras.models.Model(inputs, logits)
  return model