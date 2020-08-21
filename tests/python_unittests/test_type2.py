import unittest
import tensorflow as tf
from src.python.upstride.type2.tf.keras import layers
from src.python.upstride import generic_layers

class TestUpstride(unittest.TestCase):
  def test_network(self):
    generic_layers.change_upstride_type(2, ["", "12", "23", "13"], (3, 0, 0))
    inputs = tf.keras.layers.Input(shape=(3, 224, 224))
    x = layers.TF2Upstride()(inputs)
    self.assertEqual(tuple(x.shape), (None, 3, 224, 224))
    x = layers.Conv2D(4, (3, 3), data_format = 'channels_first')(x)
    print(x.shape)
    self.assertEqual(len(x), 4)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(4, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(4, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100)(x)
    x = layers.Upstride2TF()(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    model.summary()