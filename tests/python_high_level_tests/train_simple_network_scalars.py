import tensorflow as tf
import argparse
import sys

# tf.debugging.experimental.enable_dump_debug_info("/tmp/log", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

sys.path.append('../../src/python')


def Model(upstride=False, input_shape=(3, 32, 32), nclasses=10):
  inputs = tf.keras.layers.Input(input_shape)
  x = inputs

  if upstride:
    import upstride.type2.tf.keras.layers as up
    framework = up
    print("Using upstride")
  else:
    framework = tf.keras.layers
    print("Using Tensorflow")

  kwargs = {'require_input_grad': False} if upstride else {}
  x = framework.Conv2D(4, 3, data_format='channels_first', use_bias=False, activation=tf.nn.relu, **kwargs)(x)
  x = framework.Conv2D(4, 3, data_format='channels_first', use_bias=False, activation=tf.nn.relu)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(nclasses)(x)
  logits = tf.keras.layers.Activation('softmax')(x)
  model = tf.keras.models.Model(inputs, logits)
  model.summary()
  return model


def main():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  parser = argparse.ArgumentParser()
  parser.add_argument('--upstride', "-u", type=int, default=1, help='')
  args = parser.parse_args()

  # prepare CIFAR10
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train, x_test = tf.cast(x_train, tf.float32) / 255.0, tf.cast(x_test, tf.float32) / 255.0
  x_train = tf.transpose(x_train, [0, 3, 1, 2])
  x_test = tf.transpose(x_test, [0, 3, 1, 2])
  tf.keras.backend.set_image_data_format('channels_first')

  # prepare a model
  model = Model(args.upstride == 1)
  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  # g-g-go
  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='/tmp/log', histogram_freq=1, profile_batch=[2, 5], write_graph=False, write_images=False)
  model.fit(x_train, y_train,
            epochs=60,
            batch_size=128,
            validation_data=(x_test, y_test),
            callbacks=[tensorboard_cb,
                       tf.keras.callbacks.TerminateOnNaN()
                       ])


if __name__ == "__main__":
  main()
