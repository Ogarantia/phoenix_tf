import tensorflow as tf
import argparse
import sys
import time

def Model(upstride=False, input_shape=(3, 32, 32), nclasses=10):
  inputs = tf.keras.layers.Input(input_shape)
  x = inputs

  if upstride:
    import upstride.scalar.tf.keras.layers as up
    framework = up
    print("Using upstride")
  else:
    framework = tf.keras.layers
    print("Using Tensorflow")

  kwargs = {'require_input_grad': False} if upstride else {}
  x = framework.Conv2D(4, 3, data_format='channels_first', use_bias=False, activation=tf.nn.relu, **kwargs)(x)
  x = framework.Conv2D(4, 3, data_format='channels_first', use_bias=False, activation=tf.nn.relu)(x)
  x = tf.keras.layers.Flatten()(x)
  x = framework.Dense(nclasses)(x)
  logits = tf.keras.layers.Activation('softmax')(x)
  model = tf.keras.models.Model(inputs, logits)
  model.summary()
  return model


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--upstride', "-u", type=int, default=1, help='')
  parser.add_argument('--logdir', "-ld", type=str, default="/tmp/log", help='')
  parser.add_argument('--epochs', "-e", type=int, default=60, help='')
  parser.add_argument('--batch_size', "-bs", type=int, default=1000, help='')
  parser.add_argument('--step_per_epoch', "-se", type=int, default=30, help='')
  args = parser.parse_args()

  # Allow to solve some memory issues on GPUs
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  # Enable debug mode for tensorboard
  tf.debugging.experimental.enable_dump_debug_info(args.logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

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

  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=args.logdir, histogram_freq=1, profile_batch=[2, 5], write_graph=False, write_images=False)

  t = time.time()
  model.fit(x_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.step_per_epoch,
            validation_data=(x_test, y_test),
            callbacks=[tensorboard_cb,
                       tf.keras.callbacks.TerminateOnNaN()
                       ])
  print(f"Training elapsed time: {time.time() - t}")


if __name__ == "__main__":
  main()
