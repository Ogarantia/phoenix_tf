import tensorflow as tf
import argparse, sys, os
from model_conv import Model as model_conv
from model_separable_conv import Model as model_separable_conv
from model_dense import Model as model_dense
import shutil
import tempfile
import time


MODELS = {'model_conv' : model_conv,
          'model_separable_conv': model_separable_conv,
          'model_dense' : model_dense}

script_path = os.path.dirname(os.path.abspath(__file__))

def framework_mapping(upstride_bool, datatype_int):
  """ Converts a pair of upstride_bool and datatype_int into a framework.
  """
  print('*')
  if datatype_int == 0 and upstride_bool: # scalar upstride
    import upstride.scalar.tf.keras.layers as up
  elif datatype_int == 0 and not upstride_bool: # scalar tensorflow
    framework = tf.keras.layers
  elif datatype_int == 2: # type2 upstride (phoenix OR python)
    import upstride.type2.tf.keras.layers as up
  # elif datatype_int == 2 and not upstride_bool:
  #   # raise ValueError(f"Incoherent arguments datatype_int {datatype_int} and upstride_bool False. When upstride_bool is false, datatype_int is expected to be 0.")
  else:
    raise ValueError("The datatype_int chosen is not implemented.")

  if datatype_int == 0:
    factor = 1
  else:
    factor = 4

  if upstride_bool:
    print("* Using Upstride phoenix")
    framework = up
  else:
    if datatype_int == 0:
      print("* Using Tensorflow")
    else:
      framework = up
      print("* Using Upstride python")

  print('*')
  return framework, factor


def main():
  parser = argparse.ArgumentParser("Trains or runs inference of a small model on CIFAR100",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--upstride', "-u", type=int, default=1,
                      help='If set to 0, Tensorflow/Tensorflow-based engine is used, Upstride Phoenix otherwise.')
  parser.add_argument('--datatype', "-d", type=int, default=0,
                      help='Upstride datatype index (an integer)')
  parser.add_argument('--logdir', "-ld", type=str, default="/tmp/log",
                      help='Path to a directory to write out Tensorboard logs')
  parser.add_argument('--epochs', "-e", type=int, default=1,
                      help='Number of epochs (used in training only)')
  parser.add_argument('--batch_size', "-bs", type=int, default=128,
                      help='Batch size')
  parser.add_argument('--train', "-t", type=float, default=1,
                      help='If set to 1, running training in single floating point precision (32 bits).'
                           'If set to 0.5, running training in mixed precision (16 and 32 bits).'
                           'Else, running inference only.')
  parser.add_argument('--print_model_summary', "-p", type=int, default=0,
                      help='If set to 0, does not print model summary. Otherwise, print model summary')
  parser.add_argument('--model', "-m", type=str, default="",
                      help='Selects a model to test. If no model is provided, it iterates over all the available models. '
                           'Options: , '.join(iter(MODELS.keys())))
  args = parser.parse_args()
  framework, factor = framework_mapping(args.upstride, args.datatype)

  # enable memory growth to avoid OOM issues
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  # prepare CIFAR10
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
  x_train, x_test = tf.cast(x_train, tf.float32) / 255.0, tf.cast(x_test, tf.float32) / 255.0
  # TODO: introduce channels_last option when running in TF on CPU
  dataformat = 'channels_first'
  x_train = tf.transpose(x_train, [0, 3, 1, 2])
  x_test = tf.transpose(x_test, [0, 3, 1, 2])
  tf.keras.backend.set_image_data_format(dataformat)

  # prepare a model
  train_models = []
  if args.model == "":
    files = [f for f in os.listdir(script_path) if os.path.isfile(os.path.join(script_path, f))]
    if not files:
      print(f"Beware, no file ending by \"_bench.py\" found in {script_path}")
    for f in files:
      if f.startswith("model_"):
        fname = os.path.splitext(f)[0]
        train_models.append(fname)
  else:
    train_models.append(args.model)

  for i in train_models:
    model = MODELS[i](framework, args.upstride, args.datatype, factor, dataformat, nclasses=100)
    if args.print_model_summary:
      model.summary()

    if args.train in [0.5, 1]:
      if args.train == 0.5:
        # enable MPT
        print("Training with mixed precision")
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)


      # run training
      tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=args.logdir, histogram_freq=1, profile_batch=[2, 5], write_graph=False, write_images=False)
      model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
      start_time = time.time()
      model.fit(x_train, y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_data=(x_test, y_test),
                callbacks=[tensorboard_cb, tf.keras.callbacks.TerminateOnNaN()])
      duration = time.time() - start_time
      print(f'Trained {i} in {"mixed" if args.train == 0.5 else "standard"} precision training with {args.epochs} epochs and batch size {args.batch_size}.')
      path = tempfile.mkdtemp()
      tf.saved_model.save(model, path)
      shutil.rmtree(path)  # remove local model directory
    else:
      # run inference
      start_time = time.time()
      model.predict(x_test, batch_size=args.batch_size, callbacks=[tf.keras.callbacks.TerminateOnNaN()])
      duration = time.time() - start_time
      print(f'Inference of {i} with batch size {args.batch_size} took {round(duration, 3)} ms')

  print('All good.')

if __name__ == "__main__":
  main()
