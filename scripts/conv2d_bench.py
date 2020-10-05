import sys
import os
import numpy
import time
import argparse
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../src/python")
import upstride as up
from upstride.type_generic.custom_op import upstride_ops
import upstride.scalar.tf.keras.layers as uplayers
import tensorflow.keras.layers as tflayers

# padding = 'SAME'
# strides = [2, 5]
# dilations = [3, 2]
padding = 'VALID'
strides = [1, 1]
dilations = [1, 1]

context = {
  "op_name": "conv2d",
  "logdir": "."
}

def main():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', "-ld", type=str, default=".", help='')
  args = parser.parse_args()
  
  context["logdir"] = args.logdir

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  input_channels = [3, 16, 32]
  output_channels = [32, 64, 128]
  image_sizes = [112, 32]
  batch_sizes = [1, 2, 5, 8, 12]
  use_biases = [True, False]

  # first run for preparing the computer
  # print('Warmup run')
  for input_channel in input_channels:
    for output_channel in output_channels:
      for image_size in image_sizes:
        for batch_size in batch_sizes:
          for use_bias in use_biases:
            benchmark(input_channel,
                      output_channel,
                      image_size,
                      batch_size,
                      use_bias)

  # 3 runsto benchmark
  with open(f'{context["logdir"]}/{context["op_name"]}/{context["op_name"]}_output.csv', 'w+') as f:
    f.write(f"iter,input_channel,output_channel,image_size,batch_size,use_bias,engine,execution time\n")
  for i in range(3):
    for input_channel in input_channels:
      for output_channel in output_channels:
        for image_size in image_sizes:
          for batch_size in batch_sizes:
            for use_bias in use_biases:
              benchmark(input_channel,
                        output_channel,
                        image_size,
                        batch_size,
                        use_bias)


def benchmark_engine(filter, inputs, bias, use_bias, engine):
  inputs = inputs.copy()
  num_rep = len(inputs)
  times = [0] * num_rep
  filters = filter.shape[0]
  kernel_size = filter.shape[2:]
  if 'phoenix_tf' in engine.__file__:
    data_format = 'channels_first'
    bias = tf.expand_dims(bias, axis=0)
  elif 'tensorflow' in engine.__file__:
    data_format = 'channels_last'
    for i in range(num_rep):
      inputs[i] = tf.transpose(inputs[i], [0, 2, 3, 1])
    filter = tf.transpose(filter, [2, 3, 1, 0])
  else:
    raise ValueError('The engine provided does not exist or is not available.')
  bias = bias if use_bias else []
  # first run to init tf
  conv2d = engine.Conv2D(filters=filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=padding,
                         dilation_rate=dilations,
                         use_bias=use_bias,
                         data_format=data_format) # TODO implement possibility of toggling use_bias
  conv2d(inputs[12])
  upstride_ops.wait()    # wait till all the kernels in CUDA stream are actually executed
  if use_bias:
    conv2d.set_weights([filter, bias])
  else:
    conv2d.set_weights([filter])

  for i in range(num_rep):
    start = time.time()
    output = conv2d(inputs[i])
    upstride_ops.wait()
    times[i] = 1000 * (time.time() - start)

  if 'tensorflow' in engine.__file__:
    output = tf.transpose(output, [0, 3, 1, 2])
  return output, times


def benchmark(input_channel, output_channel, image_size, batch_size, use_bias):
  print(f'benchmark with: {input_channel}, {output_channel}, {image_size}, {batch_size}, {use_bias}')

  num_rep = 20
  filter = tf.random.uniform((output_channel, input_channel, 3, 3), dtype=tf.float32) / (3*3*input_channel)
  inputs = [tf.random.uniform((batch_size, input_channel, image_size, image_size), dtype=tf.float32) for _ in range(num_rep)]
  bias = tf.random.uniform((output_channel, ), dtype=tf.float32)

  output_tf, times_tf = benchmark_engine(filter, inputs, bias, use_bias, engine=tflayers)
  output_up, times_up = benchmark_engine(filter, inputs, bias, use_bias, engine=uplayers)

  err = tf.math.reduce_max(tf.math.abs(output_up - output_tf))
  print('Error:', err.numpy())

  if not os.path.exists(f'{context["logdir"]}/{context["op_name"]}'):
    os.makedirs(f'{context["logdir"]}/{context["op_name"]}')

  with open(f'{context["logdir"]}/{context["op_name"]}/{context["op_name"]}_output.csv', 'a') as f:
    for i in range(num_rep):
      f.write(f"{i}, {input_channel}, {output_channel}, {image_size}, {batch_size}, {use_bias}, upstride, {times_up[i]}\n")
      f.write(f"{i}, {input_channel}, {output_channel}, {image_size}, {batch_size}, {use_bias}, tf, {times_tf[i]}\n")


if __name__ == "__main__":
  main()
