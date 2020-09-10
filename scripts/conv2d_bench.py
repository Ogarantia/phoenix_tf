import sys
import os
sys.path.append('../src/python')
from upstride.type_generic.custom_op import upstride_ops
import tensorflow as tf
import numpy
import time

# padding = 'SAME'
# strides = [2, 5]
# dilations = [3, 2]
padding = 'VALID'
strides = [1, 1]
dilations = [1, 1]

operation = "conv2d"

def main():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  input_channels = [3, 16, 32]
  output_channels = [32, 64, 128]
  image_sizes = [112, 32]
  batch_sizes = [1, 2, 5, 8, 12]

  # first run for preparing the computer
  print('Dummy run')
  for input_channel in input_channels:
    for output_channel in output_channels:
      for image_size in image_sizes:
        for batch_size in batch_sizes:
          benchmark(input_channel,
                    output_channel,
                    image_size,
                    batch_size)

  # 3 runsto benchmark
  with open(f'{operation}/{operation}_output.csv', 'w') as f:
    f.write(f"iter,input_channel,output_channel,image_size,batch_size,engine,execution time\n")
  for i in range(3):
    for input_channel in input_channels:
      for output_channel in output_channels:
        for image_size in image_sizes:
          for batch_size in batch_sizes:
            benchmark(input_channel,
                      output_channel,
                      image_size,
                      batch_size)


def benchmark_upstride(filter, inputs):
  num_rep = len(inputs)
  times = [0] * num_rep
  bias = []
  use_bias = False
  # first run to init upstride
  upstride_ops.upstride_conv2d(
    inputs[12], filter, bias,
    use_bias=use_bias,
    strides=strides,
    padding=padding,
    dilations=dilations,
    data_format='NCHW'
  )
  upstride_ops.wait()    # wait till all the kernels in CUDA stream are actually executed
 
  for i in range(num_rep):
    start = time.time()
    output = upstride_ops.upstride_conv2d(
      inputs[i], filter, bias,
      use_bias=use_bias,
      strides=strides,
      padding=padding,
      dilations=dilations,
      data_format='NCHW'
    )
    upstride_ops.wait()
    times[i] = 1000 * (time.time() - start)
  return output, times


def benchmark_tf(filter, inputs):
  inputs = inputs.copy()
  num_rep = len(inputs)
  times = [0] * num_rep
  for i in range(num_rep):
    inputs[i] = tf.transpose(inputs[i], [0, 2, 3, 1])
  filter = tf.transpose(filter, [2, 3, 1, 0])
  # first run to init tf
  tf.nn.conv2d(
      inputs[12], filter,
      strides=strides,
      padding=padding,
      dilations=dilations
  )
  upstride_ops.wait()    # wait till all the kernels in CUDA stream are actually executed

  for i in range(num_rep):
    start = time.time()
    output = tf.nn.conv2d(
        inputs[i], filter,
        strides=strides,
        padding=padding,
        dilations=dilations
    )
    upstride_ops.wait()
    times[i] = 1000 * (time.time() - start)

  output = tf.transpose(output, [0, 3, 1, 2])
  return output, times


def benchmark(input_channel, output_channel, image_size, batch_size):
  print(f'benchmark with : {input_channel}, {output_channel}, {image_size}, {batch_size}')

  num_rep = 20
  filter = tf.random.uniform((output_channel, input_channel, 3, 3), dtype=tf.float32) / (3*3*input_channel)
  inputs = [tf.random.uniform((batch_size, input_channel, image_size, image_size), dtype=tf.float32) for _ in range(num_rep)]

  output_tf, times_tf = benchmark_tf(filter, inputs)
  output_up, times_up = benchmark_upstride(filter, inputs)

  err = tf.math.reduce_max(tf.math.abs(output_up - output_tf))
  #print('Error:', err.numpy())

  if not os.path.exists(operation):
    os.makedirs(operation)

  with open(f'{operation}/{operation}_output.csv', 'a') as f:
    for i in range(num_rep):
      f.write(f"{i}, {input_channel}, {output_channel}, {image_size}, {batch_size}, upstride, {times_up[i]}\n")
      f.write(f"{i}, {input_channel}, {output_channel}, {image_size}, {batch_size}, tf, {times_tf[i]}\n")


if __name__ == "__main__":
  main()
