import sys
import os
sys.path.append('../src/python')
sys.path.append('src/python')
from upstride.type_generic.custom_op import upstride_ops
import tensorflow as tf
import numpy
import time
import upstride.scalar.tf.keras.layers as uplayers

operation = "dense"

def main():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  input_features = [32, 256, 1024, 4096, 8192]
  output_features = [32, 256, 1024, 4096, 8192]
  batch_sizes = [1, 2, 5, 8, 16]
  use_biases = [True, False]

  # first run for preparing the computer
  print('Dummy run')
  for input_feature in input_features:
    for output_feature in output_features:
      for batch_size in batch_sizes:
        for use_bias in use_biases:
          benchmark(input_feature, output_feature, batch_size, use_bias)

  # 3 runs to benchmark
  with open(f'{operation}/{operation}_output.csv', 'w') as f:
    f.write(f"iter,input_features,output_features,batch_size,use_bias,engine,execution time\n")
  for i in range(3):
    for input_feature in input_features:
      for output_feature in output_features:
        for batch_size in batch_sizes:
          for use_bias in use_biases:
            benchmark(input_feature, output_feature, batch_size, use_bias)

def benchmark_upstride_through_python(weights, inputs, bias, use_bias):
  inputs = inputs.copy()
  num_rep = len(inputs)
  times = [0] * num_rep
  model_up = uplayers.Dense(weights.shape[-1], use_bias=use_bias)
  model_up(inputs[12])
  if use_bias:
    model_up.bias = tf.expand_dims(bias, 0)
  model_up.kernel = weights

  for i in range(num_rep):
    start = time.time()
    output = model_up(inputs[12])
    upstride_ops.wait()
    times[i] = 1000 * (time.time() - start)
  # print_op_tensors("Upstride through Python", inputs[12], weights, output, use_bias, bias)
  return output, times

def benchmark_upstride(weights, inputs, bias, use_bias):
  inputs = inputs.copy()
  num_rep = len(inputs)
  times = [0] * num_rep
  # first run to init upstride
  initializer_out = upstride_ops.upstride_dense(
    inputs[12], weights, tf.expand_dims(bias, 0) if use_bias else [], uptype=0, require_input_grad=True, use_bias=use_bias
  )
  upstride_ops.wait()    # wait until all the kernels in CUDA stream are actually executed
 
  for i in range(num_rep):
    start = time.time()
    output = upstride_ops.upstride_dense(
      inputs[12], weights, tf.expand_dims(bias, 0) if use_bias else [], uptype=0, require_input_grad=True, use_bias=use_bias
    )
    upstride_ops.wait()
    times[i] = 1000 * (time.time() - start)
  # print_op_tensors("Upstride direct C++", inputs[12], weights, output, use_bias, bias)
  return output, times

def benchmark_tf_nn(weights, inputs, bias, use_bias):
  inputs = inputs.copy()
  num_rep = len(inputs)
  times = [0] * num_rep
  # first run to init tf (not sure if needed for tf, but it does not harm)
  output_temp = tf.linalg.matmul(inputs[12], weights)
  if use_bias:
    output_temp += tf.nn.bias_add(output_temp, bias)
  upstride_ops.wait()    # wait until all the kernels in CUDA stream are actually executed
  
  for i in range(num_rep):
    start = time.time()
    output = tf.linalg.matmul(inputs[12], weights)
    if use_bias:
      output += tf.nn.bias_add(output, bias)
    upstride_ops.wait()
    times[i] = 1000 * (time.time() - start)
  # print_op_tensors("TensorFlow NN", inputs[12], weights, output, use_bias, bias)
  return output, times

def benchmark_tf_keras(weights, inputs, bias, use_bias):
  inputs = inputs.copy()
  num_rep = len(inputs)
  times = [0] * num_rep
  # first run to init tf (not sure if needed for tf, but hwo knows)
  model_tf = tf.keras.layers.Dense(weights.shape[-1], use_bias=use_bias)
  model_tf(inputs[12])
  upstride_ops.wait()    # wait until all the kernels in CUDA stream are actually executed
  if use_bias:
    model_tf.bias = bias
  model_tf.kernel = weights

  for i in range(num_rep):
    start = time.time()
    output = model_tf(inputs[12])
    upstride_ops.wait()
    times[i] = 1000 * (time.time() - start)
  # print_op_tensors("TensorFlow keras", inputs[12], weights, output, use_bias, bias)
  return output, times

def print_op_tensors(engine_name, inputs, weights, output, use_bias, bias=[]):
  print("   ### ", engine_name)
  print("inputs\n", inputs)
  print("weights\n", weights)
  print("output\n", output)
  print("use_bias: ", use_bias)
  if use_bias:
    print("bias\n", bias)
  print("\n")

def benchmark(input_features, output_features, batch_size, use_bias):
  print(f'benchmark with: {input_features}, {output_features}, {batch_size}, {use_bias}')

  num_rep = 20
  weights = tf.random.uniform((input_features, output_features), dtype=tf.float32)
  inputs = [tf.random.uniform((batch_size, input_features), dtype=tf.float32) for _ in range(num_rep)]
  bias = tf.random.uniform((output_features, ), dtype=tf.float32)

  output_tf, times_tf = benchmark_tf_keras(weights, inputs, bias, use_bias)
  output_tf_nn, times_tf_nn = benchmark_tf_nn(weights, inputs, bias, use_bias)
  output_up, times_up = benchmark_upstride(weights, inputs, bias, use_bias)
  output_up_python, times_up_python = benchmark_upstride_through_python(weights, inputs, bias, use_bias)

  err = tf.math.reduce_max(tf.math.abs(output_up - output_tf))
  if err > 1e-2:
    raise Exception(f"Difference between output_up and output_tf is {err} and it is considered to be too high. The computations might not be the same on both engines. Aborting.")

  if not os.path.exists(operation):
    os.makedirs(operation)

  with open(f'{operation}/{operation}_output.csv', 'a') as f:
    for i in range(num_rep):
      f.write(f"{i},{input_features},{output_features},{batch_size},{use_bias},upstride_cpp,{times_up[i]}\n")
      f.write(f"{i},{input_features},{output_features},{batch_size},{use_bias},tf_keras,{times_tf[i]}\n")
      f.write(f"{i},{input_features},{output_features},{batch_size},{use_bias},upstride_python,{times_up_python[i]}\n")
      f.write(f"{i},{input_features},{output_features},{batch_size},{use_bias},tf_nn,{times_tf_nn[i]}\n")

if __name__ == "__main__":
  main()
