from src.python.upstride.type2.tf.keras.layers import *
import tensorflow as tf
import numpy
import time

num_rep = 100
filter = tf.random.uniform((32, 3, 3, 3), dtype=tf.float32) / (3*3*3)
inputs = [tf.random.uniform((1, 3, 224, 224), dtype=tf.float32) for _ in range(num_rep)]

timesTest = [0] * num_rep
timesRef = [0] * num_rep

# first run to init upstride
upstride_ops.upstride_conv2d(
    inputs[42], filter,
    strides=[1, 1],
    padding='VALID',
    data_format='NCHW'
)

for i in range(num_rep):
  start = time.time()
  outputTest = upstride_ops.upstride_conv2d(
      inputs[i], filter,
      strides=[1, 1],
      padding='VALID',
      data_format='NCHW'
  )
  timesTest[i] = time.time() - start

# prepare tf
for i in range(num_rep):
  inputs[i] = tf.transpose(inputs[i], [0, 2, 3, 1])
filter = tf.transpose(filter, [2, 3, 1, 0])

# first run to init tf
outputRef = tf.nn.conv2d(
    inputs[42], filter,
    strides=[1, 1],
    padding='VALID'
)

for i in range(num_rep):
  start = time.time()
  outputRef = tf.nn.conv2d(
      inputs[i], filter,
      strides=[1, 1],
      padding='VALID'
  )
  timesRef[i] = time.time() - start


outputRef = tf.transpose(outputRef, [0, 3, 1, 2])

err = tf.math.reduce_max(tf.math.abs(outputTest - outputRef))
print('Error:', err.numpy())
print('Test time: ', timesTest)
print('Reference time: ', timesRef)

with open('output.csv', 'w') as f:
  f.write(f"iter, test, ref\n")
  for i in range(num_rep):
    f.write(f"{i}, {timesTest[i]}, {timesRef[i]}\n")
