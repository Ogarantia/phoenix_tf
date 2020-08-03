import tensorflow as tf
from src.python.upstride.type2.tf.keras.layers import *
import numpy as np
np.set_printoptions(precision=2, threshold=10000)

def test(input_size=9, filter_size=3, in_channels=1, out_channels=1, padding='SAME', strides=[1,1], dilations=[1,1]):    

  seed = 42
  tf.random.set_seed(seed)
  filter = tf.random.uniform((out_channels, in_channels, filter_size, filter_size), dtype=tf.float32)
  input = tf.random.uniform((1, in_channels, input_size, input_size), dtype=tf.float32)
  input_t = tf.transpose(input, [0, 2, 3, 1])
  filter_t = tf.transpose(filter, [2, 3, 1, 0])

  input_t = tf.identity(input_t)
  filter_t = tf.identity(filter_t)

  # input_t = tf.convert_to
  # filter = 2*tf.ones_like(filter)
  # filter_flatten = tf.range(1, tf.size(filter) + 1, 1, dtype=tf.float32)
  # filter = tf.reshape(filter_flatten, tf.shape(filter))
  # print("filter", filter)
  # input_flatten = tf.range(1, tf.size(input) + 1, 1, dtype=tf.float32)
  # input = tf.reshape(input_flatten, tf.shape(input)) + 3
  # print("input", input)
  
  with tf.GradientTape() as gt:
    gt.watch(input)
    output_test = upstride_ops.upstride_conv2d(
              input, filter,
              strides=strides,
              padding=padding,
              dilations=dilations,
              data_format='NCHW')
  grad_test = gt.gradient(output_test, input) 
  
  with tf.GradientTape() as gt:
    gt.watch(filter_t)
    # #                            N  H  W  C 
    # input = tf.transpose(input, [0, 2, 3, 1])
    # #                              H  W  I  O  
    # filter = tf.transpose(filter, [2, 3, 1, 0])
    output_ref = tf.nn.conv2d(
              input_t, filter_t,
              strides=strides,
              padding=padding,
              dilations=dilations) 
  
  # filter = tf.transpose(filter, [3, 2, 0, 1])
  grad_reference = gt.gradient(output_ref, filter_t)
  #                                              O  I  H  W 
  grad_reference = tf.transpose(grad_reference, [3, 2, 0, 1])
  
  output_ref = tf.transpose(output_ref, [0, 3, 1, 2])
  print("conv_Test")
  print(output_test)
  print("conv_Ref")
  print(output_ref)
  print("DIFF")
  print(output_ref - output_test)

  print("Grad_Test")
  print(grad_test)
  print("Grad_Ref")
  print(grad_reference)
  print("Grad_Ref - Grad_test")
  print(grad_reference - grad_test)

test()