import tensorflow as tf
from src.python.upstride.type2.tf.keras.layers import *
import numpy as np
np.set_printoptions(precision=2, threshold=10000)

import argparse 
parser = argparse.ArgumentParser(description='Benchmarks  processor.')
parser.add_argument('--pytorch', "-p",type=int, default=0, help='')
args = parser.parse_args()

if args.pytorch:
  import torch
  from torch import nn

def test(input_size=128, filter_size=3, in_channels=2, out_channels=1, padding='SAME', strides=[1,1], dilations=[1,1]):

  input = tf.random.uniform((1, in_channels, input_size, input_size), dtype=tf.float32)
  filter = tf.random.uniform((out_channels, in_channels, filter_size, filter_size), dtype=tf.float32)

  ## UPSTRIDE
  with tf.GradientTape() as gt:
    gt.watch(filter)
    output_test = upstride_ops.upstride_conv2d(
              input, filter,
              strides=strides,
              padding=padding,
              dilations=dilations,
              data_format='NCHW')
    output_test_summed = tf.reduce_sum(output_test)
  grad_test = gt.gradient(output_test_summed, filter)

  ## PYTORCH
  if args.pytorch:
    filter_pt = torch.from_numpy(filter)
    input_pt = torch.from_numpy(input)
    input_pt.requires_grad = True
    model_pt = nn.Conv2d(in_channels, out_channels, filter_size, bias=False, padding=1, stride=strides, dilation=dilations)
    model_pt.weight = nn.Parameter(filter_pt)
    output_ref = model_pt(input_pt)
    (output_ref.sum()).backward()
    grad_reference_input = input_pt.grad
    grad_reference_PT = model_pt.weight.grad

  ## TENSORFLOW
  # #                            N  H  W  C
  input_t = tf.transpose(input, [0, 2, 3, 1])
  # #                              H  W  I  O
  filter_t = tf.transpose(filter, [2, 3, 1, 0])
  input_t = tf.identity(input_t)
  filter_t = tf.identity(filter_t)
  with tf.GradientTape(persistent=True) as gt:
    gt.watch(filter_t)
    output_ref_TF = tf.nn.conv2d(
              input_t, filter_t,
              strides=strides,
              padding=padding,
              dilations=dilations) 
  
  grad_reference = gt.gradient(output_ref_TF, filter_t)
  #                                              O  I  H  W 
  grad_reference = tf.transpose(grad_reference, [3, 2, 0, 1])

  grad_reference_TF = gt.gradient(output_ref_TF, filter_t)
  #                                              O  I  H  W 
  grad_reference_TF = tf.transpose(grad_reference_TF, [3, 2, 0, 1])
  output_ref_TF = tf.transpose(output_ref_TF, [0, 3, 1, 2])

  ## COMPARISONS
  print("conv_Test\n", output_test, "\n")
  print("conv_Ref\n", output_ref_TF, "\n")
  print("conv_Ref - conv_Test\n", output_ref_TF.numpy() - output_test.numpy(), "\n")

  print("Grad_Test\n", grad_test, "\n")
  print("Grad_Ref_TF\n", grad_reference_TF, "\n")
  print("Grad_Ref_TF - Grad_test\n", grad_reference_TF.numpy() - grad_test.numpy(), "\n")
  
  if args.pytorch:
    print("Grad_Ref_PT - Grad_Ref_TF\n", grad_reference_PT.detach().numpy() - grad_reference_TF.numpy(), "\n")

test()