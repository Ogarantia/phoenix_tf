from src.python.upstride.type2.tf.keras.layers import *
import tensorflow as tf
import numpy, time

input = tf.random.uniform((1, 3, 224, 224), dtype=tf.float32)
filter = tf.random.uniform((32, 3, 3, 3), dtype=tf.float32)

numRep = 100
timesTest = [0] * numRep
timesRef = [0] * numRep
for i in range(numRep):
    start = time.process_time()
    outputTest = upstride_ops.upstride_conv2d(
        input, filter,
        strides=[1,1],
        padding='VALID',
        data_format='NCHW'
    )
    timesTest[i] = time.process_time() - start

input = tf.transpose(input, [0,2,3,1])
filter = tf.transpose(filter, [2,3,1,0])
for i in range(numRep):
    start = time.process_time()
    outputRef = tf.nn.conv2d(
        input, filter,
        strides=[1,1],
        padding='VALID'
    )
    timesRef[i] = time.process_time() - start


outputRef = tf.transpose(outputRef, [0,3,1,2])

err = tf.math.reduce_max(tf.math.abs(outputTest - outputRef))
print('Error:', err.numpy())
print('Test time: ', numpy.mean(timesTest))
print('Reference time: ', numpy.mean(timesRef))
