# import unittest
import tensorflow as tf

# class TestConv2D(unittest.TestCase):
#     def test_conv2d_grouped_test(self, img_size=8, filter_size=3, in_channels=4, out_channels=6, padding='VALID', strides=[1, 1], dilations=[1, 1], groups=1):
#         """ Runs a single convolution and compares the result with its expected output """
#         inputs_channels_first = tf.random.uniform((1, in_channels, img_size, img_size), dtype=tf.float32, minval=-0.5, maxval=0.5)
#         inputs_channels_last = tf.transpose(inputs_channels_first, [0, 2, 3, 1])
#         # inputs = tf.ones_like(inputs)

#         input_shape = inputs_channels_last.shape
#         model = tf.keras.layers.Conv2D(out_channels, filter_size, input_shape=input_shape[1:])
#         output_keras = model(inputs_channels_last)
#         print(model.weights[0].shape)

#         # run upstride convolution
#         output_nn = tf.nn.conv2d(
#           inputs_channels_last, model.weights[0],
#           strides=strides,
#           padding=padding,
#           dilations=dilations,
#           data_format='NHWC',
#         #   groups=groups
#         )

#         err = tf.math.reduce_max(tf.math.abs(output_nn - output_keras))
#         self.assertLess(err, 1e-4, f"Absolute difference with the reference is too big: {err}")


if __name__ == '__main__':
    # unittest.main()
    img_size=8
    filter_size=3
    in_channels=4
    out_channels=6
    padding='VALID'
    strides=[1, 1]
    dilations=[1, 1]
    groups=1
    
    inputs_channels_first = tf.random.uniform((1, in_channels, img_size, img_size), dtype=tf.float32, minval=-0.5, maxval=0.5)
    inputs_channels_last = tf.transpose(inputs_channels_first, [0, 2, 3, 1])
    # inputs = tf.ones_like(inputs)

    input_shape = inputs_channels_last.shape
    model = tf.keras.layers.Conv2D(out_channels, filter_size, input_shape=input_shape[1:]
    # )
    , groups=2)
    output_keras = model(inputs_channels_last)
    print(model.weights[0].shape)