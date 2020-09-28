import tensorflow as tf

# tf.__version__ is of type string and is manually casted into a number
tf_version_list = tf.__version__.split('.')
tf_version = int(tf_version_list[0]) + 0.1*int(tf_version_list[1])