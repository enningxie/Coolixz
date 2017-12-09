# Feeding data to TensorFlow
# contants
import tensorflow as tf
import numpy as np

actual_data = np.random.normal(size=[100])

data = tf.constant(actual_data)

# placeholders
data1 = tf.placeholder(tf.float32)

prediction = tf.square(data1) + 1

actual_data = np.random.normal(size=[100])

tf.Session().run(prediction, feed_dict={data1: actual_data})

# Dataset API

actual_data = np.random.normal(size=[100])
data_set = tf.contrib.data.Dataset.from_tensor_slices(actual_data)
data = data_set.make_one_shot_iterator().get_next()

# TensorFlow also overloads a range of arithmetic and logical operators:
z = -x  # z = tf.negative(x)
z = x + y  # z = tf.add(x, y)
z = x - y  # z = tf.subtract(x, y)
z = x * y  # z = tf.mul(x, y)
z = x / y  # z = tf.div(x, y)
z = x // y  # z = tf.floordiv(x, y)
z = x % y  # z = tf.mod(x, y)
z = x ** y  # z = tf.pow(x, y)
z = x @ y  # z = tf.matmul(x, y)
z = x > y  # z = tf.greater(x, y)
z = x >= y  # z = tf.greater_equal(x, y)
z = x < y  # z = tf.less(x, y)
z = x <= y  # z = tf.less_equal(x, y)
z = abs(x)  # z = tf.abs(x)
z = x & y  # z = tf.logical_and(x, y)
z = x | y  # z = tf.logical_or(x, y)
z = x ^ y  # z = tf.logical_xor(x, y)
z = ~x  # z = tf.logical_not(x)
