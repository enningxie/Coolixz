# the basic operations using TensorFlow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# the Constant op
a = tf.constant(2)
b = tf.constant(3)

# launch the default graph.
with tf.Session() as sess:
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants: %i" % sess.run(a*b))


# tf graph input
# placeholder op
c = tf.placeholder(tf.int32)
d = tf.placeholder(tf.int32)

# define some op
add = tf.add(c, d)
mul = tf.multiply(c, d)

# launch
with tf.Session() as sess:
    print("Addition with variables: %i" % sess.run(add, feed_dict={c: 2, d: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={c: 2, d: 3}))

# about matrix
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

# matmul
product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)