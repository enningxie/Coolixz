# HelloWorld
from __future__ import print_function

import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")

with tf.Session() as sess:
    print(sess.run(hello))