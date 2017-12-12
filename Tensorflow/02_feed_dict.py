import tensorflow as tf

# ex1
a = tf.placeholder(tf.float32, shape=[3])

b = tf.constant([5, 5, 5], tf.float32)

c = a + b

with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: [1, 2, 3]}))

# ex2
a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
    replace_dict = {a: 15}
    print(sess.run(b, feed_dict=replace_dict))