import tensorflow as tf


x = tf.constant(-2.0, name='x', dtype=tf.float32)
a = tf.constant(5.0, name='a', dtype=tf.float32)
b = tf.constant(13.0, name='b', dtype=tf.float32)

y = tf.Variable(tf.add(tf.multiply(a, x), b))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    print(sess.run(y))

tf.where()