import tensorflow as tf

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
linear_model = W*x + b

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    # print(sess.run(linear_model, feed_dict={x: [1, 2, 3, 4]}))
    print(W.eval())