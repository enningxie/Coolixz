import tensorflow as tf

# tf.Variable()

# Ex1
W = tf.Variable(10)
assign_op = W.assign(100)

with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())
    print(sess.run(assign_op))

# Ex2
my_var = tf.Variable(2, name="my_var")
my_var_times_two = my_var.assign(2 * my_var)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(my_var_times_two))
    print(sess.run(my_var_times_two))
    print(sess.run(my_var_times_two))

# Ex3
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))
print(sess2.run(W.assign_sub(2)))

print(sess1.run(W.assign_add(100)))
print(sess2.run(W.assign_sub(50)))

sess1.close()
sess2.close()