# linear regression
import tensorflow as tf
import numpy as np


test_data_size = 2000
iterations = 10000
learning_rate = 0.005
global_step = 0


def generate_test_values():
    train_x = []
    train_y = []

    for _ in range(test_data_size):
        x1 = np.random.rand()
        x2 = np.random.rand()
        x3 = np.random.rand()
        y_f = 2 * x1 + 3 * x2 + 7 * x3 + 4
        train_x.append([x1, x2, x3])
        train_y.append(y_f)

    return np.array(train_x), np.transpose([train_y])


x = tf.placeholder(tf.float32, [None, 3], name='x')
W = tf.Variable(tf.zeros([3, 1]), name='W')
b = tf.Variable(tf.zeros([1]), name='b')
y = tf.placeholder(tf.float32, [None, 1])

model = tf.add(tf.matmul(x, W), b)

cost = tf.reduce_mean(tf.square(y - model))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

train_data, train_values = generate_test_values()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for _ in range(iterations):
        _, costs = sess.run([train_op, cost], feed_dict={x: train_data, y: train_values})
        global_step += 1
        if global_step % 100 == 0:
            print("step: ", global_step, " cost: ", costs)
    print("W: ", sess.run(W))
    print("b: ", sess.run(b))