# logstic regeression
from __future__ import print_function

import tensorflow as tf

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/enningxie/Documents/DataSets/mnist/", one_hot=True)

# print(mnist.train.images.shape)

# parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 128
display_step = 1

# tf graph input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax

# cost, cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# train_op
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# init_op
init_op = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    sess.run(init_op)

    # training cycle
    batch_num = int(mnist.train.num_examples/batch_size)
    for epoch in range(training_epochs):
        total_loss = 0
        for i in range(batch_num):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, loss = sess.run([train_op, cost], feed_dict={x: batch_x, y: batch_y})
            total_loss += loss / batch_num
        if (epoch+1) % display_step==0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(total_loss))
    print("Optimizeation Finished!")

    # test model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print("Accuracy on test datasets: {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))
    print("done.")
