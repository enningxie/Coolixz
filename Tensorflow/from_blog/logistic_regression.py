# logistic_regression
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

# define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30

# read data
mnist = input_data.read_data_sets("/home/enningxie/Documents/DataSets/mnist", one_hot=True)

# create placeholders for features and labels
X = tf.placeholder(tf.float32, [batch_size, 784], name='X_placeholder')
Y = tf.placeholder(tf.int32, [batch_size, 10], name='Y_placeholder')

# create weights and bias
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1, 10]), name='bias')

# build model
logits = tf.matmul(X, w) + b

# define loss func
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)

# define training op
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    # for visualize
    writer = tf.summary.FileWriter("./logisitc_reg", sess.graph)
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples/batch_size)
    for i in range(n_epochs):
        total_loss = 0
        for _ in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch
        print("Averge loss epoch {0}: {1}".format(i+1, total_loss))
    print("Total time: {0} seconds".format(time.time() - start_time))
    print('Optimization finished.')

    # test the model
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y: Y_batch})
        total_correct_preds += accuracy_batch
    X_batch, Y_batch = mnist.test.next_batch(batch_size)
    accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y: Y_batch})
    total_correct_preds += accuracy_batch
