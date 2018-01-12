# nearest neighbor using tensorflow
import tensorflow as tf
import numpy as np

# input mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/enningxie/Documents/DataSets/mnist/", one_hot=True)

num_test = 200

# train/test data
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# tf graph input
xtr = tf.placeholder(tf.float32, [None, 784])
xte = tf.placeholder(tf.float32, [784])

# distance
dis = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# prediction
pred = tf.argmin(dis, 0)

accuracy = 0.

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(num_test):
        index = sess.run(pred, feed_dict={xtr: x_train, xte: x_test[i]})
        print("Test", i, "Prediction: ", np.argmax(y_train[index]), "True Class: ", np.argmax(y_test[i]))

        # Calculate accuracy
        if np.argmax(y_train[index]) == np.argmax(y_test[i]):
            accuracy += 1. / num_test
    print("Done!")
    print("Accuracy:", accuracy)