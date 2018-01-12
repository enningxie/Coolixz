# Regularization
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

learning_rate = 0.01
training_epochs = 1000
reg_lambda = 0.


# create a helper method to split the dataset
def split_dataset(x_dataset, y_dataset, ratio):
    arr = np.arange(x_dataset.size)
    np.random.shuffle(arr)
    num_train = int(ratio * x_dataset.size)
    x_train = x_dataset[arr[0:num_train]]
    y_train = y_dataset[arr[0:num_train]]
    x_test = x_dataset[arr[num_train:x_dataset.size]]
    y_test = y_dataset[arr[num_train:x_dataset.size]]
    return x_train, x_test, y_train, y_test


# create fake dataset y = x^2
x_dataset = np.linspace(-1, 1, 100)

num_coeffs = 9
y_dataset_params = [0.] * num_coeffs
y_dataset_params[2] = 1
y_dataset = 0
for i in range(num_coeffs):
    y_dataset += y_dataset_params[i] * np.power(x_dataset, i)

# make some noise
y_dataset += np.random.randn(*x_dataset.shape) * 0.3

# split the dataset into 70% train_set and 30% test_set
(x_train, x_test, y_train, y_test) = split_dataset(x_dataset, y_dataset, 0.7)

# placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# define the model
def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)

w = tf.Variable([0.] * num_coeffs, name="param")
y_model = model(X, w)
cost = tf.div(tf.add(tf.reduce_sum(tf.square(Y-y_model)), tf.multiply(reg_lambda, tf.reduce_sum(tf.square(w)))), 2*x_train.size)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for reg_lambda in np.linspace(0, 1, 100):
        for epoch in range(training_epochs):
            sess.run(train_op, feed_dict={X: x_train, Y: y_train})
        final_cost = sess.run(cost, feed_dict={X: x_test, Y: y_test})
        print("reg lambda", reg_lambda)
        print("final cost", final_cost)