# linear regression
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# some parameters
learning_rate = 0.01
training_epochs = 100

# fake data
x_train = np.linspace(-1, 1, 101)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33

plt.scatter(x_train, y_train)
# plt.show()

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# define the model
def model(X, w):
    return tf.multiply(X, w)


w = tf.Variable(0., name="weights")
y_model = model(X, w)
cost = tf.reduce_mean(tf.square(Y-y_model))

# training
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        for (x, y) in zip(x_train, y_train):
            sess.run(train_op, feed_dict={X: x, Y: y})
    w_val = sess.run(w)
    print("w: {}".format(w_val))
    plt.scatter(x_train, y_train)
    y_learned = x_train*w_val
    plt.plot(x_train, y_learned, 'g')
    plt.show()

