# Polynomial_regression
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


learning_rate = 0.01
training_epochs = 100

# fake data
trX = np.linspace(-1, 1, 101)

num_coeffs = 6
trY_coeffs = [1, 2, 3, 4, 5, 6]
trY = 0
for i in range(num_coeffs):
    trY += trY_coeffs[i] * np.power(trX, i)

# add some noise
trY += np.random.randn(*trX.shape) * 1.5

# plot the fake data
plt.scatter(trX, trY)
# plt.show()

# placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# model
def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)


w = tf.Variable([0.] * num_coeffs, name="parameters")
y_model = model(X, w)

cost = tf.reduce_mean(tf.square(Y-y_model))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        total_loss = 0
        for(x, y) in zip(trX, trY):
            loss, _ = sess.run([cost, train_op], feed_dict={X: x, Y: y})
            total_loss += loss
        print("epoch: {0}, loss: {1}.".format(epoch+1, total_loss/101))
    w_val = sess.run(w)
    print("--------")
    print("w: ", w_val)
    trY2 = 0
    for i in range(num_coeffs):
        trY2 += w_val[i] * np.power(trX, i)
    plt.scatter(trX, trY)
    plt.plot(trX, trY2)
    plt.show()