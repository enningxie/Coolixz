# Simple linear regression example in tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import utils

DATA_FILE = '/home/enningxie/Documents/Codes/stanford-tensorflow-tutorials/data/fire_theft.xls'

# step1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# step2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# step3: create weight and bias, initialized t0 0
w = tf.Variable(0.0, name='weight')
b = tf.Variable(0.0, name='bias')

# step4: build model to predict Y
Y_predicted = X * w + b

# step5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss') / n_samples

# step6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    # step7: initialize the necessary variables, in the case, w and b
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    # step8: train the model
    for i in range(1000):  # train the model 100 epochs
        total_loss = 0
        for x, y in data:
            # Session runs train_op and fetch values of loss
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        print("total_loss: ", total_loss)
    writer.close()

    # step9: output the values of w and b
    w, b = sess.run([w, b])

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()