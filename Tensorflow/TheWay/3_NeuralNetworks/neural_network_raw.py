# dnn using tensorflow
# 2-Hidden layers fully connected nn
from __future__ import print_function

import tensorflow as tf

# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/enningxie/Documents/DataSets/mnist/", one_hot=True)

# parameters
learning_rate = 0.01
num_classes = 10
num_features = 784
n_layer1 = 256
n_layer2 = 128
batch_size = 128
training_steps = 100
display_step = 1

# weights and bias
weight = {
    "w1": tf.Variable(initial_value=tf.truncated_normal([num_features, n_layer1])),
    "w2": tf.Variable(initial_value=tf.truncated_normal([n_layer1, n_layer2])),
    "w3": tf.Variable(initial_value=tf.truncated_normal([n_layer2, num_classes])),
}

bia = {
    "b1": tf.Variable(initial_value=tf.truncated_normal([n_layer1])),
    "b2": tf.Variable(initial_value=tf.truncated_normal([n_layer2])),
    "b3": tf.Variable(initial_value=tf.truncated_normal([num_classes]))
}


# define the model
def model(input_data):
    layer1_out = tf.nn.relu(tf.add(tf.matmul(input_data, weight['w1']), bia['b1']))
    layer2_out = tf.nn.relu(tf.add(tf.matmul(layer1_out, weight['w2']), bia['b2']))
    output = tf.add(tf.matmul(layer2_out, weight['w3']), bia['b3'])
    return output


# tf graph input
x = tf.placeholder(tf.float32, [None, num_features])
y = tf.placeholder(tf.float32, [None, num_classes])

# loss and train
out = model(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# accuracy
pred = tf.nn.softmax(out)
accuracy_ = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(accuracy_, tf.float32))

# init variables
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    num_batch = int(mnist.train.num_examples / batch_size)
    for step in range(training_steps):
        avg_loss = 0
        for i in range(num_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, batch_loss = sess.run([train_op, loss], feed_dict={x: batch_x, y: batch_y})
            avg_loss += batch_loss / num_batch
        if (step+1) % display_step == 0:
            print("epoch: {0}, loss: {1}".format(step+1, avg_loss))
    print("optimization end.")
    print("---------------------")
    print("testing...")
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("TestSet's accuracy: ", acc)