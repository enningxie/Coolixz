# cnn using tensorflow
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

# input mnist's data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/enningxie/Documents/DataSets/mnist/", one_hot=True)

# parameters
learning_rate = 0.001
num_steps = 2
batch_size = 128
display_step = 1
num_features = 784
num_classes = 10
dropout = 0.75

# placeholder
X = tf.placeholder(tf.float32, [None, num_features])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout(keep prob)

# weights and biases
weights = {
    # 5x5 conv, 1 inputs, 32 outputs
    'wc1': tf.Variable(initial_value=tf.truncated_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(initial_value=tf.truncated_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(initial_value=tf.truncated_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(initial_value=tf.truncated_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# conv
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# maxpool
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# create model
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, [-1, 28, 28, 1])

    # conv1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # maxpool
    conv1_ = maxpool2d(conv1)

    # conv2
    conv2 = conv2d(conv1_, weights['wc2'], biases['bc2'])
    # maxpool
    conv2_ = maxpool2d(conv2)

    # dense layer
    fc = tf.reshape(conv2_, [-1, weights['wd1'].shape.as_list()[0]])
    fc = tf.add(tf.matmul(fc, weights['wd1']), biases['bd1'])
    fc = tf.nn.relu(fc)

    # add dropout
    fc = tf.nn.dropout(fc, dropout)

    # output
    output = tf.add(tf.matmul(fc, weights['out']), biases['out'])
    return output


# main
def main():
    logits = conv_net(X, weights, biases, keep_prob)
    pred = tf.nn.softmax(logits)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op)

    accuracy = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy_op = tf.reduce_mean(tf.cast(accuracy, tf.float32))

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        num_batch = int(mnist.train.num_examples/batch_size)
        for step in range(num_steps):
            total_loss = 0
            for _ in range(num_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
                total_loss += loss / num_batch
            if (step+1) % display_step == 0:
                print("step: ", str(step+1), ", loss: ", str(total_loss))
        print("training end.")
        acc = sess.run(accuracy_op, feed_dict={X: mnist.test.images[:256], Y: mnist.test.labels[:256], keep_prob: 1.})
        print("Test acc: ", acc)
        print("done.")


if __name__ == '__main__':
    main()