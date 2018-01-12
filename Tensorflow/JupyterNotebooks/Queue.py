# Using Queues
# If you have a lot of training data, you probably don't
# want to load into memory at once. The QueueRunner in
# TensorFlow is a tool to efficiently employ a queue data-
# structure in a multi-threaded way.

import tensorflow as tf
import numpy as np
import multiprocessing

NUM_THREADS = multiprocessing.cpu_count()

# print(NUM_THREADS)

# Generate some fake data to work with:
xs = np.random.randn(100, 3)
ys = np.random.randint(0, 2, size=100)

# Here is a couple concrete examples of our data:
xs_and_ys = zip(xs, ys)
for _ in range(5):
    x, y = next(xs_and_ys)
    print("Input {} ---> Output {}".format(x, y))

# define a queue
queue = tf.FIFOQueue(capacity=1000, dtypes=[tf.float32, tf.int32])

# set up the enqueue and dequeue ops:
enqueue_op = queue.enqueue_many([xs, ys])
x_op, y_op = queue.dequeue()

# define a queueRunner
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    for _ in range(100):
        if coord.should_stop():
            break
        x, y = sess.run([x_op, y_op])
        print(x, y)
    coord.request_stop()
    coord.join(enqueue_threads)