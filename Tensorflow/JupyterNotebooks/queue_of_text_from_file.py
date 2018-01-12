import tensorflow as tf

filename_queue = tf.train.string_input_producer(['big.txt'])

# define an op to dequeue a line from file
reader = tf.TextLineReader()
key_op, value_op = reader.read(filename_queue)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(100):
        key, value = sess.run([key_op, value_op])
        print(key, value)