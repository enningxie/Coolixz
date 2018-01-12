import tensorflow as tf
import numpy as np


data = np.random.normal(10, 1, 100)

alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.)

update_avg = alpha * curr_value + (1 - alpha) * prev_avg

# here is what we care to visualize
avg_hist = tf.summary.scalar("running_average", update_avg)  # 1
value_hist = tf.summary.scalar("incoming_values", curr_value)

merged = tf.summary.merge_all()  # 2
writer = tf.summary.FileWriter("./logs")  # 3

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(len(data)):
        summary_str, curr_avg = sess.run([merged, update_avg], feed_dict={curr_value: data[i]})  # 4
        sess.run(tf.assign(prev_avg, curr_avg))
        print(data[i], curr_avg)
        writer.add_summary(summary_str, i)  # 5
    writer.close()