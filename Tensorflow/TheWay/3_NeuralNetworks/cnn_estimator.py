# cnn using tf.estimator
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)  # for logging

# mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/enningxie/Documents/DataSets/mnist/", one_hot=False)

# paramters
num_steps = 2000
learning_rate = 0.001
droupout_train = 0.25
batch_size = 128


# cnn model
def CNN(input, reuse, keep_prob=0.0, is_training=False):
    with tf.variable_scope("cnn", reuse=reuse):
        input_data = tf.reshape(input['images'], [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(input_data, 32, (5, 5), padding='same', activation=tf.nn.relu)
        maxpool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same')
        conv2 = tf.layers.conv2d(maxpool1, 64, (3, 3), padding='same', activation=tf.nn.relu)
        maxpool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='same')
        dense_input = tf.contrib.layers.flatten(maxpool2)
        dense1 = tf.layers.dense(dense_input, 1024, activation=tf.nn.relu)
        droupout_ = tf.layers.dropout(dense1, rate=keep_prob, training=is_training)
        output = tf.layers.dense(droupout_, 10)
    return output


# def model_fn
def model_fn(features, labels, mode):
    logits = CNN(features, keep_prob=droupout_train, is_training=True, reuse=False)
    logits_ = CNN(features, reuse=True)
    pred = tf.nn.softmax(logits_)
    pred_classes = tf.argmax(pred, 1)

    # prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"pred": pred_classes})

    # loss_op
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # train_op
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())  # get global step for logging

    # acc_op
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={"acc": acc_op}
    )


# main
def main():
    model = tf.estimator.Estimator(model_fn=model_fn)

    # train
    train_input = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images},
        y=mnist.train.labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )
    model.train(input_fn=train_input, steps=num_steps)

    # eval
    eval_input = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images},
        y=mnist.test.labels,
        num_epochs=1,
        shuffle=False
    )
    eval = model.evaluate(input_fn=eval_input)
    print("test acc: ", eval['acc'])

    # pred
    predict_input = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images[:5]},
        num_epochs=1,
        shuffle=False
    )
    pred = model.predict(input_fn=predict_input)
    for i, p in enumerate(pred):
        print("num: ", str(i+1), ", pred: ", str(p['pred']))

    # end
    print('end.')


if __name__ == '__main__':
    main()
