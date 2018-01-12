# 2-hidden layers nn using tensorflow's higher level API tf.estimator
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

# Input minist
mnist = input_data.read_data_sets('/home/enningxie/Documents/DataSets/mnist/', one_hot=False)

# parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 128
display_step = 100
num_layer_1 = 256
num_layer_2 = 128
num_features = 784
num_classes = 10


# Define the Dnn structure
def DNN(input_data_):
    input_data = input_data_['images']
    layer1 = tf.layers.dense(input_data, num_layer_1, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, num_layer_2, activation=tf.nn.relu)
    output = tf.layers.dense(layer2, num_classes)
    return output


# define the model_fn
def model_fn(features, labels, mode):
    # build the dnn
    logits = DNN(features)

    # predict
    predicted_class = tf.argmax(logits, 1)
    pred_prob = tf.nn.softmax(logits)

    # prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={"pred": predicted_class})

    # loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # evaluate
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_class)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': accuracy}
    )
    return estim_specs


# main
def main():
    # init
    model = tf.estimator.Estimator(model_fn=model_fn)

    # Define the training input
    train_input = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images},
        y=mnist.train.labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )

    # train the model
    model.train(input_fn=train_input, steps=num_steps)

    # evaluate step
    eval_input = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images},
        y=mnist.test.labels,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=False
    )

    # evaluate the model
    e = model.evaluate(input_fn=eval_input)

    print("From test set accu: ", e['accuracy'])

    # prediction
    pred_input = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images[:5]},
        num_epochs=1,
        shuffle=False
    )

    pred = model.predict(input_fn=pred_input)
    for i, p in enumerate(pred):
        print("Prediction %s: %s" % (i + 1, p["pred"]))


if __name__ == '__main__':
    main()
