# all the networks of scnn

import numpy as np
import tensorflow as tf


def std_conv_layer(input, num_filters, filter_shape, pad='same'):
    return tf.layers.conv2d(inputs=input, filters=num_filters, kernel_size=filter_shape, padding=pad, kernel_initializer=tf.random_normal(mean=0.0, stddev=0.01))


# 9x9 Regressor (R1)
class shallow_net_9x9():

    def __init__(self, input=None, name='shallow_9x9'):
        self.input = input
        self.name = name
        net = {}
        net['input'] = tf.reshape(self.input, [-1, self.input[0], self.input[1], 1])
        net['conv1_1'] = std_conv_layer(net['input'], 16, 9)
        net['pool1'] = tf.layers.max_pooling2d(net['conv1_1'], 2, 2)
        net['conv1_2'] = std_conv_layer(net['pool1'], 32, 7)
        net['pool2'] = tf.layers.max_pooling2d(net['conv1_2'], 2, 2)
        net['conv1_3'] = std_conv_layer(net['pool2'], 16, 7)
        net['conv1_4'] = std_conv_layer(net['conv1_3'], 8, 7)
        net['conv'] = std_conv_layer(net['conv1_4'], 1, 1)
        self.net = net
        self.output_layer = net['conv']