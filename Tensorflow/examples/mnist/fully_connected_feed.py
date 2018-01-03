'''
    Train and evaluates the MNIST network using a feed dictionary.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

FLAGS = None


def placeholder_inputs(batch_size):
    '''Generate placeholder variables to represent the input tensors'''
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_Set, images_pl, labels_pl):
    images_feed, labels_feed = data_Set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed
    }
    return feed_dict


def do_eval(sess, eval_correct, images)