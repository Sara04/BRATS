"""CNN common functions."""
import tensorflow as tf


def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def _weight_variable_xavier(shape):
    initial =\
        tf.get_variable("W", shape,
                        initializer=tf.contrib.layers.xavier_initializer())
    return initial


def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def _conv(x, weights, strides=[1, 1, 1, 1], padding_mode='VALID'):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1],
                        padding=padding_mode)


def _max_pool(x, ksize=[1, 2, 2, 1]):
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='VALID')
