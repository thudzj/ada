# coding: utf-8
import tensorflow as tf
import tensorflow.contrib.layers as ly
from utils import concat_y

#28*28->16*16
def mnist2usps(x, y, reuse=False):
    with tf.variable_scope('generator') as scope:
        if reuse:
            scope.reuse_variables()
        img = ly.conv2d(x, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME')
        img = ly.conv2d(img, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME')
        img = ly.conv2d(img, num_outputs=512, kernel_size=3, stride=2, activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME')

        img = ly.conv2d_transpose(concat_y(img, y), 128, 5, stride=2,
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME')
        img = ly.conv2d_transpose(concat_y(img, y), 1, 5, stride=2,
                                    activation_fn=tf.nn.tanh, padding='SAME')
    return img
