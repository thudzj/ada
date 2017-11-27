# coding: utf-8
import tensorflow as tf
import tensorflow.contrib.layers as ly

slim = tf.contrib.slim

def C(images,
        model,
        num_classes=10,
        is_training=False,
        reuse=False):

  kwargs = {
      'num_classes': num_classes,
      'is_training': is_training,
      'reuse': reuse,
      'scope': 'classifier'
  }

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      weights_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
      normalizer_params={'is_training': is_training, 'decay': 0.9, 'epsilon': 0.001,},
      weights_initializer=tf.random_normal_initializer(stddev=0.02),
      #normalizer_fn=slim.batch_norm
  ):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      if model == 'mnist':
        logits, _ = mnist_classifier(images, **kwargs)
      elif model == 'svhn':
        logits, _ = svhn_classifier(images, **kwargs)
      elif model == 'usps':
        logits, _ = usps_classifier(images, **kwargs)
      else:
        raise ValueError('Unknown task classifier %s' % model)

  return logits

def mnist_classifier(images,
                     is_training=False,
                     num_classes=10,
                     reuse=False,
                     scope='mnist'):
  net = {}

  with tf.variable_scope(scope, reuse=reuse):
    net['conv1'] = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net['pool1'] = slim.max_pool2d(net['conv1'], [2, 2], 2, scope='pool1')

    net['conv2'] = slim.conv2d(net['pool1'], 48, [5, 5], scope='conv2')
    net['pool2'] = slim.max_pool2d(net['conv2'], [2, 2], 2, scope='pool2')
    net['fc3'] = slim.fully_connected(
        slim.flatten(net['pool2']), 100, scope='fc3')
    net['fc4'] = slim.fully_connected(
        slim.flatten(net['fc3']), 100, scope='fc4')
    logits = slim.fully_connected(
        net['fc4'], num_classes, activation_fn=None, scope='fc5', normalizer_fn=None)
  return logits, net


def svhn_classifier(images,
                    is_training=False,
                    num_classes=10,
                    reuse=False,
                    private_scope='svhn'):

  net = {}

  with tf.variable_scope(scope, reuse=reuse):
    net['conv1'] = slim.conv2d(images, 64, [5, 5], scope='conv1')
    net['pool1'] = slim.max_pool2d(net['conv1'], [3, 3], 2, scope='pool1')

    net['conv2'] = slim.conv2d(net['pool1'], 64, [5, 5], scope='conv2')
    net['pool2'] = slim.max_pool2d(net['conv2'], [3, 3], 2, scope='pool2')
    net['conv3'] = slim.conv2d(net['pool2'], 128, [5, 5], scope='conv3')

    net['fc3'] = slim.fully_connected(
        slim.flatten(net['conv3']), 3072, scope='fc3')
    net['fc4'] = slim.fully_connected(
        slim.flatten(net['fc3']), 2048, scope='fc4')

    logits = slim.fully_connected(
        net['fc4'], num_classes, activation_fn=None, scope='fc5', normalizer_fn=None)

  return logits, net

def usps_classifier(images,
                    num_classes=10,
                    is_training=False,
                    reuse=False,
                    scope='usps'):

  with tf.variable_scope(scope, reuse=reuse):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')

    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
    net = slim.flatten(net)

    net = slim.fully_connected(net, 128, scope='fc3')
    net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')

    logits = slim.fully_connected(
        net, num_classes, activation_fn=None, scope='fc4', normalizer_fn=None)

    return logits, None
