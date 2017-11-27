import tensorflow as tf

def concat_y(fm, y, classes=10):
    return tf.concat([fm, tf.ones(fm.shape.as_list()[:3] + [classes]) * tf.reshape(tf.one_hot(y, classes), [-1, 1, 1, classes])], -1)

def lrelu(x, leakiness=0.2):
  """Relu, with optional leaky support."""
  return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
