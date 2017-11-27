# coding: utf-8
import os
import tensorflow as tf
import tensorflow.contrib.layers as ly
from classifiers import C
from utils import concat_y, lrelu
import data
from visualize import *
import importlib
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('source', 'mnist', 'Source domain for DA.')
flags.DEFINE_string('target', 'usps', 'Target domain for DA.')
flags.DEFINE_string('source_split_name', 'train', 'Name of the train split for the source.')
flags.DEFINE_string('target_split_name', 'train', 'Name of the train split for the target.')
flags.DEFINE_string('dataset_dir', './data_dir/', 'The directory where the datasets can be found.')
flags.DEFINE_string('train_log_dir', './log/', 'Directory where to write event logs.')
flags.DEFINE_string('tag', '1', 'Tag of log dictionary.')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_integer('print_loss_steps', 100, 'The frequency with which the losses are printed, in steps.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('gpu', '0', 'GPU.')
flags.DEFINE_integer('eval_steps', 1000, 'The frequency with which the classifier are evaluated, in steps.')
flags.DEFINE_integer('discriminator_steps', 1, 'The number of times we run the discriminator train_op in a row.')
flags.DEFINE_integer('generator_steps', 1, 'The number of times we run the generator train_op in a row.')

def discriminator(x, y, reuse=False, is_training=True):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        img = ly.conv2d(concat_y(x, y), num_outputs=128, kernel_size=5,
                        stride=2, activation_fn=lrelu, normalizer_fn=None,
                        weights_initializer=tf.random_normal_initializer(stddev=0.02),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
        img = ly.conv2d(concat_y(img, y), num_outputs=256, kernel_size=5,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.9, 'epsilon': 0.001,},
                        weights_initializer=tf.random_normal_initializer(stddev=0.02),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
        img = ly.conv2d(concat_y(img, y), num_outputs=512, kernel_size=5,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.9, 'epsilon': 0.001,},
                        weights_initializer=tf.random_normal_initializer(stddev=0.02),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
        img = ly.conv2d(concat_y(img, y), num_outputs=512, kernel_size=5,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.9, 'epsilon': 0.001,},
                        weights_initializer=tf.random_normal_initializer(stddev=0.02),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
        logit = ly.fully_connected(tf.reshape(img, [img.shape.as_list()[0], -1]), 1, activation_fn=None)
    return logit

def main(_):
    batch_size = FLAGS.batch_size
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    max_iter_step = 60000
    FLAGS.train_log_dir += FLAGS.source + '2' + FLAGS.target + '/'
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
    FLAGS.train_log_dir += FLAGS.tag + '/'
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
        os.makedirs(FLAGS.train_log_dir + 'images/')
    tf.logging.set_verbosity(tf.logging.INFO)
    source_data = getattr(data.get_dataset(FLAGS.source, FLAGS.dataset_dir), FLAGS.source_split_name)
    target_data = getattr(data.get_dataset(FLAGS.target, FLAGS.dataset_dir), FLAGS.target_split_name)
    source_image_shape = source_data._images.shape[1:]
    target_image_shape = target_data._images.shape[1:]
    print(np.max(source_data._images), np.min(source_data._images))
    print(np.max(target_data._images), np.min(target_data._images))

    with tf.device('/gpu:0'):

      xs = tf.placeholder(dtype=tf.float32, shape=(batch_size,) + source_image_shape)
      ys = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
      xt = tf.placeholder(dtype=tf.float32, shape=(batch_size,) + target_image_shape)

      generator = getattr(importlib.import_module('generators'), FLAGS.source + '2' + FLAGS.target)
      xt_ = generator(xs, ys, reuse=False)

      yt_ = C(xt, FLAGS.target, is_training=True, reuse=False)
      yt_hard_ = tf.argmax(yt_, axis=1)
      yt_test_ = tf.argmax(C(xt, FLAGS.target, is_training=False, reuse=True), axis=1)

      real_logit = discriminator(xt_, ys, reuse=False)
      fake_logit = discriminator(xt, yt_hard_, reuse=True)

      d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit)
               + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logit), logits=fake_logit))
      g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real_logit), logits=real_logit))
      c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit) * tf.reduce_max(tf.nn.softmax(yt_), axis=1))

      theta_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
      theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
      theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')

      counter_d = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
      opt_d = ly.optimize_loss(loss=d_loss, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(FLAGS.lr, counter_d, decay_steps=20000, decay_rate=0.9, staircase=True),
                        beta1=0.5),
                    variables=theta_d, global_step=counter_d,
                    summaries = 'gradient_norm')

      counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
      opt_g = ly.optimize_loss(loss=g_loss, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(FLAGS.lr, counter_g, decay_steps=20000, decay_rate=0.9, staircase=True),
                        beta1=0.5),
                    variables=theta_g, global_step=counter_g,
                    summaries = 'gradient_norm')

      counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
      opt_c = ly.optimize_loss(loss=c_loss, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(FLAGS.lr, counter_c, decay_steps=20000, decay_rate=0.9, staircase=True),
                        beta1=0.5),
                    variables=theta_c, global_step=counter_c,
                    summaries = 'gradient_norm')
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = False
    #config.gpu_options.per_process_gpu_memory_fraction = 0.95
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(FLAGS.train_log_dir, sess.graph)

        for i in range(max_iter_step):
            for j in range(FLAGS.discriminator_steps):
                bxs, bys = source_data.next_batch(batch_size)
                bxt, _ = target_data.next_batch(batch_size)
                _ = sess.run([opt_d], feed_dict={xs: bxs, ys: bys, xt: bxt})

            for j in range(FLAGS.generator_steps):
                bxs, bys = source_data.next_batch(batch_size)
                bxt, _ = target_data.next_batch(batch_size)
                _, _ = sess.run([opt_g, opt_c], feed_dict={xs: bxs, ys: bys, xt: bxt})

            if i % 100 == 99:
                bxs, bys = source_data.next_batch(batch_size)
                bxt, _ = target_data.next_batch(batch_size)
                loss_d_, loss_g_, loss_c_ = sess.run([d_loss, g_loss, c_loss], feed_dict={xs: bxs, ys: bys, xt: bxt})
                print("Training ite %d, d_loss: %f, g_loss: %f, c_loss: %f" % (i, loss_d_, loss_g_, loss_c_))

                bxt_ = sess.run(xt_, feed_dict={xs: bxs, ys: bys})
                fig = plt.figure('tda')
                grid_show(fig, (bxs + 1) / 2,  list(source_image_shape), 211)
                grid_show(fig, (bxt_ + 1) / 2,  list(target_image_shape), 212)
                fig.savefig(FLAGS.train_log_dir + 'images/' + str(int((i-99)/100)) + '.png')

            if i % 1000 == 999 or i == 0:
                saver.save(sess, os.path.join(FLAGS.train_log_dir, "model.ckpt"), global_step=i)

                target_data._index_in_epoch = 0
                accs = []
                for j in range(int(target_data._num_examples/batch_size)):
                    bxt, byt = target_data.next_batch(batch_size)
                    byt_pred = sess.run(yt_test_, feed_dict={xt:bxt})
                    accs.append(np.mean(np.equal(byt, byt_pred).astype(np.float32)))
                print(np.mean(accs))


if __name__ == '__main__':
  tf.app.run()
