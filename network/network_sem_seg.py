# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from architectures import *
import tensorflow as tf


class NetworkSemSeg(object):
    def __init__(self, n_points, n_dim, n_labels, batch_size,
            init_learning_rate, decay_step, decay_rate, bn_decay_step,
            net_options):
        self.n_points = n_points
        self.n_dim = n_dim
        self.n_labels = n_labels
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.bn_decay_step = bn_decay_step

        # Parse network options.
        print(' - Network options: {}'.format(','.join(net_options)))
        self.use_stn = False
        self.pointnet2 = False

        for opt in net_options:
            if opt == 'use_stn': self.use_stn = True
            elif opt == 'pointnet2': self.pointnet2 = True
            else: print('[Warning] Unknown network option: ({})'.format(opt))


        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self.global_step = tf.Variable(0)

            if self.bn_decay_step < 0:
                # Disable batch normalization.
                self.bn_decay = None
            else:
                self.bn_decay = get_batch_norm_decay(
                        self.global_step, self.batch_size, self.bn_decay_step)
                tf.summary.scalar('bn_decay', self.bn_decay)

            self.is_training = tf.placeholder(tf.bool, shape=())

            # Build network.
            self.build_net(self.is_training, self.bn_decay)

            self.learning_rate = get_learning_rate(
                    self.init_learning_rate, self.global_step, self.batch_size,
                    self.decay_step, self.decay_rate)
            tf.summary.scalar('learning_rate', self.learning_rate)

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    self.loss, global_step=self.global_step)

            # Define merged summary.
            self.summary = tf.summary.merge_all()

            # Define saver.
            self.saver = tf.train.Saver(max_to_keep=0)


    def build_net(self, is_training, bn_decay):
        # FIXME:
        # Make the placeholders to have dynamic sizes.

        # Point clouds.
        self.P = tf.placeholder(tf.float32,
                shape=[self.batch_size, self.n_points, self.n_dim])

        # Mask.
        self.b = tf.placeholder(tf.float32,
                shape=[self.batch_size, self.n_points])

        # One-hot labels.
        self.L = tf.placeholder(tf.bool,
                shape=[self.batch_size, self.n_points, self.n_labels])


        scope = 'PointNet'
        with tf.variable_scope(scope) as sc:
            # A: (B, N, L)
            if self.pointnet2:
                if self.use_stn:
                    print("[Warning] PointNet++ does not include STN.")

                self.A, self.stn_loss = build_pointnet2_seg(
                        self.P, self.n_labels, is_training, bn_decay, scope)
            else:
                self.A, self.stn_loss = build_pointnet_seg(
                        self.P, self.n_labels, is_training, bn_decay, scope,
                        self.use_stn)

        labels = tf.argmax(tf.cast(self.L, tf.int32), axis=-1)


        # Point classification loss.
        # (B, N)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=self.A)
        # Multiply mask.
        losses = losses * self.b


        # Accuracy.
        # (B, N)
        correct_preds = tf.cast(tf.equal(labels, tf.argmax(self.A, axis=-1)),
                tf.float32)
        # Multiply mask.
        correct_preds = correct_preds * self.b


        with tf.name_scope('loss'):
            self.loss = tf.reduce_sum(losses) / tf.reduce_sum(self.b) + \
                    self.stn_loss
        tf.summary.scalar('loss', self.loss)

        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_sum(correct_preds) / tf.reduce_sum(self.b)
        tf.summary.scalar('accuracy', self.accuracy)

