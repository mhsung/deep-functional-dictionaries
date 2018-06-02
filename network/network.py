# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from architectures import *
import tensorflow as tf


class Network(object):
    def __init__(self, n_points, n_dim, n_seg_ids, K, batch_size,
            init_learning_rate, decay_step, decay_rate, bn_decay_step,
            l21_norm_weight, net_options):
        assert(K > 1)

        self.n_points = n_points
        self.n_dim = n_dim
        self.n_seg_ids = n_seg_ids
        self.K = K
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.bn_decay_step = bn_decay_step
        self.l21_norm_weight = l21_norm_weight

        # Parse network options.
        print(' - Network options: {}'.format(','.join(net_options)))
        self.softmax = False
        self.column_softmax = False
        self.sigmoid = False
        self.unit_column = False
        self.clip_A = False
        self.use_stn = False
        self.pointnet2 = False
        self.resnet = False

        for opt in net_options:
            if opt == 'softmax': self.softmax = True
            elif opt == 'column_softmax': self.column_softmax = True
            elif opt == 'sigmoid': self.sigmoid = True
            elif opt == 'unit_column': self.unit_column = True
            elif opt == 'clip_A': self.clip_A = True
            elif opt == 'use_stn': self.use_stn = True
            elif opt == 'pointnet2': self.pointnet2 = True
            elif opt == 'resnet': self.resnet = True
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

        # Functions.
        self.b = tf.placeholder(tf.float32,
                shape=[self.batch_size, self.n_points])

        # Linear combination weights.
        self.x = tf.placeholder(tf.float32,
                shape=[self.batch_size, self.K])

        # One-hot labels.
        self.S = tf.placeholder(tf.bool,
                shape=[self.batch_size, self.n_points, self.n_seg_ids])


        scope = 'PointNet'
        with tf.variable_scope(scope) as sc:
            # A: (B, N, K)
            if self.pointnet2:
                if self.use_stn:
                    print("[Warning] PointNet++ does not include STN.")

                self.A, self.stn_loss = build_pointnet2_seg(
                        self.P, self.K, is_training, bn_decay, scope)
            elif self.resnet:
                if self.use_stn:
                    print("[Warning] PointNet++ does not include STN.")

                self.A, self.stn_loss = build_resnet_seg(
                        self.P, self.K, is_training, bn_decay, scope)
            else:
                self.A, self.stn_loss = build_pointnet_seg(
                        self.P, self.K, is_training, bn_decay, scope,
                        self.use_stn)

            if self.clip_A:
                self.A = tf.clip_by_value(self.A, -10., +10.)

            if self.softmax:
                assert(not self.column_softmax)
                assert(not self.sigmoid)
                self.A = tf.nn.softmax(self.A, dim=-1)

            if self.column_softmax:
                assert(not self.softmax)
                assert(not self.sigmoid)
                self.A = tf.nn.softmax(self.A, dim=1)

            if self.sigmoid:
                assert(not self.softmax)
                assert(not self.column_softmax)
                self.A = tf.nn.sigmoid(self.A)

            if self.unit_column:
                self.A = tf.nn.l2_normalize(self.A, dim=1)


        # Matrix computations for solving constrained least square.
        # A^T A: (B, K, K)
        self.AtA = tf.matmul(tf.transpose(self.A, perm=[0, 2, 1]), self.A)

        # b^T A: (B, K)
        self.btA = tf.squeeze(tf.matmul(tf.expand_dims(self.b, 1), self.A))


        # Projection loss.
        # b_pred: (B, N)
        self.b_pred = tf.reduce_sum(tf.multiply(
            self.A, tf.expand_dims(self.x, -2)), axis=-1)

        # ||Ax - b||_2^2
        self.proj_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(self.b_pred - self.b), axis=-1))


        # Regularization loss.
        # ||A||_2,1
        self.l21_norm_loss = tf.reduce_mean(
                tf.norm(tf.norm(self.A, ord=2, axis=-2), ord=1, axis=-1))


        # Accuracy.
        self.binarized_A = tf.cast(tf.one_hot(tf.argmax(self.A, axis=-1),
            depth=self.A.get_shape()[-1].value), dtype=tf.bool)

        # (B, N, S, 1)
        S_exp = tf.expand_dims(self.S, axis=-1)
        # (B, N, 1, K)
        binarized_A_exp = tf.expand_dims(self.binarized_A, axis=-2)

        # (B, S, K)
        intersections = tf.reduce_sum(tf.cast(tf.logical_and(
            S_exp, binarized_A_exp), tf.float32), axis=1)
        unions = tf.reduce_sum(tf.cast(tf.logical_or(
            S_exp, binarized_A_exp), tf.float32), axis=1)
        self.IoU = tf.where(unions > 0., intersections / unions,
                tf.zeros_like(unions))


        with tf.name_scope('loss'):
            if self.column_softmax or self.unit_column:
                self.loss = (self.proj_loss + self.l21_norm_weight * \
                        self.l21_norm_loss) + self.stn_loss
            else:
                self.loss = (self.proj_loss + self.l21_norm_weight * \
                        self.l21_norm_loss) / self.n_points + \
                        self.stn_loss
        tf.summary.scalar('loss', self.loss)

