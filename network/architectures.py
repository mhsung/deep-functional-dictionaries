# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from pointnet_util import pointnet_sa_module, pointnet_fp_module
from transform_nets import *
import tensorflow as tf
import tf_util


def get_batch_norm_decay(global_step, batch_size, bn_decay_step):
    BN_INIT_DECAY = 0.5
    BN_DECAY_RATE = 0.5
    BN_DECAY_CLIP = 0.99

    bn_momentum = tf.train.exponential_decay(
                    BN_INIT_DECAY,
                    global_step*batch_size,
                    bn_decay_step,
                    BN_DECAY_RATE,
                    staircase=True)

    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def get_learning_rate(init_learning_rate, global_step, batch_size,
        decay_step, decay_rate):
    learning_rate = tf.train.exponential_decay(
                        init_learning_rate,
                        global_step*batch_size,
                        decay_step,
                        decay_rate,
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate


def build_pointnet_seg(X, out_dim, is_training, bn_decay, scope, use_spn=True):
    n_points = X.get_shape()[1].value
    n_dim = X.get_shape()[2].value

    if use_spn:
        ### STN 1 ###
        with tf.variable_scope('stn_1') as sc:
            transform_1 = input_transform_net(X, is_training, bn_decay, K=n_dim)
        X = tf.matmul(X, transform_1)
        ###

    X_expanded = tf.expand_dims(X, -1)

    net = tf_util.conv2d(X_expanded, 64, [1,n_dim], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv1')

    net = tf_util.conv2d(net, 64, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv2')

    if use_spn:
        ### STN 2 ###
        with tf.variable_scope('stn_2') as sc:
            transform_2 = feature_transform_net(net, is_training, bn_decay, K=64)
        net = tf.matmul(tf.squeeze(net, axis=2), transform_2)
        net = tf.expand_dims(net, axis=2)
        ###

    net_per_points = net

    net = tf_util.conv2d(net, 64, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv3')

    net = tf_util.conv2d(net, 128, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv4')

    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv5')

    net = tf_util.max_pool2d(net, [n_points, 1], padding='VALID',
            scope=scope+'_maxpool')

    net_global = tf.tile(net, [1, n_points, 1, 1])

    net = tf.concat([net_per_points, net_global], 3)

    net = tf_util.conv2d(net, 512, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv6')

    net = tf_util.conv2d(net, 256, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv7')

    net = tf_util.conv2d(net, 128, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv8')

    net = tf_util.conv2d(net, 128, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv9')

    net = tf_util.conv2d(net, out_dim, [1,1], padding='VALID',
            stride=[1,1], activation_fn=None, scope=scope+'_net')

    net = tf.squeeze(net)

    # Transform loss
    if use_spn:
        stn_loss = transform_loss(transform_2)
    else:
        stn_loss = 0.

    return net, stn_loss


def build_pointnet2_seg(X, out_dim, is_training, bn_decay, scope):
    n_points = X.get_shape()[1].value

    l0_xyz = tf.slice(X, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(X, [0,0,3], [-1,-1,0])

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points,
            npoint=512, radius=0.2, nsample=64, mlp=[64,64,128],
            mlp2=None, group_all=False, is_training=is_training,
            bn_decay=bn_decay, scope='layer1')

    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points,
            npoint=128, radius=0.4, nsample=64, mlp=[128,128,256],
            mlp2=None, group_all=False, is_training=is_training,
            bn_decay=bn_decay, scope='layer2')

    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points,
            npoint=None, radius=None, nsample=None, mlp=[256,512,1024],
            mlp2=None, group_all=True, is_training=is_training,
            bn_decay=bn_decay, scope='layer3')

    # Feature Propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points,
            [256,256], is_training, bn_decay, scope='fa_layer1')

    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points,
            [256,128], is_training, bn_decay, scope='fa_layer2')

    l0_points = pointnet_fp_module(l0_xyz, l1_xyz,
            tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128],
            is_training, bn_decay, scope='fa_layer3')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True,
            is_training=is_training, scope='fc1', bn_decay=bn_decay)

    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
            scope='dp1')

    net = tf_util.conv1d(net, out_dim, 1, padding='VALID', activation_fn=None,
            scope='fc2')

    return net, 0


def residual_block(X, out_dim, is_training, bn_decay, scope):
    n_dim = X.get_shape()[2].value
    X_expanded = tf.expand_dims(X, -2)

    net = tf_util.conv2d(X_expanded, out_dim, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, activation_fn=tf.nn.relu, scope=scope+'_conv1')

    net = tf_util.conv2d(net, out_dim, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, activation_fn=None, scope=scope+'_conv2')

    if n_dim != out_dim:
        X_expanded = tf_util.conv2d(X_expanded, out_dim, [1,1], padding='VALID',
                stride=[1,1], activation_fn=None, scope=scope+'_conv3')

    net = tf.nn.relu(net + X_expanded)
    net = tf.squeeze(net)

    return net


def build_resnet_seg(X, out_dim, is_training, bn_decay, scope):
    n_dim = X.get_shape()[2].value

    n_layers = 7

    net = X
    for i in range(n_layers):
        dim = out_dim if (i + 1) == n_layers else n_dim
        net = residual_block(net, dim, is_training, bn_decay,
                '{}_{:d}'.format(scope, i))

    return net, 0

