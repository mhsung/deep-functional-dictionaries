# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import math
import numpy as np
import tensorflow as tf
import tf_util


def predict_A(P, sess, net):
    n_data = len(P)
    n_batches_in_epoch = int(math.ceil(float(n_data) / net.batch_size))
    A = None

    for index_in_epoch in range(n_batches_in_epoch):
        start = index_in_epoch * net.batch_size
        end = min(start + net.batch_size, n_data)
        n_step_size = end - start
        step_P = P[start:end]

        # NOTE:
        # Add dummy.
        if n_step_size < net.batch_size:
            assert(P.ndim > 1)
            dummy_shape = list(P.shape)
            dummy_shape[0] = net.batch_size - n_step_size
            step_P = np.vstack((step_P, np.zeros(dummy_shape)))

        step_A = sess.run(net.A, feed_dict={
            net.P: step_P, net.is_training: False})

        # NOTE:
        # Remove dummy data.
        step_A = step_A[:n_step_size]

        if index_in_epoch == 0:
            A = step_A
        else:
            A = np.vstack((A, step_A))

    return A


def save_A(P, sess, net, out_file_prefix):
    n_data = len(P)
    n_batches_in_epoch = int(math.ceil(float(n_data) / net.batch_size))
    print('# iterations: {:d}'.format(n_batches_in_epoch))

    for index_in_epoch in range(n_batches_in_epoch):
        start = index_in_epoch * net.batch_size
        end = min(start + net.batch_size, n_data)
        n_step_size = end - start
        step_P = P[start:end]

        # NOTE:
        # Add dummy.
        if n_step_size < net.batch_size:
            assert(P.ndim > 1)
            dummy_shape = list(P.shape)
            dummy_shape[0] = net.batch_size - n_step_size
            step_P = np.vstack((step_P, np.zeros(dummy_shape)))

        step_A = sess.run(net.A, feed_dict={
            net.P: step_P, net.is_training: False})

        # NOTE:
        # Remove dummy data.
        step_A = step_A[:n_step_size]

        out_file = '{}_{:d}.npy'.format(out_file_prefix, index_in_epoch)
        np.save(out_file, step_A)
        print('{} / {} processed...'.format(
            index_in_epoch + 1, n_batches_in_epoch))


def predict_IoU(P, S, sess, net):
    n_data = len(P)
    assert(len(S) == n_data)
    n_batches_in_epoch = int(math.ceil(float(n_data) / net.batch_size))
    IoU = None

    for index_in_epoch in range(n_batches_in_epoch):
        start = index_in_epoch * net.batch_size
        end = min(start + net.batch_size, n_data)
        n_step_size = end - start
        step_P = P[start:end]
        step_S = S[start:end]

        # NOTE:
        # Add dummy.
        if n_step_size < net.batch_size:
            assert(P.ndim > 1)
            P_dummy_shape = list(P.shape)
            P_dummy_shape[0] = net.batch_size - n_step_size
            step_P = np.vstack((step_P, np.zeros(P_dummy_shape)))

            assert(S.ndim > 1)
            S_dummy_shape = list(S.shape)
            S_dummy_shape[0] = net.batch_size - n_step_size
            step_S = np.vstack((step_S, np.zeros(S_dummy_shape)))

        step_IoU = sess.run(net.IoU, feed_dict={
            net.P: step_P, net.S: step_S, net.is_training: False})

        # NOTE:
        # Remove dummy data.
        step_IoU = step_IoU[:n_step_size]

        if index_in_epoch == 0:
            IoU = step_IoU
        else:
            IoU = np.vstack((IoU, step_IoU))

    return IoU

