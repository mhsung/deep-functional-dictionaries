# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..'))

from datasets import *
from generate_outputs import *
from scipy.optimize import linear_sum_assignment
#import matplotlib.pyplot as plt
import numpy as np


def compute_all_keypoints(sess, net, data):
    P = data.point_clouds
    assert(P.shape[0] == data.n_data)
    assert(P.shape[1] == data.n_points)

    KP = data.keypoints
    assert(KP.shape[0] == data.n_data)
    assert(KP.shape[1] == data.n_labels)

    A = predict_A(P, sess, net)
    assert(A.shape[0] == data.n_data)
    assert(A.shape[1] == data.n_points)
    assert(A.shape[2] == net.K)

    pred_KP = np.argmax(A, axis=1)

    return P, KP, pred_KP


def evaluate_PCK(P, KP, pred_KP):
    n_data = P.shape[0]
    n_points = P.shape[1]
    n_labels = KP.shape[1]
    K = pred_KP.shape[1]

    # dists_info: (point_cloud_index, label, basis_index, distance)
    dists_info = []

    for k in range(n_data):
        # NOTE:
        # Skip if the keypoint does not exist.
        labels = [i for i in range(n_labels) if KP[k,i] >= 0]

        # Find the closest prediction (w/o matching).
        for i, label in enumerate(labels):
            all_dists = np.zeros(K)

            idx_i = KP[k,label]
            assert(idx_i < n_points)
            p_i = P[k,idx_i]

            for j in range(K):
                idx_j = pred_KP[k,j]
                assert(idx_j < n_points)
                p_j = P[k,idx_j]

                all_dists[j] = np.linalg.norm(p_i - p_j)

            j = np.argmin(all_dists)
            dists_info.append((k, i, j, all_dists[j]))

    dists_info = np.array(dists_info)

    return dists_info


def evaluate_PCK_after_label_basis_matching(P, KP, pred_KP):
    n_data = P.shape[0]
    n_points = P.shape[1]
    n_labels = KP.shape[1]
    K = pred_KP.shape[1]

    # Find the best mapping from labels to bases.
    all_dists = np.zeros((n_data, n_labels, K))
    label_counts = np.zeros(n_labels)

    for k in range(n_data):
        for i in range(n_labels):

            # NOTE:
            # Skip if the keypoint does not exist.
            if KP[k,i] < 0: continue

            idx_i = KP[k,i]
            assert(idx_i < n_points)
            p_i = P[k,idx_i]

            label_counts[i] += 1.

            for j in range(K):
                idx_j = pred_KP[k,j]
                assert(idx_j < n_points)
                p_j = P[k,idx_j]

                all_dists[k,i,j] += np.linalg.norm(p_i - p_j)

    mean_dists = np.sum(all_dists, axis=0) / \
            np.expand_dims(label_counts, axis=-1)
    row_ind, col_ind = linear_sum_assignment(mean_dists)


    # dists_info: (point_cloud_index, label, basis_index, distance)
    dists_info = []

    for k in range(n_data):
        for (i, j) in zip(row_ind, col_ind):
            if KP[k,i] < 0: continue
            dists_info.append((k, i, j, all_dists[k,i,j]))

    dists_info = np.array(dists_info)

    return dists_info


def save_results(dists_info, out_dir, postfix=None):
    # dists_info: (point_cloud_index, label, basis_index, distance)
    dists = dists_info[:,3]

    if postfix is not None:
        out_file = os.path.join(out_dir, 'distances_{}.npy'.format(postfix))
    else:
        out_file = os.path.join(out_dir, 'distances.npy')

    np.save(out_file, dists)
    print("Saved '{}'.".format(out_file))

    '''
    # Draw plot.
    n_matches = dists.size

    x_list = np.linspace(0.0, 0.1, 20 + 1)
    counts = np.zeros(x_list.size, dtype=int)

    for i in range(x_list.size):
        counts[i] = np.sum(dists <= x_list[i])

    y_list = counts.astype(x_list.dtype) / float(n_matches)

    plt.clf()
    plt.plot(x_list, y_list)
    plt.ylim(0., 1.)
    plt.yticks(np.linspace(0., 1., 10 + 1))

    if postfix is not None:
        out_file = os.path.join(out_dir, 'pck_{}.png'.format(postfix))
    else:
        out_file = os.path.join(out_dir, 'pck.png')

    plt.savefig(out_file)
    print("Saved '{}'.".format(out_file))
    '''


def evaluate(sess, net, data, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    P, KP, pred_KP = compute_all_keypoints(sess, net, data)

    dists = evaluate_PCK(P, KP, pred_KP)
    save_results(dists, out_dir)

    dists_after_matching = evaluate_PCK_after_label_basis_matching(
            P, KP, pred_KP)
    save_results(dists_after_matching, out_dir, postfix='after_matching')

