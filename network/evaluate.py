# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..'))

from datasets import *
from generate_outputs import *
from scipy.optimize import linear_sum_assignment
import json
import numpy as np


def compute_all_IoU(sess, net, data):
    P = data.point_clouds
    assert(P.shape[0] == data.n_data)
    assert(P.shape[1] == data.n_points)

    S = data.one_hot_seg_ids
    assert(S.shape[0] == data.n_data)
    assert(S.shape[1] == data.n_points)
    assert(S.shape[2] == data.n_seg_ids)

    S_mask = data.seg_id_exists
    assert(S_mask.shape[0] == data.n_data)
    assert(S_mask.shape[1] == data.n_seg_ids)

    S_to_L = data.seg_ids_to_labels
    assert(S_to_L.shape[0] == data.n_data)
    assert(S_to_L.shape[1] == data.n_seg_ids)

    IoU = predict_IoU(P, S, sess, net)
    assert(IoU.shape[0] == data.n_data)
    assert(IoU.shape[1] == data.n_seg_ids)
    assert(IoU.shape[2] == net.K)

    return P, S_mask, S_to_L, IoU


def compute_mean_IoU(S_mask, IoU):
    n_data = IoU.shape[0]
    n_seg_ids = IoU.shape[1]
    K = IoU.shape[2]

    sum_mean_IoU = 0.

    for k in range(n_data):
        assert(K >= np.sum(S_mask[k]))
        sum_IoU = 0.
        count = 0

        row_ind, col_ind = linear_sum_assignment(-IoU[k])
        assert(len(row_ind) == min(n_seg_ids, K))
        assert(len(col_ind) == min(n_seg_ids, K))
        for (i, j) in zip(row_ind, col_ind):

            # NOTE:
            # Ignore when the label does not exist in the shape.
            if not S_mask[k,i]:
                assert(IoU[k,i,j] == 0.)
                continue

            sum_IoU += IoU[k,i,j]
            count += 1

        sum_mean_IoU += sum_IoU / float(count)

    mean_IoU = sum_mean_IoU / float(n_data)
    return mean_IoU


def compute_mean_IoU_after_label_basis_matching(S_mask, S_to_L, IoU):
    n_data = IoU.shape[0]
    n_seg_ids = IoU.shape[1]
    K = IoU.shape[2]

    # Find the best mapping from labels to bases.
    # NOTE:
    # Temporarily set (max label + 1) as the number of labels.
    n_labels = np.amax(S_to_L) + 1
    label_to_basis_sum_IoU = np.zeros((n_labels, K))
    label_counts = np.zeros(n_labels, dtype=int)

    for k in range(n_data):
        for i in range(n_seg_ids):

            # NOTE:
            # Ignore when the label does not exist in the shape.
            if not S_mask[k,i]:
                assert(np.all(IoU[k,i,:] == 0.))
                continue

            label = S_to_L[k,i]
            label_counts[label] += 1

            for j in range(K):
                label_to_basis_sum_IoU[label,j] += IoU[k,i,j]

    label_to_basis_mean_IoU = label_to_basis_sum_IoU / \
            np.expand_dims(label_counts, -1)

    labels = np.where(label_counts > 0)[0]
    label_to_basis_mean_IoU = label_to_basis_mean_IoU[labels, :]

    row_ind, col_ind = linear_sum_assignment(-label_to_basis_mean_IoU)
    label_to_basis = {}
    for (i, j) in zip(row_ind, col_ind):
        label_to_basis[labels[i]] = j


    sum_mean_IoU = 0.

    for k in range(n_data):
        assert(K >= np.sum(S_mask[k]))
        sum_IoU = 0.
        count = 0

        for i in range(n_seg_ids):
            if not S_mask[k,i]: continue
            label = S_to_L[k,i]
            j = label_to_basis[label]
            sum_IoU += IoU[k,i,j]
            count += 1

        sum_mean_IoU += sum_IoU / float(count)

    mean_IoU = sum_mean_IoU / float(n_data)
    return mean_IoU


def evaluate_mean_IoU(data, S_mask, IoU, outputs):
    outputs['Category mIoU'] = {}

    # Per-category mean IoU
    for category_id in data.unique_category_ids:

        if data.category_names is not None:
            assert(category_id < len(data.category_names))
            category_name = data.category_names[category_id]
        else:
            category_name = '{:02d}'.format(category_id)

        mask = np.where(data.category_ids == category_id)[0]
        category_mean_IoU = compute_mean_IoU(S_mask[mask], IoU[mask])
        outputs['Category mIoU'][category_name] = category_mean_IoU

    # Total mean IoU
    mean_IoU = compute_mean_IoU(S_mask, IoU)
    outputs['Mean mIoU'] = mean_IoU

    return outputs


def evaluate_mean_IoU_after_label_basis_matching(
        data, S_mask, S_to_L, IoU, outputs):
    outputs['Category mIoU after matching'] = {}

    # Per-category mean IoU after matching
    count = 0
    sum_mean_IoU = 0.

    for category_id in data.unique_category_ids:

        if data.category_names is not None:
            assert(category_id < len(data.category_names))
            category_name = data.category_names[category_id]
        else:
            category_name = '{:02d}'.format(category_id)

        mask = np.where(data.category_ids == category_id)[0]
        category_mean_IoU = compute_mean_IoU_after_label_basis_matching(
                S_mask[mask], S_to_L[mask], IoU[mask])
        outputs['Category mIoU after matching'][category_name] = \
                category_mean_IoU

        category_count = np.sum(mask)
        count += category_count
        sum_mean_IoU += (category_mean_IoU * category_count)

    # Total mean IoU
    mean_IoU = sum_mean_IoU / float(count)
    outputs['Mean mIoU after matching'] = mean_IoU

    return outputs


def evaluate(sess, net, data, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    P, S_mask, S_to_L, IoU = compute_all_IoU(sess, net, data)

    outputs = {}

    outputs = evaluate_mean_IoU(data, S_mask, IoU, outputs)

    outputs = evaluate_mean_IoU_after_label_basis_matching(
            data, S_mask, S_to_L, IoU, outputs)

    # Save results.
    print(json.dumps(outputs, sort_keys=True, indent=2))
    out_file = os.path.join(out_dir, 'evaluation.json')
    with open(out_file, 'w') as f:
        f.write(json.dumps(outputs, sort_keys=True, indent=2) + '\n')
    print("Saved '{}'.".format(out_file))

