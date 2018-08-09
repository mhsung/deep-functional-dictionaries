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


def collect_labels_and_max_IoUs(S_mask, S_to_L, IoU):
    n_data = IoU.shape[0]
    n_seg_ids = IoU.shape[1]
    K = IoU.shape[2]

    labels = []
    max_IoUs = []
    dict_ids = []

    for k in range(n_data):
        assert(K >= np.sum(S_mask[k]))

        for i in range(n_seg_ids):
            # NOTE:
            # Ignore when the segment does not exist in the shape.
            if not S_mask[k,i]: continue

            label = S_to_L[k,i]
            assert(label >= 0)
            labels.append(label)

            j = np.argmax(IoU[k,i])
            max_IoUs.append(IoU[k,i,j])
            dict_ids.append(j)

    labels = np.array(labels, dtype=int)
    max_IoUs = np.array(max_IoUs)
    dict_ids = np.array(dict_ids, dtype=int)

    return labels, max_IoUs, dict_ids


def evaluate_proposal_recall(labels, max_IoUs, outputs, IoU_tol_list=None):
    outputs['Recall with threshold'] = {}
    outputs['Category recall with threshold'] = {}

    if IoU_tol_list is None:
        IoU_tol_list = np.linspace(0.5, 1.0, int((1.0 - 0.5) / 0.05) + 1)

    label_list = np.unique(labels)

    for IoU_tol in IoU_tol_list:
        outputs['Recall with threshold'][IoU_tol] = 0.
        outputs['Category recall with threshold'][IoU_tol] = {}

        # Count true positives and all instances with the IoU threshold.
        for label in label_list:
            # Number of ground truth instances.
            idxs = np.where(labels == label)[0]
            n_total = idxs.size
            assert(n_total > 0)

            # True positive
            label_max_IoUs = max_IoUs[idxs]
            n_tp = np.sum(label_max_IoUs > IoU_tol)

            outputs['Category recall with threshold'][IoU_tol][label] = \
                    float(n_tp) / float(n_total)

        n_total = max_IoUs.size
        n_tp = np.sum(max_IoUs > IoU_tol)
        outputs['Recall with threshold'][IoU_tol] = \
                float(n_tp) / float(n_total)

    return outputs


def evaluate(sess, net, data, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    P, S_mask, S_to_L, IoU = compute_all_IoU(sess, net, data)

    labels, max_IoUs, dict_ids = collect_labels_and_max_IoUs(
            S_mask, S_to_L, IoU)

    # Save files.
    np.save(os.path.join(out_dir, 'S_mask.npy'), S_mask)
    print("Saved '{}'.".format(os.path.join(out_dir, 'S_mask.npy')))

    np.save(os.path.join(out_dir, 'S_to_L.npy'), S_to_L)
    print("Saved '{}'.".format(os.path.join(out_dir, 'S_to_L.npy')))

    np.save(os.path.join(out_dir, 'IoU.npy'), IoU)
    print("Saved '{}'.".format(os.path.join(out_dir, 'IoU.npy')))

    np.save(os.path.join(out_dir, 'instance_labels.npy'), labels)
    print("Saved '{}'.".format(os.path.join(out_dir, 'instance_labels.npy')))

    np.save(os.path.join(out_dir, 'instance_max_IoUs.npy'), max_IoUs)
    print("Saved '{}'.".format(os.path.join(out_dir, 'instance_max_IoUs.npy')))

    np.save(os.path.join(out_dir, 'instance_dict_ids.npy'), dict_ids)
    print("Saved '{}'.".format(os.path.join(out_dir, 'instance_dict_ids.npy')))


    # Compute results.
    outputs = {}

    outputs = evaluate_proposal_recall(labels, max_IoUs, outputs)

    # Save results.
    print(json.dumps(outputs, sort_keys=True, indent=2))
    out_file = os.path.join(out_dir, 'evaluation.json')
    with open(out_file, 'w') as f:
        f.write(json.dumps(outputs, sort_keys=True, indent=2) + '\n')
    print("Saved '{}'.".format(out_file))

