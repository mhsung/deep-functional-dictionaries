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


def compute_all_L(sess, net, data):
    P = data.point_clouds
    assert(P.shape[0] == data.n_data)
    assert(P.shape[1] == data.n_points)

    L = data.one_hot_labels
    assert(L.shape[0] == data.n_data)
    assert(L.shape[1] == data.n_points)
    assert(L.shape[2] == data.n_labels)

    L_mask = data.label_exists
    assert(L_mask.shape[0] == data.n_data)
    assert(L_mask.shape[1] == data.n_labels)

    A = predict_A(P, sess, net)
    assert(A.shape[0] == data.n_data)
    assert(A.shape[1] == data.n_points)
    assert(A.shape[2] == data.n_labels)

    L_pred = np.zeros((data.n_data, data.n_points, data.n_labels), dtype=bool)
    for i in range(data.n_data):
        for j in range(data.n_points):
            k = np.argmax(A[i,j])
            L_pred[i,j,k] = True

    return P, L, L_mask, L_pred


def compute_mean_IoU(L, L_mask, L_pred):
    n_data = L_pred.shape[0]
    n_labels = L_pred.shape[2]

    sum_mean_IoU = 0.

    for k in range(n_data):
        sum_IoU = 0.
        count = 0

        for i in range(n_labels):
            # NOTE:
            # Ignore when the label does not exist in the shape.
            if not L_mask[k,i]: continue
            intersection = np.sum(np.logical_and(L[k,:,i], L_pred[k,:,i]))
            union = np.sum(np.logical_or(L[k,:,i], L_pred[k,:,i]))
            #assert(union > 0.)
            IoU = float(intersection) / float(union) if union > 0. else 1.
            sum_IoU += IoU
            count += 1

        sum_mean_IoU += sum_IoU / float(count)

    mean_IoU = sum_mean_IoU / float(n_data)
    return mean_IoU


def evaluate_mean_IoU(data, L, L_mask, L_pred, outputs):
    outputs['Category mIoU'] = {}

    # Per-category mean IoU
    for category_id in data.unique_category_ids:

        if data.category_names is not None:
            assert(category_id < len(data.category_names))
            category_name = data.category_names[category_id]
        else:
            category_name = '{:02d}'.format(category_id)

        mask = np.where(data.category_ids == category_id)[0]
        category_mean_IoU = compute_mean_IoU(
                L[mask], L_mask[mask], L_pred[mask])
        outputs['Category mIoU'][category_name] = category_mean_IoU

    # Total mean L_pred
    mean_IoU = compute_mean_IoU(L, L_mask, L_pred)
    outputs['Mean mIoU'] = mean_IoU

    return outputs


def evaluate(sess, net, data, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    P, L, L_mask, L_pred = compute_all_L(sess, net, data)

    outputs = {}

    outputs = evaluate_mean_IoU(data, L, L_mask, L_pred, outputs)

    # Save results.
    print(json.dumps(outputs, sort_keys=True, indent=2))
    out_file = os.path.join(out_dir, 'evaluation.json')
    with open(out_file, 'w') as f:
        f.write(json.dumps(outputs, sort_keys=True, indent=2) + '\n')
    print("Saved '{}'.".format(out_file))

