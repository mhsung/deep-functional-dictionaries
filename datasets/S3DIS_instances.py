# Minhyuk Sung (mhsung@cs.stanford.edu)
# May 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from global_variables import *
from dataset import Dataset
import h5py
import numpy as np


def create_dataset(exp_type, batch_size, split_name):
    hdf5_file_list_txt = os.path.join(g_S3DIS_dir,
            '{}_hdf5_file_list.txt'.format(split_name))
    assert(os.path.exists(hdf5_file_list_txt))

    with open(hdf5_file_list_txt, 'r') as f:
        hdf5_file_list = f.read().splitlines()

    ###
    hdf5_file_list = [os.path.basename(x) for x in hdf5_file_list]
    ###

    point_clouds = []
    labels = []
    seg_ids = []
    category_ids = []
    category_names = []

    curr_category_id = 0

    for i, hdf5_file in enumerate(hdf5_file_list):
        f = h5py.File(os.path.join(g_S3DIS_dir, hdf5_file))
        point_clouds.append(f['data'][:])
        labels.append(f['seglabel'][:])
        seg_ids.append(f['pid'][:])

        # Add scene name as category name.
        scene_name = os.path.splitext(hdf5_file)[0]
        n_scene_data = f['data'][:].shape[0]
        category_ids.append(np.full(n_scene_data, curr_category_id, dtype=int))
        category_names.append(scene_name)
        curr_category_id += 1

        print("Loaded '{}'.".format(hdf5_file))

    point_clouds = np.concatenate(point_clouds)
    labels = np.concatenate(labels)
    seg_ids = np.concatenate(seg_ids)
    category_ids = np.concatenate(category_ids)

    return Dataset('S3DIS', exp_type, batch_size, point_clouds,
            labels, seg_ids, category_ids=category_ids,
            category_names=category_names)

