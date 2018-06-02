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
    hdf5_file_list_txt = os.path.join(g_shapenet_parts_dir,
            '{}_hdf5_file_list.txt'.format(split_name))
    assert(os.path.exists(hdf5_file_list_txt))

    with open(hdf5_file_list_txt, 'r') as f:
        hdf5_file_list = f.read().splitlines()

    point_clouds = []
    labels = []
    category_ids = []

    for i, hdf5_file in enumerate(hdf5_file_list):
        f = h5py.File(os.path.join(g_shapenet_parts_dir, hdf5_file))
        point_clouds.append(f['data'][:])
        labels.append(f['pid'][:])
        category_ids.append(f['label'][:])
        print("Loaded '{}'.".format(hdf5_file))

    point_clouds = np.concatenate(point_clouds)
    labels = np.concatenate(labels)
    category_ids = np.concatenate(category_ids)

    category_name_file = os.path.join(g_shapenet_parts_dir,
            'all_object_categories.txt')
    assert(os.path.exists(category_name_file))
    with open(category_name_file, 'r') as f:
        category_names = f.read().splitlines()
        for i, name in enumerate(category_names):
            category_names[i] = name.split('\t')[0]

    print(category_names)

    return Dataset('ShapeNetParts', exp_type, batch_size, point_clouds, labels,
            category_ids=category_ids, category_names=category_names)

