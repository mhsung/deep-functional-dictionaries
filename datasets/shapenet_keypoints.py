# Minhyuk Sung (mhsung@cs.stanford.edu)
# May 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from global_variables import *
from dataset_keypoints import DatasetKeypoints
import h5py
import numpy as np


def create_dataset(exp_type, batch_size, split_name):
    category_name = 'Chair'
    category_id = '03001627'

    hdf5_file = os.path.join(g_shapenet_keypoints_dir,
        '{}_{}.h5'.format(category_name, split_name))
    f = h5py.File(hdf5_file)
    mesh_names = f['mesh_names'][:]
    point_clouds = f['point_clouds'][:]
    keypoints = f['keypoints'][:]
    print("Loaded '{}'.".format(hdf5_file))

    # NOTE:
    # Load ShapeNet mesh files if exist.
    '''
    mesh_files = [os.path.join(g_shapenet_core_dir, category_id, x, 'model.obj')
            for x in mesh_names]
    for x in mesh_files: assert(os.path.exists(x))
    '''
    mesh_files = None

    n_data = point_clouds.shape[0]
    n_points = point_clouds.shape[1]
    print(point_clouds.shape)
    print(keypoints.shape)
    n_labels = keypoints.shape[1]

    return DatasetKeypoints('Keypoints[{}]'.format(category_name), exp_type,
            batch_size, point_clouds, keypoints, mesh_files=mesh_files)

