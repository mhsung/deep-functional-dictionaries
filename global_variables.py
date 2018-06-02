#!/usr/bin/python
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Define all the global variables for the project
#------------------------------------------------------------------------------

from __future__ import division
import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))


# Executales
g_renderer = '/orions3-zfs/projects/minhyuk/app/libigl-renderer/build/OSMesaRenderer'
g_mesh_merger = '/orions3-zfs/projects/minhyuk/app/libigl-mesh-merger/build/OffscreenRenderer'


# ShapeNet segmentation directories.
g_shapenet_core_dir = '/orions3-zfs/projects/msavva/zip/ShapeNetCore.v1/'
g_shapenet_parts_dir = '/orions3-zfs/projects/minhyuk/data/shapenet_part_seg_hdf5_data/hdf5_data/'
g_shapenet_keypoints_dir = '/orions3-zfs/projects/minhyuk/data/SyncSpecCNN/Keypoints'
g_shapenet_keypoints_benchmark_file = '/orions4-zfs/projects/ericyi/SCNN/Data/keypt_pred_comparison_curve_data.mat'
g_S3DIS_dir = '/orions3-zfs/projects/minhyuk/data/SGPN/S3DIS'
g_NYUD2_dir = '/orions3-zfs/projects/minhyuk/data/SGPN/NYUD2'
g_shapenet_laplacian_dir = '/orions3-zfs/projects/minhyuk/data/SyncSpecCNN/Data/Categories'
g_MPI_FAUSE_dir = '/orions3-zfs/projects/minhyuk/data/MPI-FAUST'
