#!/usr/bin/python

#------------------------------------------------------------------------------
# Define all the global variables for the project
#------------------------------------------------------------------------------

from __future__ import division
import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

# Specify the data directories.
g_shapenet_parts_dir = os.path.join(BASE_DIR, 'datasets', 'shapenet_parts')
g_S3DIS_dir = os.path.join(BASE_DIR, 'datasets', 'S3DIS')
