#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '..', 'network'))

from global_variables import *
from datasets import *
from main import run
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--exp_type', type=str, default='ours',\
        choices=['ours', 'sem_seg'])
parser.add_argument('--eval_type', action='append', default=['eval_keypoints'])

parser.add_argument('--net_options', action='append',
        default=['column_softmax', 'clip_A', 'use_stn'])

parser.add_argument('--in_model_dirs', type=str, default='', help='')
parser.add_argument('--in_model_scopes', type=str, default='', help='')
parser.add_argument('--out_model_dir', type=str, default='model', help='')
parser.add_argument('--out_dir', type=str, default='outputs', help='')
parser.add_argument('--log_dir', type=str, default='log', help='')

parser.add_argument('--train', action="store_true", help='')
parser.add_argument('--init_learning_rate', type=float, default=0.0001)
parser.add_argument('--decay_step', type=int, default=6243 * 20)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--bn_decay_step', type=int, default=6243 * 20 * 2)

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--snapshot_epoch', type=int, default=50)
parser.add_argument('--validation_epoch', type=int, default=10)

parser.add_argument('--K', type=int, default=10)
parser.add_argument('--l21_norm_weight', type=float, default=0.0)

parser.add_argument('--part_removal_fraction', type=float, default=0.0)
parser.add_argument('--indicator_noise_probability', type=float, default=0.0)

args = parser.parse_args()


if __name__ == '__main__':
    # Set root directory.
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'ShapeNetKeypoints')

    if args.exp_type == 'ours':
        root_dir += '_{:d}'.format(args.K)
        root_dir += '_{:f}'.format(args.l21_norm_weight)

        if args.part_removal_fraction > 0.:
            root_dir += '_part_removal_{:f}'.format(
                    args.part_removal_fraction)
        if args.indicator_noise_probability > 0.:
            root_dir += '_indicator_noise_{:f}'.format(
                    args.indicator_noise_probability)
    else:
        root_dir += '_{}'.format(args.exp_type)

    assert('column_softmax' in args.net_options)
    assert('clip_A' in args.net_options)
    assert('use_stn' in args.net_options)
    if 'pointnet2' in args.net_options:
        root_dir += '_pn2'

    if not os.path.exists(root_dir): os.makedirs(root_dir)


    if args.train:
        args.out_model_dir = os.path.join(root_dir, 'model')
    else:
        args.in_model_dirs = os.path.join(root_dir, 'model')
    args.out_dir = os.path.join(root_dir, 'outputs')
    args.log_dir = os.path.join(root_dir, 'log')


    # Create data.
    train_data = shapenet_keypoints.create_dataset(
            args.exp_type, args.batch_size, 'train')
    test_data = shapenet_keypoints.create_dataset(
            args.exp_type, args.batch_size, 'test')

    '''
    if args.part_removal_fraction > 0.:
        train_data.remove_parts(args.part_removal_fraction)
    if args.indicator_noise_probability > 0.:
        train_data.set_indicator_noise_probability(
                args.indicator_noise_probability)
    '''

    # Run.
    run(args, train_data, test_data, test_data)

