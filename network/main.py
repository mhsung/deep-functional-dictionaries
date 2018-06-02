#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..'))

from global_variables import *
from datasets import *
from datetime import datetime
from generate_outputs import *
from network import Network
from network_sem_seg import NetworkSemSeg
from train_util import validate, train
import argparse
import numpy as np
import evaluate
import evaluate_sem_seg
import random
import tensorflow as tf


'''
parser = argparse.ArgumentParser()

parser.add_argument('--exp_type', type=str, default='ours',\
        choices=['ours', 'sem_seg'])
parser.add_argument('--eval_type', action='append', default=['eval'],
        help='[eval, eval_keypoints, eval_obj_det, save_dict]')

parser.add_argument('--net_options', action='append', default=['softmax', 'use_stn'],
        help='[softmax, column_softmax, sigmoid, clip_A, use_stn, pointnet2]')

parser.add_argument('--in_model_dirs', type=str, default='', help='')
parser.add_argument('--in_model_scopes', type=str, default='', help='')
parser.add_argument('--out_model_dir', type=str, default='model', help='')
parser.add_argument('--out_dir', type=str, default='outputs', help='')
parser.add_argument('--log_dir', type=str, default='log', help='')

parser.add_argument('--train', action="store_true", help='')
parser.add_argument('--init_learning_rate', type=float, default=0.001,\
        help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000,\
        help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7,\
        help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--bn_decay_step', type=int, default=200000,\
        help='Decay step for bn decay [default: 200000]')

parser.add_argument('--n_epochs', type=int, default=1000,\
        help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32,\
        help='Batch size')
parser.add_argument('--snapshot_epoch', type=int, default=100,\
        help='Interval of snapshot')
parser.add_argument('--validation_epoch', type=int, default=10,\
        help='Interval of validation')

parser.add_argument('--K', type=int, default=10,\
        help='Number of predicted basis functions [default: 10]')
parser.add_argument('--l21_norm_weight', type=float, default=0.0,
        help='L2,1 norm regularizer weight [default: 0.0]')

parser.add_argument('--part_removal_fraction', type=float, default=0.0,
        help='Fraction of parts to be removed [default: 0.0]')
parser.add_argument('--indicator_noise_probability', type=float, default=0.0,
        help='Probability of adding noise in indicator functions [default: 0.0]')

args = parser.parse_args()
'''


def load_model(sess, in_model_dir, include=''):
    # Read variables names in checkpoint.
    var_names = [x for x,_ in tf.contrib.framework.list_variables(in_model_dir)]

    # Find variables with given names.
    # HACK:
    # Convert unicode to string and remove postfix ':0'.
    var_list = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)\
            if str(x.name)[:-2] in var_names]

    if include != '':
        var_list = [x for x in var_list if include in x.name]
    #print([x.name for x in var_list])

    saver = tf.train.Saver(var_list)

    ckpt = tf.train.get_checkpoint_state(in_model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("Loaded '{}'.".format(ckpt.model_checkpoint_path))
    else:
        print ("Failed to loaded '{}'.".format(in_model_dir))
        return False
    return True


def run(args, train_data, val_data, test_data):
    tf.set_random_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    print('\n==== PARAMS ====')
    for arg in vars(args):
        print('{}={}'.format(arg, getattr(args, arg)))
    print('========\n')


    if args.exp_type == 'ours':
        net = Network(train_data.n_points, train_data.n_dim,
                test_data.n_seg_ids, args.K, args.batch_size,
                args.init_learning_rate, args.decay_step, args.decay_rate,
                args.bn_decay_step, args.l21_norm_weight, args.net_options)
    elif args.exp_type == 'sem_seg':
        print("## Sementic Segmentation ##")
        net = NetworkSemSeg(train_data.n_points, train_data.n_dim,
                train_data.n_labels, args.batch_size, args.init_learning_rate,
                args.decay_step, args.decay_rate, args.bn_decay_step,
                args.net_options)
    else:
        assert(False)


    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config, graph=net.graph) as sess:
        sess.run(tf.global_variables_initializer(), {net.is_training: True})

        if args.in_model_dirs:
            include = ''
            for in_model_dir in args.in_model_dirs.split(','):
                assert(load_model(sess, in_model_dir, include))

        if args.train:
            train(sess, net, args.exp_type, train_data, val_data,
                    n_epochs=args.n_epochs, snapshot_epoch=args.snapshot_epoch,
                    validation_epoch=args.validation_epoch,
                    model_dir=args.out_model_dir, log_dir=args.log_dir,
                    data_name=train_data.name, output_generator=None)

        train_loss, _ = validate(sess, net, args.exp_type, train_data)
        test_loss, _ = validate(sess, net, args.exp_type, test_data)

        msg = "|| Train Loss: {:6f}".format(train_loss)
        msg += " | Test Loss: {:6f}".format(test_loss)
        msg += " ||"
        print(msg)

        if args.train:
            # Save training result.
            if not os.path.exists(args.out_dir): os.makedirs(args.out_dir)
            out_file = os.path.join(args.out_dir, '{}.txt'.format(
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
            with open(out_file, 'w') as f:
                f.write(msg + '\n')
            print("Saved '{}'.".format(out_file))

        if args.exp_type == 'ours':
            if 'eval' in args.eval_type:
                evaluate.evaluate(sess, net, test_data, args.out_dir)
            if 'eval_keypoints' in args.eval_type:
                evaluate_keypoints.evaluate(sess, net, test_data, args.out_dir)
            if 'eval_obj_det' in args.eval_type:
                evaluate_obj_det.evaluate(sess, net, test_data, args.out_dir)
            if 'save_dict' in args.eval_type:
                P = test_data.point_clouds
                A = predict_A(P, sess, net)
                out_file = os.path.join(args.out_dir, 'dictionary.npy')
                np.save(out_file, A)
                print("Saved '{}'".format(out_file))
        elif args.exp_type == 'sem_seg':
            evaluate_sem_seg.evaluate(sess, net, test_data, args.out_dir)

