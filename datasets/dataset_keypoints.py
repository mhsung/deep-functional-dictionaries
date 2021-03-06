# Minhyuk Sung (mhsung@cs.stanford.edu)
# May 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from itertools import chain, combinations
import numpy as np
import random


def powerset(iterable, exclude_entire_set=False):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    # https://docs.python.org/3/library/itertools.html#recipes
    # NOTE: Do not include empty set.
    s = list(iterable)
    if exclude_entire_set:
        ret = chain.from_iterable(combinations(s, r) for r in range(1,len(s)))
    else:
        ret = chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))
    return ret


class DatasetKeypoints(object):
    def __init__(self, name, exp_type, batch_size, point_clouds, keypoints,
            mesh_files=None, category_ids=None, category_names=None):
        self.name = name
        self.exp_type = exp_type
        self.batch_size = batch_size
        self.point_clouds = point_clouds
        self.keypoints = keypoints
        self.mesh_files = mesh_files
        self.category_ids = category_ids
        self.category_names = category_names
        self.indicator_noise_probability = 0.

        self.n_data = self.point_clouds.shape[0]
        assert(self.keypoints.shape[0] == self.n_data)

        self.n_points = self.point_clouds.shape[1]
        self.n_dim = self.point_clouds.shape[2]


        ###
        self.n_labels = self.keypoints.shape[1]
        self.unique_labels = np.arange(self.n_labels)
        print(self.unique_labels)

        # Compute labels and functions from keypoints.
        kSigma = 0.001
        assert(self.n_dim == 3)

        self.functions = np.empty((self.n_data, self.n_points, self.n_labels))
        self.labels = np.full((self.n_data, self.n_points), -1, dtype=int)

        for i in range(self.n_data):
            P = self.point_clouds[i]

            for label, seed in enumerate(self.keypoints[i]):
                if seed < 0: continue
                self.labels[i,seed] = label

                sqrdists = np.sum(np.square(
                    P - np.expand_dims(P[seed], 0)), axis=-1)
                function = np.exp(-sqrdists / kSigma)
                # Normalize.
                function /= np.sum(function)
                self.functions[i,:,label] = function
        ###


        ###
        print(' - # data: {}'.format(self.n_data))
        print(' - # points: {}'.format(self.n_points))
        print(' - # labels: {}'.format(self.n_labels))
        # self.point_clouds: (n_data, n_points, n_dim)
        # self.labels: (n_data, n_points)
        ###

        if mesh_files is not None:
            assert(len(mesh_files) == self.n_data)

        if category_ids is not None:
            assert(len(category_ids) == self.n_data)
            self.unique_category_ids = np.sort(np.unique(category_ids))
            self.n_category_ids = self.unique_category_ids.size
            print(' - # categories: {}'.format(self.n_category_ids))

        if category_names is not None:
            assert(np.amax(self.unique_category_ids) < len(self.category_names))


        # Create one hot label matrix.
        #(n_data, n_points, n_labels)
        self.one_hot_labels = self.convert_to_one_hot_matrix(self.labels)
        self.label_exists = np.any(self.one_hot_labels, axis=1)


        # NOTE:
        # Consider segment IDs are the same with labels if not provided.
        self.seg_ids = np.copy(self.labels)
        self.one_hot_seg_ids = np.copy(self.one_hot_labels)
        self.seg_id_exists = np.copy(self.label_exists)
        self.n_seg_ids = self.one_hot_seg_ids.shape[-1]


        # Compute segment ID to label map.
        self.seg_ids_to_labels = np.full(
                (self.n_data, self.n_seg_ids), -1, dtype=int)
        for i in range(self.n_data):
            for j in range(self.n_points):
                seg_id = self.seg_ids[i,j]
                label = self.labels[i,j]

                # NOTE:
                # Ignore negative segment ID.
                if seg_id < 0: continue

                if self.seg_ids_to_labels[i,seg_id] >= 0:
                    assert(self.seg_ids_to_labels[i,seg_id] == label)
                else:
                    self.seg_ids_to_labels[i,seg_id] = label


        self.n_all_parts = np.sum(self.seg_id_exists)
        print(' - # all parts: {}'.format(self.n_all_parts))

        # Find all subset of segment IDs in each object.
        self.find_all_segment_subsets()


    def convert_to_one_hot_matrix(self, dense_matrix, value_set=None):
        if value_set is None:
            n = np.nanmax(dense_matrix) + 1
            value_set = np.arange(n)

        n_data = dense_matrix.shape[0]
        n_points = dense_matrix.shape[1]
        n_values = value_set.size

        one_hot_matrix = np.zeros((n_data, n_points, n_values), dtype=bool)
        for i in range(n_data):
            for j in range(n_points):
                # NOTE:
                # Ignore negative values.
                if dense_matrix[i,j] < 0: continue
                k = np.where(value_set == dense_matrix[i,j])[0]
                one_hot_matrix[i,j,k] = True

        return one_hot_matrix


    def find_all_segment_subsets(self):
        self.all_segment_subsets = [None] * self.n_data
        for i in range(self.n_data):
            unique_point_cloud_seg_ids = np.where(self.seg_id_exists[i])[0]

            # NOTE:
            # Check whether all shapes have at least two segments.
            if len(unique_point_cloud_seg_ids) < 2:
                self.all_segment_subsets[i] = [list(unique_point_cloud_seg_ids)]
            else:
                self.all_segment_subsets[i] = list(
                        powerset(unique_point_cloud_seg_ids,
                            exclude_entire_set=True))


    def remove_parts(self, part_removal_fraction):
        # Remove parts (unset @seg_id_exists elements) if fraction is greater
        # than zero.
        if part_removal_fraction == 0.: return
        assert(part_removal_fraction > 0.)
        assert(part_removal_fraction < 1.)

        n_removed_parts = int(round(self.n_all_parts * \
                part_removal_fraction))
        print('  -- {:d} parts are removed.'.format(n_removed_parts))

        row_inds, col_inds = np.where(self.seg_id_exists)
        for k in random.sample(range(self.n_all_parts), n_removed_parts):
            i = row_inds[k]
            j = col_inds[k]
            assert(self.seg_id_exists[i,j])
            self.seg_id_exists[i,j] = False
        self.n_all_parts -= n_removed_parts

        # Recompute all subset of seg_ids in each object.
        self.find_all_segment_subsets()


    def set_indicator_noise_probability(self, indicator_noise_probability):
        ###
        assert(False)
        ###


    def generate_random_samples(self, point_cloud_idxs):
        n_samples = point_cloud_idxs.size

        # Point clouds.
        P = np.empty((n_samples, self.n_points, self.n_dim))

        # Functions.
        b = np.empty((n_samples, self.n_points))

        # One-hot labels.
        # NOTE:
        # Used only for semantic segmentation and accuracy measure.
        L = np.empty((n_samples, self.n_points, self.n_labels), dtype=bool)


        for i, point_cloud_idx in enumerate(point_cloud_idxs):
            P[i] = self.point_clouds[point_cloud_idx]

            if self.exp_type == 'ours':
                # Take a random subset of *selected* segment IDs.
                seg_id_subset = random.choice(
                        self.all_segment_subsets[point_cloud_idx])
            elif self.exp_type == 'sem_seg':
                # Take all *selected* segment IDs as a mask.
                seg_id_subset = np.where(self.seg_id_exists[point_cloud_idx])[0]
            else:
                assert(False)

            ###
            point_cloud_functions = self.functions[point_cloud_idx]
            b[i] = np.sum(point_cloud_functions[:, seg_id_subset], axis=-1)
            ###

            L[i] = self.one_hot_labels[point_cloud_idx]

        return P, b, L


    def __iter__(self):
        self.index_in_epoch = 0
        self.perm = np.arange(self.n_data)
        np.random.shuffle(self.perm)
        return self


    def next(self):
        self.start = self.index_in_epoch * self.batch_size

        # FIXME:
        # Fix this when input placeholders have dynamic sizes.
        #self.end = min(self.start + self.batch_size, self.n_data)
        self.end = self.start + self.batch_size

        self.step_size = self.end - self.start
        self.index_in_epoch = self.index_in_epoch + 1

        # FIXME:
        # Fix this when input placeholders have dynamic sizes.
        #if self.start < self.n_data:
        if self.end <= self.n_data:
            shuffled_indices = self.perm[self.start:self.end]
            step_P, step_b, step_L = self.generate_random_samples(
                    shuffled_indices)
            return step_P, step_b, step_L
        else:
            raise StopIteration()

