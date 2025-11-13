#!/usr/bin/python2.7

import torch
import numpy as np
import random
import pandas as pd
import os
from data_utils import *


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, target_fps=10):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.features_path = features_path
        self.target_fps = target_fps
        self.gts_segments = load_temporal_labels(gt_path)
        self.features_cache = {}
        self.gts_cache = {}

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        # read splits
        df = pd.read_csv(vid_list_file)
        first_column_name = df.columns[0]
        list_of_examples = df[first_column_name].tolist()

        self.list_of_examples = [i for i in list_of_examples if i not in excluded_samples]
        print("Discarded {} samples".format(len(excluded_samples)))
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            # Load features
            if vid in self.features_cache:
                features = self.features_cache[vid]
                gt = self.gts_cache[vid]
            else:
                tmp = vid.split('/')
                feature_path = '/'.join([tmp[0], 'features', 'i3d'] + tmp[1:]) + '.h5'
                feature_path = os.path.join(self.features_path, feature_path)
                try:
                    features = load_feature(feature_path)
                except:
                    raise ValueError("Error loading feature file: {}".format(feature_path))
                fps = get_fps(vid)
                features = subsample(features, original_fps=fps, target_fps=self.target_fps)

                # Prepare labels
                num_frames = features.shape[0]
                gt_segments = self.gts_segments[vid]
                gt = convert_segments_to_segmentations(gt_segments, num_frames, fps=self.target_fps, default_class_id=9)

                # This codebase expects features to have shape [D, T]
                features = np.transpose(features)
                self.features_cache[vid] = features
                self.gts_cache[vid] = gt

            batch_input.append(features)
            batch_target.append(gt)

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
