import os
import pdb
import cv2
import argparse
import random
import torch
import torchvision
import numpy as np
import tensorflow as tf

# import sys
# sys.path.insert(0,'..')
from misc.utils import *
from third_party.mixture_loader.mixture import *

class DataGenerator:
    """ Data Generator
    Generating non_iid_50 task
    
    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, args):
        self.args = args
        self.seprate_ratio = (0.7, 0.2, 0.1) # train, test, valid
        self.mixture_dir = self.args.task_path
        self.mixture_filename = 'mixture.npy'
        self.base_dir = os.path.join(self.args.task_path, self.args.task) 
        self.did_to_dname = {
            0: 'cifar10',
            1: 'cifar100',
            2: 'mnist',
            3: 'svhn',
            4: 'fashion_mnist',
            5: 'traffic_sign',
            6: 'face_scrub',
            7: 'not_mnist',
        }
        self.generate_data()

    def generate_data(self):
        saved_mixture_filepath = os.path.join(self.mixture_dir, self.mixture_filename)
        if os.path.exists(saved_mixture_filepath):
            print('loading mixture data: {}'.format(saved_mixture_filepath))
            mixture = np.load(saved_mixture_filepath, allow_pickle=True)
        else:
            print('downloading & processing mixture data')
            mixture = get(base_dir=self.mixture_dir, fixed_order=True)
            np_save(self.mixture_dir, self.mixture_filename, mixture)
        self.generate_tasks(mixture)

    def generate_tasks(self, mixture):
        print('generating tasks ...')
        self.task_cnt = -1
        for dataset_id in self.args.datasets:
            self._generate_tasks(dataset_id, mixture[0][dataset_id])
    
    def _generate_tasks(self, dataset_id, data):
        # concat train & test
        x_train = data['train']['x']
        y_train = data['train']['y']
        x_test = data['test']['x']
        y_test = data['test']['y']
        x_valid = data['valid']['x']
        y_valid = data['valid']['y']
        x = np.concatenate([x_train, x_test, x_valid])
        y = np.concatenate([y_train, y_test, y_valid])

        # shuffle dataset
        idx_shuffled = np.arange(len(x))
        random_shuffle(self.args.seed, idx_shuffled)
        x = x[idx_shuffled]
        y = y[idx_shuffled]

        if self.args.task == 'non_iid_50':
            self._generate_non_iid_50(dataset_id, x, y)

    def _generate_non_iid_50(self, dataset_id, x, y):
        labels = np.unique(y)
        random_shuffle(self.args.seed, labels)
        labels_per_task = [labels[i:i+self.args.num_classes] for i in range(0, len(labels), self.args.num_classes)]
        for task_id, _labels in enumerate(labels_per_task):
            if dataset_id == 5 and task_id == 8:
                continue
            elif dataset_id in [1,6] and task_id > 15:
                continue
            self.task_cnt += 1
            idx = np.concatenate([np.where(y[:]==c)[0] for c in _labels], axis=0)
            random_shuffle(self.args.seed, idx)
            x_task = x[idx]
            y_task = y[idx]

            idx_labels = [np.where(y_task[:]==c)[0] for c in _labels]
            for i, idx_label in enumerate(idx_labels):
                y_task[idx_label] = i # reset class_id 
            y_task = tf.keras.utils.to_categorical(y_task, len(_labels))
            
            filename = '{}_{}'.format(self.did_to_dname[dataset_id], task_id)
            self._save_task(x_task, y_task, _labels, filename, dataset_id)

    def _save_task(self, x_task, y_task, labels, filename, dataset_id):
        # pairs = list(zip(x_task, y_task))
        num_examples = len(x_task)
        num_train = int(num_examples*self.seprate_ratio[0]) 
        num_test = int(num_examples*self.seprate_ratio[1]) 
        num_valid = num_examples - num_train - num_test
        train_name = filename+'_train'
        save_task(base_dir=self.base_dir, filename=train_name, data={
            'x_train': x_task[:num_train],
            'y_train': y_task[:num_train],
            'labels': labels,
            'name': train_name,
            'dataset_id': dataset_id
        })
        valid_name = filename+'_valid'
        save_task(base_dir=self.base_dir, filename=valid_name, data={
            'x_valid': x_task[num_train+num_test:],
            'y_valid': y_task[num_train+num_test:],
            'dataset_id': dataset_id
        })
        test_name = filename+'_test'
        save_task(base_dir=self.base_dir, filename=test_name, data={
            'x_test' : x_task[num_train:num_train+num_test],
            'y_test' : y_task[num_train:num_train+num_test],
            'dataset_id': dataset_id
        })

        print('filename:{}, labels:[{}], num_train:{}, num_valid:{}, num_test:{}'.format(filename,', '.join(map(str, labels)), num_train, num_valid, num_test))

    
