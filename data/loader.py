import os
import pdb
import glob
import numpy as np

from misc.utils import *

class DataLoader:
    """ Data Loader
        
    Loading data for the corresponding clients
    
    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, args):
        self.args = args
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

    def init_state(self, cid):
        self.state = {
            'client_id': cid,
            'tasks': []
        }
        self.load_tasks(cid)

    def load_state(self, cid):
        self.state = np_load(os.path.join(self.args.state_dir, '{}_data.npy'.format(cid))).item()

    def save_state(self):
        np_save(self.args.state_dir, '{}_data'.format(self.state['client_id']), self.state)

    def load_tasks(self, cid):
        if self.args.task in ['non_iid_50']:
            task_set = {
                0: ['cifar100_5', 'cifar100_13', 'face_scrub_0', 'cifar100_14', 'svhn_1', 'traffic_sign_0', 'not_mnist_1', 'cifar100_8', 'face_scrub_13', 'cifar100_4'],
                1: ['cifar100_2', 'traffic_sign_5', 'face_scrub_14', 'traffic_sign_4', 'not_mnist_0', 'mnist_0', 'face_scrub_2', 'face_scrub_15', 'cifar100_1', 'fashion_mnist_1'],
                2: ['face_scrub_11', 'svhn_0', 'face_scrub_10', 'face_scrub_6', 'face_scrub_7', 'cifar100_3', 'cifar100_10', 'mnist_1', 'face_scrub_1', 'traffic_sign_1'],
                3: ['fashion_mnist_0', 'cifar100_15', 'face_scrub_3', 'cifar10_1', 'cifar100_7', 'face_scrub_8', 'cifar10_0', 'face_scrub_9', 'cifar100_0', 'cifar100_6'],
                4: ['traffic_sign_7', 'face_scrub_5', 'traffic_sign_6', 'traffic_sign_3', 'traffic_sign_2','cifar100_12', 'cifar100_11', 'cifar100_9', 'face_scrub_12', 'face_scrub_4']
            }
            self.state['tasks'] = task_set[self.state['client_id']]
        
        else:
            print('no correct task was given: {}'.format(self.args.task))
            os._exit(0)

    def get_train(self, task_id):
        return load_task(self.base_dir, self.state['tasks'][task_id]+'_train.npy').item()
    

    def get_valid(self, task_id):
        valid = load_task(self.base_dir, self.state['tasks'][task_id]+'_valid.npy').item()
        return valid['x_valid'], valid['y_valid']

    def get_test(self, task_id):
        x_test_list = []
        y_test_list = []
        for tid, task in enumerate(self.state['tasks']):
            if tid <= task_id:
                test = load_task(self.base_dir, task+'_test.npy').item()
                x_test_list.append(test['x_test'])
                y_test_list.append(test['y_test'])
        return x_test_list, y_test_list

    
