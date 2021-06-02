import os
import sys
import pdb
import copy
import time
import math
import random
import threading
import atexit
import tensorflow as tf

from misc.utils import *
from misc.logger import Logger
from data.loader import DataLoader
from modules.nets import NetModule
from modules.train import TrainModule

class ServerModule:
    """ Superclass for Server Module
    This module contains common server functions, 
    such as laoding data, training global model, handling clients, etc.
    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, args, ClientObj):
        self.args = args
        self.clients = {}
        self.threads = []
        self.ClientObj = ClientObj
        self.limit_gpu_memory()
        self.logger = Logger(self.args)
        self.nets = NetModule(self.args)
        self.train = TrainModule(self.args, self.logger, self.nets)
        
        self.nets.init_state(None)
        self.train.init_state(None)
        atexit.register(self.atexit)

    def limit_gpu_memory(self):
        self.gpu_ids = np.arange(len(self.args.gpu.split(','))).tolist()
        self.gpus = tf.config.list_physical_devices('GPU')
        if len(self.gpus)>0:
            for i, gpu_id in enumerate(self.gpu_ids):
                gpu = self.gpus[gpu_id]
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(gpu, 
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*self.args.gpu_mem_multiplier)])

    def run(self):
        self.logger.print('server', 'started')
        self.start_time = time.time()
        self.init_global_weights()
        self.init_clients()
        self.train_clients()

    def init_global_weights(self):
        self.global_weights = self.nets.init_global_weights()

    def init_clients(self):
        opt_copied = copy.deepcopy(self.args)
        num_gpus = len(self.gpu_ids)
        num_iteration = self.args.num_clients//num_gpus
        residuals = self.args.num_clients%num_gpus

        offset = 0
        self.parallel_clients = []
        for i in range(num_iteration):
            offset = i*num_gpus
            self.parallel_clients.append(np.arange(num_gpus)+offset)
        if residuals>0:
            offset = self.parallel_clients[-1][-1]+1
            self.parallel_clients.append(np.arange(residuals)+offset)

        initial_weights = self.global_weights
        if len(self.gpus)>0:
            for i, gpu_id in enumerate(self.gpu_ids):
                gpu = self.gpus[gpu_id]
                with tf.device('/device:GPU:{}'.format(gpu_id)):
                    self.clients[gpu_id] =self.ClientObj(gpu_id, opt_copied, initial_weights)
        else:
            num_parallel = 5
            self.clients = {i:self.ClientObj(i, opt_copied, initial_weights) for i in range(num_parallel)}

    def get_weights(self):
        return self.global_weights

    def set_weights(self, weights):
        self.global_weights = weights

    def atexit(self):
        for thrd in self.threads:
            thrd.join()
        self.logger.print('server', 'all client threads have been destroyed.' )


class ClientModule:
    """ Superclass for Client Module 
    This module contains common client functions, 
    such as loading data, training local model, switching states, etc.
    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, gid, args, initial_weights):
        self.args = args
        self.state = {'gpu_id': gid}
        self.lock = threading.Lock()
        self.logger = Logger(self.args)
        self.loader = DataLoader(self.args)

        self.nets = NetModule(self.args)
        self.train = TrainModule(self.args, self.logger, self.nets)
        
        self.init_model(initial_weights)

    def init_model(self, initial_weights):
        decomposed = True if self.args.model in ['fedweit'] else False
        if self.args.base_network == 'lenet':
            self.nets.build_lenet(initial_weights, decomposed=decomposed)

    def switch_state(self, client_id):
        if self.is_new(client_id):
            self.loader.init_state(client_id)
            self.nets.init_state(client_id)
            self.train.init_state(client_id)
            self.init_state(client_id)
        else: # load_state
            self.loader.load_state(client_id)
            self.nets.load_state(client_id)
            self.train.load_state(client_id)
            self.load_state(client_id)

    def is_new(self, client_id):
        return not os.path.exists(os.path.join(self.args.state_dir, f'{client_id}_client.npy'))

    def init_state(self, cid):
        self.state['client_id'] = cid
        self.state['task_names'] = {}
        self.state['curr_task'] =  -1
        self.state['round_cnt'] = 0
        self.state['done'] = False

    def load_state(self, cid):
        self.state = np_load(os.path.join(self.args.state_dir, '{}_client.npy'.format(cid))).item()
        self.update_train_config_by_tid(self.state['curr_task'])

    def save_state(self):
        np_save(self.args.state_dir, '{}_client.npy'.format(self.state['client_id']), self.state)
        self.loader.save_state()
        self.nets.save_state()
        self.train.save_state()

    def init_new_task(self):
        self.state['curr_task'] += 1
        self.state['round_cnt'] = 0
        self.load_data()
        self.train.init_learning_rate()
        self.update_train_config_by_tid(self.state['curr_task'])

    def update_train_config_by_tid(self, tid):
        self.target_model = self.nets.get_model_by_tid(tid)
        self.trainable_variables = self.nets.get_trainable_variables(tid)
        self.trainable_variables_body = self.nets.get_trainable_variables(tid, head=False)
        self.train.set_details({
            'loss': self.loss,
            'model': self.target_model,
            'trainables': self.trainable_variables,
        })

    def load_data(self):
        data = self.loader.get_train(self.state['curr_task'])
        self.state['task_names'][self.state['curr_task']] = data['name']
        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_valid, self.y_valid = self.loader.get_valid(self.state['curr_task'])
        self.x_test_list, self.y_test_list = self.loader.get_test(self.state['curr_task'])
        self.train.set_task({
            'x_train': self.x_train,
            'y_train': self.y_train,
            'x_valid': self.x_valid,
            'y_valid': self.y_valid,
            'x_test_list': self.x_test_list,
            'y_test_list': self.y_test_list,
            'task_names': self.state['task_names'],
        })

    def get_model_by_tid(self, tid):
        return self.nets.get_model_by_tid(tid)

    def set_weights(self, weights):
        if self.args.model in ['fedweit']:
            if weights is None:
                return None
            for i, w in enumerate(weights):
                sw = self.nets.get_variable('shared', i)
                residuals = tf.cast(tf.equal(w, tf.zeros_like(w)), dtype=tf.float32)
                sw.assign(sw*residuals+w)
        else:
            self.nets.set_body_weights(weights)

    def get_weights(self):
        if self.args.model in ['fedweit']:
            if self.args.sparse_comm:
                hard_threshold = []
                sw_pruned = []
                masks = self.nets.decomposed_variables['mask'][self.state['curr_task']]
                for lid, sw in enumerate(self.nets.decomposed_variables['shared']):
                    mask = masks[lid]
                    m_sorted = tf.sort(tf.keras.backend.flatten(tf.abs(mask)))
                    thres = m_sorted[math.floor(len(m_sorted)*(self.args.client_sparsity))]
                    m_bianary = tf.cast(tf.greater(tf.abs(mask), thres), tf.float32).numpy().tolist()
                    hard_threshold.append(m_bianary)
                    sw_pruned.append(sw.numpy()*m_bianary)
                self.train.calculate_communication_costs(sw_pruned)
                return sw_pruned, hard_threshold
            else:
                return [sw.numpy() for sw in self.nets.decomposed_variables['shared']]
        else:
            return self.nets.get_body_weights()

    def get_train_size(self):
        return len(self.x_train)

    def get_task_id(self):
        return self.curr_task

    def stop(self):
        self.done = True
