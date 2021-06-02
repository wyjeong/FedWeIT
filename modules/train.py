import pdb
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics as tf_metrics

from misc.utils import *

class TrainModule:
    """ Common module for model training 
    This module manages training procedures for both server and client
    Saves and loads all states whenever client is switched.
    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, args, logger, nets):
        self.args = args
        self.logger = logger
        self.nets = nets
        self.metrics = {
            'train_lss': tf_metrics.Mean(name='train_lss'),
            'train_acc': tf_metrics.CategoricalAccuracy(name='train_acc'),
            'valid_lss': tf_metrics.Mean(name='valid_lss'),
            'valid_acc': tf_metrics.CategoricalAccuracy(name='valid_acc'),
            'test_lss' : tf_metrics.Mean(name='test_lss'),
            'test_acc' : tf_metrics.CategoricalAccuracy(name='test_acc')
        }

    def init_state(self, cid):
        self.state = {
            'client_id': cid,
            'scores': {
                'test_loss': {},
                'test_acc': {},
            },
            'capacity': {
                'ratio': [],
                'num_shared_activ': [],
                'num_adapts_activ': [],
            },
            'communication': {
                'ratio': [],
                'num_actives': [],
            },
            'num_total_params': 0,
            'optimizer': []
        }
        self.init_learning_rate()

    def load_state(self, cid):
        self.state = np_load(os.path.join(self.args.state_dir, '{}_train.npy'.format(cid))).item()
        self.optimizer = tf.keras.optimizers.deserialize(self.state['optimizer'])

    def save_state(self):
        self.state['optimizer'] = tf.keras.optimizers.serialize(self.optimizer)
        np_save(self.args.state_dir, '{}_train.npy'.format(self.state['client_id']), self.state)

    def init_learning_rate(self):
        self.state['early_stop'] = False
        self.state['lowest_lss'] = np.inf
        self.state['curr_lr'] = self.args.lr
        self.state['curr_lr_patience'] = self.args.lr_patience
        self.init_optimizer(self.state['curr_lr'])

    def init_optimizer(self, curr_lr):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=curr_lr)

    def adaptive_lr_decay(self):
        vlss = self.vlss
        if vlss<self.state['lowest_lss']:
            self.state['lowest_lss'] = vlss
            self.state['curr_lr_patience'] = self.args.lr_patience
        else:
            self.state['curr_lr_patience']-=1
            if self.state['curr_lr_patience']<=0:
                prev = self.state['curr_lr']
                self.state['curr_lr']/=self.args.lr_factor
                self.logger.print(self.state['client_id'], 'epoch:%d, learning rate has been dropped from %.5f to %.5f' \
                                                    %(self.state['curr_epoch'], prev, self.state['curr_lr']))
                if self.state['curr_lr']<self.args.lr_min:
                    self.logger.print(self.state['client_id'], 'epoch:%d, early-stopped as minium lr reached to %.5f'%(self.state['curr_epoch'], self.state['curr_lr']))
                    self.state['early_stop'] = True
                self.state['curr_lr_patience'] = self.args.lr_patience
                self.optimizer.lr.assign(self.state['curr_lr'])

    def train_one_round(self, curr_round, round_cnt, curr_task):
        tf.keras.backend.set_learning_phase(1) 
        self.state['curr_round'] = curr_round
        self.state['round_cnt'] = round_cnt
        self.state['curr_task'] = curr_task
        self.curr_model = self.nets.get_model_by_tid(curr_task)
        for epoch in range(self.args.num_epochs):
            self.state['curr_epoch'] = epoch+1
            for i in range(0, len(self.task['x_train']), self.args.batch_size):
                x_batch = self.task['x_train'][i:i+self.args.batch_size]
                y_batch = self.task['y_train'][i:i+self.args.batch_size]
                with tf.GradientTape() as tape:
                    y_pred = self.curr_model(x_batch)
                    loss = self.params['loss'](y_batch, y_pred)
                gradients = tape.gradient(loss, self.params['trainables'])
                self.optimizer.apply_gradients(zip(gradients, self.params['trainables']))
            self.validate()
            self.evaluate()
            if self.args.model in ['fedweit']:
                self.calculate_capacity()
            self.adaptive_lr_decay()
            if self.state['early_stop']:
                continue

    def validate(self):
        tf.keras.backend.set_learning_phase(0)
        for i in range(0, len(self.task['x_valid']), self.args.batch_size):
            x_batch = self.task['x_valid'][i:i+self.args.batch_size]
            y_batch = self.task['y_valid'][i:i+self.args.batch_size]
            y_pred = self.curr_model(x_batch)
            loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred)
            self.add_performance('valid_lss', 'valid_acc', loss, y_batch, y_pred)
        self.vlss, self.vacc = self.measure_performance('valid_lss', 'valid_acc')

    def evaluate(self):
        tf.keras.backend.set_learning_phase(0)
        for tid in range(self.state['curr_task']+1):
            if self.args.model == 'stl':
                if not tid == self.state['curr_task']:
                    continue
            x_test = self.task['x_test_list'][tid]
            y_test = self.task['y_test_list'][tid]
            model = self.nets.get_model_by_tid(tid)
            for i in range(0, len(x_test), self.args.batch_size):
                x_batch = x_test[i:i+self.args.batch_size]
                y_batch = y_test[i:i+self.args.batch_size]
                y_pred = model(x_batch)
                loss = tf.keras.losses.categorical_crossentropy(y_batch, y_pred)
                self.add_performance('test_lss', 'test_acc', loss, y_batch, y_pred)
            lss, acc = self.measure_performance('test_lss', 'test_acc')
            if not tid in self.state['scores']['test_loss']:
                self.state['scores']['test_loss'][tid] = []
                self.state['scores']['test_acc'][tid] = []
            self.state['scores']['test_loss'][tid].append(lss)
            self.state['scores']['test_acc'][tid].append(acc)
            self.logger.print(self.state['client_id'], 'round:{}(cnt:{}),epoch:{},task:{},lss:{},acc:{} ({},#_train:{},#_valid:{},#_test:{})'
                .format(self.state['curr_round'], self.state['round_cnt'], self.state['curr_epoch'], tid, round(lss, 3), \
                    round(acc, 3), self.task['task_names'][tid], len(self.task['x_train']), len(self.task['x_valid']), len(x_test)))

    def add_performance(self, lss_name, acc_name, loss, y_true, y_pred):
        self.metrics[lss_name](loss)
        self.metrics[acc_name](y_true, y_pred)

    def measure_performance(self, lss_name, acc_name):
        lss = float(self.metrics[lss_name].result())
        acc = float(self.metrics[acc_name].result())
        self.metrics[lss_name].reset_states()
        self.metrics[acc_name].reset_states()
        return lss, acc

    def calculate_capacity(self):
        def l1_pruning(weights, hyp):
            hard_threshold = np.greater(np.abs(weights), hyp).astype(np.float32)
            return weights*hard_threshold

        if self.state['num_total_params'] == 0:
            for dims in self.nets.shapes:
                params = 1
                for d in dims:
                    params *= d
                self.state['num_total_params'] += params
        num_total_activ = 0
        num_shared_activ = 0
        num_adapts_activ = 0
        for var_name in self.nets.decomposed_variables:
            if var_name == 'adaptive':
                for tid in range(self.state['curr_task']+1):
                    for lid in self.nets.decomposed_variables[var_name][tid]:
                        var = self.nets.decomposed_variables[var_name][tid][lid]
                        var = l1_pruning(var.numpy(), self.args.lambda_l1)
                        actives = np.not_equal(var, np.zeros_like(var)).astype(np.float32)
                        actives = np.sum(actives)
                        num_adapts_activ += actives
            elif var_name == 'shared':
                for var in self.nets.decomposed_variables[var_name]:
                    actives = np.not_equal(var.numpy(), np.zeros_like(var)).astype(np.float32)
                    actives = np.sum(actives)
                    num_shared_activ += actives
            else:
                continue
        num_total_activ += (num_adapts_activ + num_shared_activ)
        ratio = num_total_activ/self.state['num_total_params']
        self.state['capacity']['num_adapts_activ'].append(num_adapts_activ)
        self.state['capacity']['num_shared_activ'].append(num_shared_activ)
        self.state['capacity']['ratio'].append(ratio)
        self.logger.print(self.state['client_id'], 'model capacity: %.3f' %(ratio))

    def calculate_communication_costs(self, params):
        if self.state['num_total_params'] == 0:
            for dims in self.nets.shapes:
                params = 1
                for d in dims:
                    params *= d
                self.state['num_total_params'] += params

        num_actives = 0
        for i, pruned in enumerate(params):
            actives = np.not_equal(pruned, np.zeros_like(pruned)).astype(np.float32)
            actives = np.sum(actives)
            num_actives += actives

        ratio = num_actives/self.state['num_total_params']
        self.state['communication']['num_actives'].append(num_actives)
        self.state['communication']['ratio'].append(ratio)
        self.logger.print(self.state['client_id'], 'communication cost: %.3f' %(ratio))

    def set_details(self, details):
        self.params = details

    def set_task(self, task):
        self.task = task

    def get_scores(self):
        return self.state['scores']

    def get_capacity(self):
        return self.state['capacity']

    def get_communication(self):
        return self.state['communication']

    def aggregate(self, updates):
        if self.args.sparse_comm and self.args.model in ['fedweit']:
            client_weights = [u[0][0] for u in updates]
            client_masks = [u[0][1] for u in updates]
            client_sizes = [u[1] for u in updates]
            new_weights = [np.zeros_like(w) for w in client_weights[0]]
            epsi = 1e-15
            total_sizes = epsi
            client_masks = tf.ragged.constant(client_masks, dtype=tf.float32)
            for _cs in client_masks:
                total_sizes += _cs
            for c_idx, c_weights in enumerate(client_weights): # by client
                for lidx, l_weights in enumerate(c_weights): # by layer
                    ratio = 1/total_sizes[lidx]
                    new_weights[lidx] += tf.math.multiply(l_weights, ratio).numpy()
        else:
            client_weights = [u[0] for u in updates]
            client_sizes = [u[1] for u in updates]
            new_weights = [np.zeros_like(w) for w in client_weights[0]]
            total_size = len(client_sizes)
            for c in range(len(client_weights)): # by client
                _client_weights = client_weights[c]
                for i in range(len(new_weights)): # by layer
                    new_weights[i] +=  _client_weights[i] * float(1/total_size)
        return new_weights

