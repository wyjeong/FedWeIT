import pdb
import math
import random
import tensorflow as tf

from misc.utils import *
from modules.federated import ClientModule

class Client(ClientModule):
    """ FedWeIT Client
    Performing fedweit cleint algorithms 
    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, gid, args, initial_weights):
        super(Client, self).__init__(gid, args, initial_weights)
        self.state['gpu_id'] = gid

    def train_one_round(self, client_id, curr_round, selected, global_weights=None, from_kb=None):
        ######################################
        self.switch_state(client_id)
        ######################################
        self.state['round_cnt'] += 1
        self.state['curr_round'] = curr_round
        
        if not from_kb == None:
            for lid, weights in enumerate(from_kb):
                tid = self.state['curr_task']+1
                self.nets.decomposed_variables['from_kb'][tid][lid].assign(weights)
        
        if self.state['curr_task']<0:
            self.init_new_task()
            self.set_weights(global_weights) 
        else:
            is_last_task = (self.state['curr_task']==self.args.num_tasks-1)
            is_last_round = (self.state['round_cnt']%self.args.num_rounds==0 and self.state['round_cnt']!=0)
            is_last = is_last_task and is_last_round
            if is_last_round:
                if is_last_task:
                    if self.train.state['early_stop']:
                        self.train.evaluate()
                    self.stop()
                    return
                else:
                    self.init_new_task()
                    self.state['prev_body_weights'] = self.nets.get_body_weights(self.state['curr_task'])
            else:
                self.load_data()

        if selected:
            self.set_weights(global_weights)

        with tf.device('/device:GPU:{}'.format(self.state['gpu_id'])):
            self.train.train_one_round(self.state['curr_round'], self.state['round_cnt'], self.state['curr_task'])
        
        self.logger.save_current_state(self.state['client_id'], {
            'scores': self.train.get_scores(),
            'capacity': self.train.get_capacity(),
            'communication': self.train.get_communication()
        })
        self.save_state()
        
        if selected:
            return self.get_weights(), self.get_train_size()

    def loss(self, y_true, y_pred):
        weight_decay, sparseness, approx_loss = 0, 0, 0
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        for lid in range(len(self.nets.shapes)):
            sw = self.nets.get_variable(var_type='shared', lid=lid)
            aw = self.nets.get_variable(var_type='adaptive', lid=lid, tid=self.state['curr_task'])
            mask = self.nets.get_variable(var_type='mask', lid=lid, tid=self.state['curr_task'])
            g_mask = self.nets.generate_mask(mask)
            weight_decay += self.args.wd * tf.nn.l2_loss(aw)
            weight_decay += self.args.wd * tf.nn.l2_loss(mask)
            sparseness += self.args.lambda_l1 * tf.reduce_sum(tf.abs(aw))
            sparseness += self.args.lambda_mask * tf.reduce_sum(tf.abs(mask))
            if self.state['curr_task'] == 0:
                weight_decay += self.args.wd * tf.nn.l2_loss(sw)
            else:
                for tid in range(self.state['curr_task']):
                    prev_aw = self.nets.get_variable(var_type='adaptive', lid=lid, tid=tid)
                    prev_mask = self.nets.get_variable(var_type='mask', lid=lid, tid=tid)
                    g_prev_mask = self.nets.generate_mask(prev_mask)
                    #################################################
                    restored = sw * g_prev_mask + prev_aw
                    a_l2 = tf.nn.l2_loss(restored-self.state['prev_body_weights'][lid][tid])
                    approx_loss += self.args.lambda_l2 * a_l2
                    #################################################
                    sparseness += self.args.lambda_l1 * tf.reduce_sum(tf.abs(prev_aw))
        
        loss += weight_decay + sparseness + approx_loss 
        return loss

    def get_adaptives(self):
        adapts = []
        for lid in range(len(self.nets.shapes)):
            aw = self.nets.get_variable(var_type='adaptive', lid=lid, tid=self.state['curr_task']).numpy()
            hard_threshold = np.greater(np.abs(aw), self.args.lambda_l1).astype(np.float32)
            adapts.append(aw*hard_threshold)
        return adapts
