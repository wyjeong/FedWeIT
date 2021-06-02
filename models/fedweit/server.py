import pdb
import sys
import time
import random
import threading
import tensorflow as tf

from misc.utils import *
from .client import Client
from modules.federated import ServerModule

class Server(ServerModule):
    """ FedWeIT Server
    Performing fedweit server algorithms 
    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, args):
        super(Server, self).__init__(args, Client)
        self.client_adapts = []

    def train_clients(self):
        cids = np.arange(self.args.num_clients).tolist()
        num_selection = int(round(self.args.num_clients*self.args.frac_clients))
        for curr_round in range(self.args.num_rounds*self.args.num_tasks):
            self.updates = []
            self.curr_round = curr_round+1
            self.is_last_round = self.curr_round%self.args.num_rounds==0
            if self.is_last_round:
                self.client_adapts = []
            selected_ids = random.sample(cids, num_selection) # pick clients
            self.logger.print('server', 'round:{} train clients (selected_ids: {})'.format(curr_round, selected_ids))
            # train selected clients in parallel
            for clients in self.parallel_clients:
                self.threads = []
                for gid, cid in enumerate(clients):
                    client = self.clients[gid]
                    selected = True if cid in selected_ids else False
                    with tf.device('/device:GPU:{}'.format(gid)):
                        thrd = threading.Thread(target=self.invoke_client, args=(client, cid, curr_round, selected, self.get_weights(), self.get_adapts()))
                        self.threads.append(thrd)
                        thrd.start()
                # wait all threads each round
                for thrd in self.threads:
                    thrd.join()
            # update
            aggr = self.train.aggregate(self.updates)
            self.set_weights(aggr)
        self.logger.print('server', 'done. ({}s)'.format(time.time()-self.start_time))
        sys.exit()

    def invoke_client(self, client, cid, curr_round, selected, weights, adapts):
        update = client.train_one_round(cid, curr_round, selected, weights, adapts)
        if not update == None:
            self.updates.append(update)
            if self.is_last_round:
                self.client_adapts.append(client.get_adaptives())

    def get_adapts(self):
        if self.curr_round%self.args.num_rounds==1 and not self.curr_round==1:
            from_kb = []
            for lid, shape in enumerate(self.nets.shapes):
                shape = np.concatenate([self.nets.shapes[lid],[int(round(self.args.num_clients*self.args.frac_clients))]], axis=0)
                from_kb_l = np.zeros(shape)
                for cid, ca in enumerate(self.client_adapts):
                    try:
                        if len(shape)==5:
                            from_kb_l[:,:,:,:,cid] = ca[lid]
                        else:
                            from_kb_l[:,:,cid] = ca[lid]
                    except:
                        pdb.set_trace()           
                from_kb.append(from_kb_l)
            return from_kb
        else:
            return None
        