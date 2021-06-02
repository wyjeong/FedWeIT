import os
import pdb
import json
import random
import threading
import numpy as np

from datetime import datetime

def random_shuffle(seed, _list):
    random.seed(seed)
    random.shuffle(_list)

def random_sample(seed, _list, num_pick):
    random.seed(seed)
    return random.sample(_list, num_pick)

def random_int(seed, start, end):
    random.seed(seed)
    random.randint(start, end)

def np_save(base_dir, filename, data):
    if os.path.isdir(base_dir) == False:
        os.makedirs(base_dir)
    np.save(os.path.join(base_dir, filename), data)

def save_task(base_dir, filename, data):    
    np_save(base_dir, filename, data)

def save_weights(base_dir, filename, weights):
    np_save(base_dir, filename, weights)

def write_file(filepath, filename, data):
    if os.path.isdir(filepath) == False:
        os.makedirs(filepath)
    with open(os.path.join(filepath, filename), 'w+') as outfile:
        json.dump(data, outfile)

def np_load(path):
    loaded = np.load(path, allow_pickle=True)
    return loaded

def load_task(base_dir, task):
    loaded = np_load(os.path.join(base_dir, task))
    return loaded

def load_weights(path):
    return np_load(path)

def debugger():
    pdb.set_trace()

    
