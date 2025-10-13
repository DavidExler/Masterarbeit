import numpy as np
import os
import pickle

#
# takes masks, images (Z,X,Y,C)?, label_store.json
# returns best_combo, predicted, checkpoints
#

from helpers.optimize import train
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def train_combo(in_folder, out_folder):

    path = os.path.join(BASE_DIR, 'data', in_folder, 'best_combo.pkl')
    with open(path, 'rb') as f:
        (enc, dec, pre, pretrain) = pickle.load(f) 
    

    best_acc = train(enc, dec, pre, pretrain, in_folder, out_folder)
    return best_acc
    
    