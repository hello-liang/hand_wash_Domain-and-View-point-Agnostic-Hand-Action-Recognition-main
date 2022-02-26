#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:19:18 2021

@author: asabater
"""

import os

import os
import pickle
import numpy as np
from tqdm import tqdm
import time
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from data_generator import DataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
knn_neighbors = [7]
# aug_loop = [0,10,20,40]
#aug_loop = [0,40]
aug_loop = [0]
num_augmentations = max(aug_loop)
weights = 'distance'
crop_train = 80000
crop_test = np.inf
np.random.seed(0)
from joblib import Parallel, delayed

def print_results(dataset_name, total_res, knn_neighbors, aug_loop, frame=True):
    if frame: print('-'*81)
    print('# | {} | {}'.format(dataset_name,
            ' | '.join([ '[{}] {:.1f} / {:.1f}'.format(k, max([ total_res[0][k][n] for n in knn_neighbors ])*100,
              max([ total_res[na][k][n] for n in knn_neighbors for na in aug_loop ])*100) for k in total_res[0].keys() ])
        ))
    if frame: print('-'*81)
    

if __name__ == '__main__':
    # %%

    # =============================================================================
    # Load model
    # =============================================================================
    
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import prediction_utils
    import time
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate action recognition in different datasets')
    parser.add_argument('--path_model', type=str, help='path to the trained model',default="./pretrained_models/xdom_summarization")
    parser.add_argument('--loss_name', type=str, help='key to load weights',default="mixknn_best")
    parser.add_argument('--eval_fphab', action='store_true', help='evaluate on F-PHAB splits')
    parser.add_argument('--eval_shrec', action='store_true', help='evaluate on SHREC splits')
    parser.add_argument('--eval_msra', action='store_true', help='evaluate on MSRA dataset')
    args = parser.parse_args()
    model, model_params = prediction_utils.load_model(args.path_model, False, loss_name = args.loss_name)
    model_params['use_rotations'] = None
    print('* Model loaded')
    


t = time.time()
model_params['skip_frames'] = [1] #this one is only skip one image ,default is skip 3 image.

n_splits=4
seq_perc=-1
data_format=model_params['joints_format']
#from dataset_scripts.MSRA import load_data#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from dataset_scripts.wash_hand_kaggle import load_data

np.random.seed(0)
if seq_perc == -1: total_data = load_data.actions_to_samples(load_data.load_data(data_format), -1)
actions_list, actions_labels, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet = load_data.get_folds(total_data, n_splits=n_splits)
total_annotations=actions_list
num_augmentations
model_params
load_from_files=False
return_sequences=True
np.random.seed(0)
data_gen = DataGenerator(**model_params)
if return_sequences: data_gen.max_seq_len = 0
if load_from_files: skels_ann = [ data_gen.load_skel_coords(ann) for ann in total_annotations ]
else: skels_ann = total_annotations
test_i=0
def get_pose_features(validation=False):
    print(len(skels_ann))
    action_sequences =[]
    for i in range(len(skels_ann)):
        skels=skels_ann[i]
        action_sequences.append(data_gen.get_pose_data_v2(skels, validation=validation))
    if not return_sequences: action_sequences = pad_sequences(action_sequences, abs(model_params['max_seq_len']), dtype='float32', padding='pre')
    return action_sequences

action_sequences = get_pose_features(validation=True)
print('* Data sequences loaded')

action_sequences_augmented = None
return_sequences=True
t = time.time()
model.set_encoder_return_sequences(return_sequences)
# Get embeddings from all annotations
if return_sequences: 
    # embs = np.array([ model.get_embedding(s[None]).numpy()[0] for s in action_sequences ])
    embs = [ model.get_embedding(s[None]) for s in action_sequences ]

print('* Embeddings calculated')

embs_aug = None

tf = time.time()
sys.stdout.flush

if return_sequences: embs = np.array([ e[0] for e in embs ])
if num_augmentations > 0:
    if return_sequences: embs_aug = np.array([ [ s[0] for s in samples ] for samples in embs_aug ])
    else: embs_aug = [ np.concatenate([ s for s in samples ]) for samples in embs_aug ]
num_sequences = len(embs)
if num_augmentations > 0: num_sequences += sum([ len(e) for e in embs_aug ])
print(' ** Prediction time **   Secuences evaluated [{}] | time [{:.3f}s] | ms per sequence [{:.3f}]'.format(num_sequences, tf-t, (tf-t)*1000/num_sequences))
return_sequences=True
total_res = {}
n_aug =0
print('***', n_aug, '***')
total_res[n_aug] = {}
print(n_aug, 'posturenet_online')
folds_data=folds_posturenet
total_labels=actions_labels
num_augmentations=n_aug
embs_aug=embs_aug
leave_one_out=False
evaluate_all_folds = False
groupby=None
return_sequences=return_sequences
res = {}
num_folds = len(folds_data) if evaluate_all_folds else 1
num_fold =0
if leave_one_out:
    train_indexes =  np.concatenate([ f['indexes'] for i,f in folds_data.items() if i!= num_fold])
    test_indexes = folds_data[num_fold]['indexes']
else:
    train_indexes = folds_data[num_fold]['indexes']
    test_indexes =  np.concatenate([ f['indexes'] for i,f in folds_data.items() if i!= num_fold])


X_train = embs[train_indexes]
X_test = embs[test_indexes]
y_train = total_labels[train_indexes]
y_test = total_labels[test_indexes]




if return_sequences:
    y_train = [ y for y,seq in zip(y_train, X_train) for _ in range(len(seq)) ]
    y_test = [ y for y,seq in zip(y_test, X_test) for _ in range(len(seq)) ]
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)





if len(y_train) > crop_train:
    idx = np.random.choice(np.arange(len(y_train)), crop_train, replace=False)
    X_train = X_train[idx]
    y_train = np.array(y_train)[idx]

res[num_fold] = {}
knn = KNeighborsClassifier(n_neighbors=1, n_jobs=8, weights=weights).fit(X_train, y_train)
for n in knn_neighbors:
    knn = knn.set_params(**{'n_neighbors': n})
    t = time.time()
    preds = knn.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(acc)
    res[num_fold][n] = acc
    tf = time.time()
    print(' ** Classification time ** num_fold [{}] | k [{}] | X_test [{}] | time [{:.3f}s] | ms per sequence [{:.3f}]'.format(num_fold, n,
                                                               len(X_test), tf-t, (tf-t)*1000/len(X_test)))




res = { n:np.mean([ res[num_fold][n] for num_fold in range(num_folds) ]) for n in knn_neighbors }
total_res[n_aug]['posturenet_online'] = res
total_res_msra_full = total_res
print_results('MSRA     ', total_res_msra_full, knn_neighbors, aug_loop)  ## | MSRA      | [posturenet] 97.1 / 97.1 | [posturenet_online] 85.8 / 86.6
print('Time elapsed: {:.2f}'.format((time.time()-t)/60))
del embs; del embs_aug; del action_sequences; del action_sequences_augmented;



    
# %%





