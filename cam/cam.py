# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:52:42 2019

@author: Insung baek, yscho
"""

# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np
import pandas as pd
import json # import json module
import time
import functools

import torch
import torch.nn as nn

import torch.nn.functional as F

#sys.path.append('D:/PROJECT/2019-ETRI-STARCRAFT/code/Code_HG/baselines/')
sys.path.append('D:/1.Project/2019.04_Game AI/Code_HG/')
start_dir = 'D:/1.Project/2019.04_Game AI/Code_HG/'
os.chdir(start_dir)
del(start_dir)

from models.resnet import ResNet3D_non_local
from features.custom_features import SPATIAL_FEATURES
from layers.embeddings import CategoricalEmbedding

import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt

###### 1. Model load ######
model_path = 'D:/1.Project/2019.04_Game AI/Code_HG/best_model/ResNet3D_non_local_025_0.973_0.900.pt'
INCLUDE = ['visibility_map', 'player_relative', 'unit_type']
num_classes = 2
model_configs = {
    'num_classes': num_classes,
    'include': INCLUDE
}
model = ResNet3D_non_local(**model_configs)
# model.embeddings.unit_type = CategoricalEmbedding(category_size=1914, embedding_dim=100, name='unit_type')
model.load_state_dict(torch.load(model_path))
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

criterion = criterion.to(device)
model = model.to(device)

###### 2. Data load ######
PROJECT_DIR = 'D:/1.Project/2019.04_Game AI/Code_HG/'
REPLAY_DIR = os.path.join(PROJECT_DIR, 'parsed/TvP/cam/')
RESULT_DIR = os.path.join(PROJECT_DIR, 'paper_result/cam_result/')
REPLAY_LIST = os.listdir(REPLAY_DIR)

assert all([os.path.isdir(PROJECT_DIR), os.path.isdir(REPLAY_DIR)])

def load_features(replay_dir, feature_type='spatial'):
    """Load parsed features from .npz file format."""
    if feature_type == 'screen':
        filepath = os.path.join(replay_dir, 'ScreenFeatures.npz')
    elif feature_type == 'minimap':
        filepath = os.path.join(replay_dir, 'MinimapFeatures.npz')
    elif feature_type == 'spatial':
        filepath = os.path.join(replay_dir, 'SpatialFeatures.npz')
    elif feature_type == 'flat':
        filepath = os.path.join(replay_dir, 'FlatFeatures.npz')
        raise NotImplementedError
    else:
        raise ValueError
    
    with np.load(filepath) as fp:
        name2feature = {k: v for k, v in fp.items()}
    
    return name2feature

def human_readable_size(size, precision=2):
    suffixes = ['B','KB','MB','GB','TB']
    suffix_idx = 0
    while size > 1024 and suffix_idx < 4:
        suffix_idx += 1     # increment the index of the suffix
        size = size / 1024.0  # apply the division
    return "%.*f%s" % (precision, size, suffixes[suffix_idx])

# data 생성
data_X = []
data_Y = []
timestep_all = []

for j in range(len(REPLAY_LIST)):
    REPLAY_DIR_read = REPLAY_DIR + REPLAY_LIST[j]

    # Load spatial features (minimap features + 'unit_type' channel)
    spatial_specs = SPATIAL_FEATURES._asdict()
    spatial_features = load_features(REPLAY_DIR_read, 'spatial')
        
    timesteps = spatial_features['visibility_map'].shape[0]

    max_timesteps = 50
    
    weight_fn = lambda x: np.log(1 + x)        
    weights = [weight_fn(i) for i in range(0, int(timesteps))] 
    weights /= np.sum(weights)
                
    timestep_indices = np.random.choice(timesteps, max_timesteps, replace=False, p=weights)
    timestep_indices = sorted(list(timestep_indices), reverse=False)

    get_vm = spatial_features['visibility_map'][timestep_indices,:,:]
    get_pr = spatial_features['player_relative'][timestep_indices,:,:]
    get_ut = spatial_features['unit_type'][timestep_indices,:,:]

    get_ut[get_ut > 1913] = 0

    get_featuremap = np.stack([get_vm, get_pr, get_ut])
    data_X.append(get_featuremap)    

    REPLAY_DIR_read_label = REPLAY_DIR + REPLAY_LIST[j] + '\\PlayerMetaInfo.json'
    
    with open(REPLAY_DIR_read_label) as json_file:
        json_data = json.load(json_file)
    
    json_data = pd.DataFrame(json_data)
    idx = np.where(json_data.T[['race', 'result']]['race']=='Terran')[0][0]
    
    tmp_label = json_data.T['result'][idx]
    if tmp_label == 'Defeat':
        tmp_y = 0
    else:
        tmp_y = 1
    data_Y.append(tmp_y)

    os.makedirs(RESULT_DIR+REPLAY_LIST[j], exist_ok=True)
    timestep_indices.append(timesteps)
    pd.DataFrame(timestep_indices, columns=[REPLAY_LIST[j]]).to_csv(RESULT_DIR+REPLAY_LIST[j]+'/time_step.csv')
    timestep_all.append(timestep_indices)

X = np.stack(data_X)
Y = np.stack(np.array(data_Y))

X_tensor = torch.from_numpy(X).to(device)
Y_tensor = torch.from_numpy(Y).to(device)

num_batch = 2
X_dict = {'visibility_map':X_tensor[0:num_batch,0,:,:], 
          'player_relative':X_tensor[0:num_batch,1,:,:], 
          'unit_type':X_tensor[0:num_batch,2,:,:]}

print('X_tensor.shape: ', X_tensor.shape)
print('Y_tensor.shape: ', Y_tensor.shape)

###### 3. class activation mapping ######
# exctract last conv layer
class SaveFeatures():
    """Extract pretrained activations"""
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn) # forward gradiets
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

final_layer = model._modules.get('block3')

activated_features = SaveFeatures(final_layer)

# probability of 2 class
prediction = model(X_dict)
pred_probabilities = F.softmax(prediction).data.squeeze()
activated_features.remove()
torch.topk(pred_probabilities,1)

weight_softmax_params = list(model._modules.get('linear').parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
class_idx = torch.topk(pred_probabilities,1)[1].int()

for obs in range(len(REPLAY_LIST)):
    #obs=0
    tmp_fm = activated_features.features[obs]
    # tmp_w = weight_softmax[obs]
    tmp_class = class_idx[obs]
        
    t, nc, h, w = tmp_fm.shape
    featuremap_weighted = []
    for c in range(len(weight_softmax[tmp_class])):
        #c=0
        tmp_fm_weighted = weight_softmax[tmp_class][c]*(tmp_fm[c])
        featuremap_weighted.append(tmp_fm_weighted)
    featuremap_weighted_array = np.stack(featuremap_weighted)
    
    t_cam = []
    for f in range(featuremap_weighted_array.shape[0]):
        #f=0
        tmp_cam = np.zeros((4,4))
        for ch in range(featuremap_weighted_array[f].shape[0]):
            #ch=0
            tmp_tmp_cam = featuremap_weighted_array[f][ch]
            tmp_cam += tmp_tmp_cam
        t_cam.append(tmp_cam)
    t_cam[40].shape
    
    # sum for each node5(0~4)
    sum_cam = []
    t_cam_stack = np.stack(t_cam)
    length = int(np.ceil(256/5))
    for l in range(length):
        #l=0
        tmp_sum_cam = np.zeros((4,4))
        if (l+5)> length:
            pass
        else:
            tmp_tmp_sum_cam = t_cam_stack[l:l+5]
            for s in range(0,5):
                tmp_sum_cam += tmp_tmp_sum_cam[s]
        sum_cam.append(tmp_sum_cam)
    
    if torch.topk(pred_probabilities,1).indices[obs] == 0:
        pred = 'Protoss Win'
    else:
        pred = 'Terran Win'

    if Y[obs] == 0:
        true = 'Protoss Win'
    else:
        true = 'Terran Win'

    cam_np = np.stack(sum_cam)
    cam_np_f = cam_np.flatten()
    
    # adjust relu function
    cam_np_f[cam_np_f < 0] = 0
    
    cam_scaler = (cam_np_f - min(cam_np_f)) / (max(cam_np_f) - min(cam_np_f))
    cam_scaler = np.reshape(cam_scaler, cam_np.shape)

    sum_cam_scaler = []       
    for k in range(cam_scaler.shape[0]):
        sum_cam_scaler.append(cam_scaler[k,:,:])

    os.makedirs(RESULT_DIR+REPLAY_LIST[obs]+'/CAM_factor', exist_ok=True)    
    for sum_t in range(len(sum_cam_scaler)):
        #sum_t=0
        plt.figure(figsize=(8,8))
        sns.heatmap(sum_cam_scaler[sum_t], center=0)
        plt.savefig(RESULT_DIR+REPLAY_LIST[obs]+'/CAM_factor/'+'time_index_{}.png'.format(sum_t))
        plt.show()
        

