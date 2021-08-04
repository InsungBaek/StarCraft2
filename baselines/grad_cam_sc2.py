# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:10:42 2021

@author: InsungBaek
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import functools

from absl import app
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

sys.path.append('D:/1.Project/2019.04_Game AI/Code_HG/')
start_dir = 'D:/1.Project/2019.04_Game AI/Code_HG/'
os.chdir(start_dir)
del(start_dir)

from models.resnet import ResNet3D, ResNet3D_non_local, ResNet3D_post
from torch.utils.data import DataLoader
from utils_sc.data import SC2ReplayDataset
from utils_sc.data import replay_collate_fn
from utils_sc.misc import count_parameters

import seaborn as sns
import matplotlib.pyplot as plt
from absl import flags

FLAGS = flags.FLAGS
FLAGS(sys.argv)
# flags.DEFINE_string('root', default='./연구미팅/Test_set/', help="")
flags.DEFINE_string('root', default='./parsed/TvP/', help="")
flags.DEFINE_string('weighting', default='log', help="")
flags.DEFINE_integer('num_workers', default=4, help="")
flags.DEFINE_integer('num_epochs', default=30, help="")
flags.DEFINE_integer('batch_size', default=4, help="")
flags.DEFINE_integer('max_timesteps', default=50, help="")
flags.DEFINE_integer('num_classes', default=2, help="Equal to the output size of the final layer.")
flags.DEFINE_float('learning_rate', default=0.001, help="")
flags.DEFINE_bool('use_gpu', default=True, help='')
flags.DEFINE_float('start_prob', default=0.0, help="")    
flags.DEFINE_float('length_prob', default=1.0, help="")
flags.DEFINE_string('choice', default='random', help="grad-cam qulitative evlauation")

project_dir = 'D:/1.Project/2019.04_Game AI/Code_HG/'
replay_dir = os.path.join(project_dir, 'parsed/TvP/cam/')
result_dir = os.path.join(project_dir, 'IEEE-comment/cam_result')
replay_list = os.listdir(replay_dir)

assert all([os.path.isdir(project_dir), os.path.isdir(replay_dir)])

model_path = 'D:/1.Project/2019.04_Game AI/Code_HG/best_model/ResNet3D_non_local_025_0.973_0.900.pt'

class SaveFeatures():
    """Extract pretrained activations"""
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn) # forward gradiets
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()


def grad_cam(activated_features, weight_softmax, class_idx, result_dir):
    
    batch_sum_cam = []
    
    for obs in range(FLAGS.batch_size):
        #obs=0
        tmp_fm = activated_features.features[obs]
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
           
        cam_np = np.stack(sum_cam)
        cam_np_f = cam_np.flatten()
        
        # apply relu function
        cam_np_f[cam_np_f < 0] = 0
        
        cam_scaler = (cam_np_f - min(cam_np_f)) / (max(cam_np_f) - min(cam_np_f))
        cam_scaler = np.reshape(cam_scaler, cam_np.shape)
    
        sum_cam_scaler = []       
        for k in range(cam_scaler.shape[0]):
            sum_cam_scaler.append(cam_scaler[k,:,:])
        
        sum_cam_scaler = np.stack(sum_cam_scaler)
        sum_cam_scaler = sum_cam_scaler[:FLAGS.max_timesteps, :, :]
        batch_sum_cam.append(sum_cam_scaler)
        
    batch_sum_cam = np.stack(batch_sum_cam)
    return batch_sum_cam

def new_input_data(cam_score, model_inputs):
            
    new_vis_batch = []
    new_player_batch = []
    new_unit_batch = []
    
    n_s = 32 # new_szie = 32 (grad_cam 1pixel = original 32pixel)
    
    for bch in range(cam_score.shape[0]):

        new_vis_time = []
        new_player_time = []
        new_unit_time = []        

        for t in range(cam_score.shape[1]):                       
            cam_score_flat = np.sort(cam_score[bch,t,:,:].flatten())
                                 
            if FLAGS.choice == 'good':
                point = np.where(cam_score[bch,t,:,:] == cam_score_flat[-1])
                point_row, point_col = int(point[0][0]*n_s), int(point[1][0]*n_s) # original size = 128(4*32)

            if FLAGS.choice == 'bad':
                point = np.where(cam_score[bch,t,:,:] == cam_score_flat[0])                

                if (cam_score_flat[0] == 0) and (len(point[0]) >= 2):
                    number = int(torch.randint(0, len(point[0]), [1]))
                    point_row = point[0][number]*n_s
                    point_col = point[1][number]*n_s
                
                else:
                    point_row, point_col = int(point[0][0])*n_s, int(point[1][0])*n_s # original size = 128(4*32)                   
                    
            if FLAGS.choice == 'random':
                point_row, point_col = int(torch.randint(0,4, [1]))*n_s, int(torch.randint(0,4, [1]))*n_s
                       
            vis = model_inputs['visibility_map'][bch,t,point_row:point_row+32,point_col:point_col+32]
            pla_rel = model_inputs['player_relative'][bch,t,point_row:point_row+32,point_col:point_col+32]
            unit_tp = model_inputs['unit_type'][bch,t,point_row:point_row+32,point_col:point_col+32]                               

            new_vis_time.append(vis)
            new_player_time.append(pla_rel)
            new_unit_time.append(unit_tp)            
                
        new_vis_time = torch.stack(new_vis_time)
        new_player_time = torch.stack(new_player_time)
        new_unit_time = torch.stack(new_unit_time)
        
        new_vis_batch.append(new_vis_time)
        new_player_batch.append(new_player_time)
        new_unit_batch.append(new_unit_time)
    
    new_visibility = torch.stack(new_vis_batch)    
    new_player_relative = torch.stack(new_player_batch)            
    new_unit_type = torch.stack(new_unit_batch)

    new_inputs = {}
    new_inputs['visibility_map'] = new_visibility
    new_inputs['player_relative'] = new_player_relative
    new_inputs['unit_type'] = new_unit_type
    
    return new_inputs

    
def main(argv):

    INCLUDE = ['visibility_map', 'player_relative', 'unit_type']

    train_set = SC2ReplayDataset(FLAGS.root, train=True, include=INCLUDE)
    print(train_set.counts)

    train_loader = DataLoader(
        train_set,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=functools.partial(
            replay_collate_fn,
            max_timesteps=FLAGS.max_timesteps,
            weighting=FLAGS.weighting,
            start_prob=FLAGS.start_prob,
            length_prob=FLAGS.length_prob        
        ),
        num_workers=FLAGS.num_workers
    )
    print(f"Number of train minibatches per epoch: {len(train_loader)}")
            
    test_set = SC2ReplayDataset(FLAGS.root, train=False, include=INCLUDE)
    print(test_set.counts)

    test_loader = DataLoader(
        test_set,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        collate_fn=functools.partial(
            replay_collate_fn,
            max_timesteps=FLAGS.max_timesteps,
            weighting=FLAGS.weighting,
            start_prob=FLAGS.start_prob,
            length_prob=FLAGS.length_prob
        ),
        num_workers=FLAGS.num_workers
    )
    print(f"Number of test minibatches per epoch: {len(test_loader)}")

    # Instantiate model
    model_configs = {
        'num_classes': FLAGS.num_classes,
        'include': INCLUDE
    }

    model = ResNet3D_non_local(**model_configs)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Device configuration
    device = 'cuda' if (torch.cuda.is_available() and FLAGS.use_gpu) else 'cpu'
    criterion = criterion.to(device)
    model = model.to(device)

    '''grad-cam'''
    final_layer = model._modules.get('block3')
    
    activated_features = SaveFeatures(final_layer)

    history = {}
    history['train_loss'] = []
    history['train_acc'] = []
    history['test_loss'] = []
    history['test_acc'] = []

    new_model = ResNet3D_post(**model_configs)
    new_model.to(device)    

    # Define optimizer
    optimizer = optim.Adam(new_model.parameters(), lr=FLAGS.learning_rate)   

    print(f"Model has {count_parameters(new_model):,} trainable parameters.")       
    
    def save_history(history, ckpt_dir):
        """Save history."""
        filepath = os.path.join(ckpt_dir, 'history.json')
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved history to {filepath}")
    
    try:
        for epoch in range(FLAGS.num_epochs):
            start = time.time()
            print(f"Epoch: [{epoch+1:>03d}/{FLAGS.num_epochs:>03d}]")

            epoch_loss = .0
            num_correct = 0
            new_model.train()
            
            for i, batch in enumerate(train_loader):
                assert isinstance(batch, dict)
                print('.', end='')
        
                model_inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
                prediction = model({k: v.to(device) for k, v in batch['inputs'].items()})
                model_targets = batch['targets'].to(device).long()    
        
                # 2개 클래스 각각에 대한 예측확률
                pred_probabilities = F.softmax(prediction).data.squeeze()
                activated_features.remove()
                torch.topk(pred_probabilities,1)
                
                weight_softmax_params = list(model._modules.get('linear').parameters())
                weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
                class_idx = torch.topk(pred_probabilities,1)[1].int()
                
                # calculate cam_score
                cam_score = grad_cam(activated_features, weight_softmax, class_idx, result_dir)
                
                # creat new inputs
                new_inputs = new_input_data(cam_score, model_inputs)
                              
                optimizer.zero_grad()
                new_inputs_ = {k: v.to(device) for k, v in new_inputs.items()}
                # model_targets = batch['targets'].to(device).long()
                logits = new_model(new_inputs_)
        
                loss = criterion(logits, model_targets)
                loss.backward()
        
                optimizer.step()
                epoch_loss += loss.item()
        
                with torch.no_grad():
                    num_correct += torch.eq(logits.argmax(-1), model_targets).float().sum().item()
        
                if (i + 1) % 40 == 0:
                    print('\n', end='')
            
            train_loss = epoch_loss / len(train_loader)
            train_acc = num_correct / len(train_loader.dataset)

            print(f"\nTrain Loss: {train_loss:>.4f} | Train Acc: {train_acc:>.4f}")
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
 
            time.sleep(3.)

            # Test
            epoch_loss_ts = .0
            num_correct_ts = 0
            true = []
            pred = []            
            
            new_model.eval()
            
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    assert isinstance(batch, dict)
                    print('.', end='')
            
                    model_inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
                    prediction = model({k: v.to(device) for k, v in batch['inputs'].items()})
                    model_targets = batch['targets'].to(device).long()    
            
                    # 2개 클래스 각각에 대한 예측확률
                    pred_probabilities = F.softmax(prediction).data.squeeze()
                    activated_features.remove()
                    torch.topk(pred_probabilities,1)
                    
                    weight_softmax_params = list(model._modules.get('linear').parameters())
                    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
                    class_idx = torch.topk(pred_probabilities,1)[1].int()
             
                    cam_score = grad_cam(activated_features, weight_softmax, class_idx, result_dir)
                    
                    new_inputs = new_input_data(cam_score, model_inputs)                  
        
                    # with torch.no_grad():
                    #     for _, batch in enumerate(test_loader):
                    #         assert isinstance(batch, dict)
                
                    logits = new_model({k: v.to(device) for k, v in new_inputs.items()})
                    model_targets = batch['targets'].to(device).long()
                    loss = criterion(logits, model_targets)
        
                    epoch_loss_ts += loss.item()            
                    num_correct_ts += torch.eq(logits.argmax(-1), model_targets).float().sum().item()
                    
                    logits.argmax(-1)
                    true.append(model_targets)
                    pred.append(logits.argmax(-1))
                
            test_loss = epoch_loss_ts / len(test_loader)
            test_acc = num_correct_ts / len(test_loader.dataset)
            
            print(f"\nTest Loss: {test_loss:>.4f} | Test Acc: {test_acc:>.4f}")
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            # Save model checkpoint
            ckpt_dir = './checkpoints/'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_name = f"{new_model.__class__.__name__}_{epoch+1:>03d}_{train_acc:.3f}_{test_acc:.3f}.pt"
            ckpt_file = os.path.join(ckpt_dir, ckpt_name)
            torch.save(new_model.state_dict(), ckpt_file)
            end = time.time()
            print('{:.2f}'.format((end-start)/60) + ' min')
            print(f"Saved checkpoint to f{ckpt_file}")
 
    except KeyboardInterrupt:
        save_history(history=history, ckpt_dir=ckpt_dir)
        sys.exit()

    # Save history
    save_history(history=history, ckpt_dir=ckpt_dir)

if __name__ == '__main__':
    app.run(main)