# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import functools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append('D:/1.Project/2019.04_Game AI/Code_HG')
start_dir = "D:/1.Project/2019.04_Game AI/Code_HG"
os.chdir(start_dir)
del(start_dir)

from models.convlstm import SimpleConvLSTM
from models.resnet import ResNet3D, ResNet3D_non_local, ResNet2D
from utils_sc.data import SC2ReplayDataset
from utils_sc.data import replay_collate_fn
from utils_sc.misc import count_parameters

FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_string('root', default='./parsed/TvP/', help="")
# flags.DEFINE_string('root', default='./연구미팅/Test_set/', help="")
flags.DEFINE_string('weighting', default='log', help="")
flags.DEFINE_integer('num_workers', default=4, help="")
flags.DEFINE_integer('num_epochs', default=50, help="")
flags.DEFINE_integer('batch_size', default=4, help="")
flags.DEFINE_integer('max_timesteps', default=1, help="")
flags.DEFINE_integer('num_classes', default=2, help="Equal to the output size of the final layer.")
flags.DEFINE_float('learning_rate', default=0.001, help="")
flags.DEFINE_bool('use_gpu', default=True, help='')
flags.DEFINE_float('start_prob', default=0.0, help="")    
flags.DEFINE_float('length_prob', default=1.0, help="")
 
def evaluate(model, iterator, criterion, device):

#    assert isinstance(model, nn.Module)
    assert isinstance(iterator, DataLoader)

    model.eval()

    epoch_loss = .0
    num_correct = 0
    
    true = []
    pred = []
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            assert isinstance(batch, dict)
            print('.', end='')

            logits = model({k: v.to(device) for k, v in batch['inputs'].items()})
            model_targets = batch['targets'].to(device).long()
            loss = criterion(logits, model_targets)

            epoch_loss += loss.item()            
            num_correct += torch.eq(logits.argmax(-1), model_targets).float().sum().item()
            
            logits.argmax(-1)
            true.append(model_targets)
            pred.append(logits.argmax(-1))
    
    test_loss = epoch_loss / len(iterator)
    test_acc = num_correct / len(iterator.dataset)
    
    return test_loss, test_acc, num_correct, true, pred

def main(argv):

    INCLUDE = ['visibility_map', 'player_relative', 'unit_type']

    test_set = SC2ReplayDataset(FLAGS.root, train=False, include=INCLUDE)
    print(test_set.counts)

    # model_dir = './paper_result/model/'
    model_dir = './IEEE-comment/07.13-2D/'
    model_list = os.listdir(model_dir)
    result_path = os.path.join(model_dir, 'result/')
    
    i = 0
    # for i in range(len(model_list)):
    # # for i in range(0,3):
    #     if i == 0:
    #         start_prob_ = 0.0
    #         length_prob_ = 0.25
        
    #     elif i == 1:
    #         start_prob_ = 0.25
    #         length_prob_ = 0.5
        
    #     elif i == 2:
    #         start_prob_ = 0.5
    #         length_prob_ = 0.75
        
    #     else:
    #         start_prob_ = 0.75
    #         length_prob_ = 1.0            
            
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
            # start_prob = start_prob_,
            # length_prob = length_prob_
        ),
        num_workers=FLAGS.num_workers
    )
    print(f"Number of test minibatches per epoch: {len(test_loader)}")

    # Instantiate model
    model_configs = {
        'num_classes': FLAGS.num_classes,
        'include': INCLUDE
    }

    model_path = os.path.join(model_dir, model_list[i])
    model = ResNet2D(**model_configs)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Device configuration
    device = 'cuda' if (torch.cuda.is_available() and FLAGS.use_gpu) else 'cpu'
    criterion = criterion.to(device)
    model = model.to(device)
    
    history = {}
    history['test_loss'] = []
    history['test_acc'] = []

    def save_history(history, model_path):
        """Save history."""
        os.makedirs(model_path, exist_ok=True)
        filepath = os.path.join(model_path, 'history.json')
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved history to {filepath}")

    os.makedirs(result_path+model_list[i]+'_CF', exist_ok=True)
    try:
        for epoch in range(FLAGS.num_epochs):

            print(f"Epoch: [{epoch+1:>03d}/{FLAGS.num_epochs:>03d}]")
    
            time.sleep(3.)

            # Test
            test_loss, test_acc, num_correct, true, pred = evaluate(model, test_loader, criterion, device)
            print(f"\nTest Loss: {test_loss:>.4f} | Test Acc: {test_acc:>.4f}")
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            true_ = true[0]
            pred_ = pred[0]
            for j in range(1, len(true)):
                true_ = torch.cat([true_, true[j]], dim=0)
                pred_ = torch.cat([pred_, pred[j]], dim=0)

            true_ = true_.to("cpu")
            pred_ = pred_.to("cpu")                

            cf_index = ['Protoss_win', 'Protoss_Lose']
            cf_col = ['pred_win','pred_lose']
            cf = pd.DataFrame(confusion_matrix(true_, pred_), index=cf_index, columns=cf_col)
            
            os.makedirs(result_path+model_list[i]+'_CF', exist_ok=True)          
            cf.to_csv(result_path+model_list[i]+'_CF'+'/CF_No_'+str(i+1)+'_epoch_'+str(epoch+1)+'.csv')

            acc = accuracy_score(true_, pred_)
            recall = cf.loc['Protoss_win','pred_win']/(cf.sum(axis=0)[0])
            precision = cf.loc['Protoss_win','pred_win']/(cf.sum(axis=1)[0])
            f1 = (2*recall*precision)/(recall+precision)
            
            print("Accracy = {0} | Recall = {1} | Preision = {2} | F1 = {3}".format(round(acc, 4), round(recall, 4),
                                                                                  round(precision, 4), round(f1, 4)))                                 

            if epoch == 0:
                results = pd.DataFrame([], columns=['Acc', 'Recall', 'Precision', 'F1'])
            else:
                results = pd.read_csv(result_path+model_list[i]+'.csv', index_col=0)

            result = np.zeros((1,4))
            result[0,0] = acc
            result[0,1] = recall
            result[0,2] = precision
            result[0,3] = f1
          
            result = pd.DataFrame(result, columns=['Acc', 'Recall', 'Precision', 'F1'])
            results = pd.concat([results, result], axis=0)
            results.to_csv(result_path+model_list[i]+'.csv', encoding='ms949')   
            # Save model checkpoint
            #ckpt_dir = './checkpoints_test/'
#                os.makedirs(result_path, exist_ok=True)
            #ckpt_name = f"{model.__class__.__name__}_{epoch:>03d_{test_acc:.3f}.pt"
            ckpt_file = os.path.join(result_path, model_list[i])
            torch.save(model.state_dict(), ckpt_file)
            print(f"Saved checkpoint to f{ckpt_file}")

    except KeyboardInterrupt:
        save_history(history=history, model_path=result_path)
        sys.exit()

    # Save history
    save_history(history=history, model_path=result_path)

if __name__ == '__main__':
    app.run(main)
    

