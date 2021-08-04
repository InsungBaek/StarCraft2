# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import functools

import torch
print(torch.__version__)
torch.cuda.is_available()
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append('D:/1.Project/2019.04_Game AI/Code_HG')
start_dir = "D:/1.Project/2019.04_Game AI/Code_HG"
os.chdir(start_dir)
del(start_dir)

from models.resnet import ResNet3D, ResNet3D_non_local, VGGnet, ResNet2D
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
flags.DEFINE_float('length_prob', default=0.25, help="")

def train(model, iterator, optimizer, criterion, device):

    assert isinstance(model, nn.Module)
    assert isinstance(iterator, DataLoader)
    assert isinstance(optimizer, optim.Optimizer)

    model.train()

    epoch_loss = .0
    num_correct = 0

    for i, batch in enumerate(iterator):
        assert isinstance(batch, dict)
        print('.', end='')

        optimizer.zero_grad()
        model_inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
        model_targets = batch['targets'].to(device).long()
        logits = model(model_inputs)

        loss = criterion(logits, model_targets)
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()

        with torch.no_grad():
            num_correct += torch.eq(logits.argmax(-1), model_targets).float().sum().item()

        if (i + 1) % 40 == 0:
            print('\n', end='')

    train_loss = epoch_loss / len(iterator)
    train_acc = num_correct / len(iterator.dataset)

    return train_loss, train_acc


def evaluate(model, iterator, criterion, device):

    assert isinstance(model, nn.Module)
    assert isinstance(iterator, DataLoader)

    model.eval()

    epoch_loss = .0
    num_correct = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            assert isinstance(batch, dict)
            print('.', end='')

            logits = model({k: v.to(device) for k, v in batch['inputs'].items()})
            model_targets = batch['targets'].to(device).long()
            loss = criterion(logits, model_targets)

            epoch_loss += loss.item()

            num_correct += torch.eq(logits.argmax(-1), model_targets).float().sum().item()

    test_loss = epoch_loss / len(iterator)
    test_acc = num_correct / len(iterator.dataset)

    return test_loss, test_acc


def main(argv):

    INCLUDE = ['visibility_map', 'player_relative', 'unit_type']
#    INCLUDE = ['unit_type', 'graphic']

    # Load dataset & data loader (train)
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

    # Load dataset & data loader (test)
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

    # model_path = 'D:/1.Project/2019.04_Game AI/Code_HG/checkpoints/ResNet3D_non_local_016_0.946_0.830.pt'
    # Instantiate model
    model_configs = {
        'num_classes': FLAGS.num_classes,
        'include': INCLUDE,
    }
    model = ResNet2D(**model_configs)
    # model.load_state_dict(torch.load(model_path))       
    print(f"Model has {count_parameters(model):,} trainable parameters.")
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Device configuration
    device = 'cuda' if (torch.cuda.is_available() and FLAGS.use_gpu) else 'cpu'
    model = model.to(device)
    criterion = criterion.to(device)

    # inp = next(iter(train_loader))['inputs']
    # tar = next(iter(train_loader))['targets']
    # model_inputs = {k: v.to(device) for k, v in inp.items()}
    # model_targets = tar.to(device).long()
    # check = model(model_inputs)

    history = {}
    history['train_loss'] = []
    history['train_acc'] = []
    history['test_loss'] = []
    history['test_acc'] = []

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

            # Train
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            print(f"\nTrain Loss: {train_loss:>.4f} | Train Acc: {train_acc:>.4f}")
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
 
            time.sleep(3.)

            # Test
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f"\nTest Loss: {test_loss:>.4f} | Test Acc: {test_acc:>.4f}")
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            # Save model checkpoint
            ckpt_dir = './checkpoints/'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_name = f"{model.__class__.__name__}_{epoch+1:>03d}_{train_acc:.3f}_{test_acc:.3f}.pt"
            ckpt_file = os.path.join(ckpt_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_file)
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

