# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convlstm import SimpleConvLSTM
from models.resnet import ResNet3D
from utils_sc.data import SC2ReplayDataset
from utils_sc.data import replay_collate_fn
from utils_sc.misc import count_parameters
from torch.utils.data import DataLoader
import json
import functools

import os
import sys
import numpy as np

sys.path.append('D:/1.Project/2019.04_Game AI/Code_HG')
start_dir = "D:/1.Project/2019.04_Game AI/Code_HG"
os.chdir(start_dir)
del(start_dir)

from layers.convolutional import BasicBlock3D
from layers.embeddings import ScalarEmbedding
from layers.embeddings import CategoricalEmbedding
from layers.convolutional import SimpleConv
from layers.recurrent import SimpleGRU
from layers.attention import VaswaniAttention
from features.custom_features import SPATIAL_FEATURES

from absl import flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_string('root', default='./sample/', help="")
flags.DEFINE_string('weighting', default='log', help="")
flags.DEFINE_integer('num_workers', default=1, help="")
flags.DEFINE_integer('num_epochs', default=5, help="")
flags.DEFINE_integer('batch_size', default=1, help="")
flags.DEFINE_integer('max_timesteps', default=1, help="")
flags.DEFINE_integer('num_classes', default=2, help="Equal to the output size of the final layer.")
flags.DEFINE_float('learning_rate', default=0.001, help="")
flags.DEFINE_bool('use_gpu', default=True, help='')
flags.DEFINE_float('start_prob', default=0.0, help="")    
flags.DEFINE_float('length_prob', default=1.0, help="")

INCLUDE=['visibility_map','player_relative', 'unit_type']

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

   for i, batch, in enumerate(train_loader):

        assert isinstance(batch, dict)
        print(f"Batch size: {len(batch)}")
        inputs_ = batch.get('inputs')     # dictionary of lists
        targets_ = batch.get('targets')   # list

#        for j, (name_, feat_) in enumerate(inputs_.items()):
#            type_ = str(SPATIAL_SPECS[name_].type).split('.')[-1]
#            scale_ = SPATIAL_SPECS[name_].scale
#            print(f"[{j:>02}] Name: {name_:<15} | Type: {type_:<11} | Scale: {scale_:>4} | Shape: {feat_.size()}")

    model_configs = {
        'num_classes': FLAGS.num_classes,
        'include': INCLUDE
    }
    model = ResNet3D(**model_configs)
    print(f"Model has {count_parameters(model):,} trainable parameters.")

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

class BasicBlock2D(nn.Module):
    """Add class docstring."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock2D, self).__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_ch)
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_ch)

        self.conv_skip = nn.Conv2d(self.in_ch, self.out_ch, 1, padding=0, bias=True)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += self.conv_skip(residual)
        out = F.relu(out)

        return out
   
class ResNet2D(nn.Module):
    """Add class docstring."""
    def __init__(self, embedding_dims=None, num_classes=2, include=['visibility_map','player_relative', 'unit_type']):
        super(ResNet2D, self).__init__()

        if embedding_dims is None:
            self.embedding_dims = {
                'height_map': 10,
                'visibility_map': 10,
                'player_relative': 10,
                'unit_type': 100
            }
        else:
            assert isinstance(embedding_dims, dict)
            self.embedding_dims = embedding_dims

        self.num_classes = num_classes
        self.include = include
        self.cnn_channel_size = 0

        """Embedding layers."""
        self.embeddings = nn.ModuleDict()
        for name, feat in SPATIAL_FEATURES._asdict().items():
            if name not in self.include:
                continue
            feat_type = str(feat.type).split('.')[-1]
            if feat_type == 'CATEGORICAL':
                self.embeddings[name] = CategoricalEmbedding(
                    category_size=feat.scale,
                    embedding_dim=self.embedding_dims.get(name),
                    name=name,
                )
            elif feat_type == 'SCALAR':
                self.embeddings[name] = ScalarEmbedding(
                    embedding_dim=self.embedding_dims.get(name),
                    name=name
                )
            else:
                raise NotImplementedError
            self.cnn_channel_size += self.embedding_dims.get(name)

        self.conv1 = nn.Conv2d(self.cnn_channel_size, 32, kernel_size=7, stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.block1 = BasicBlock2D(32, 64, stride=1, downsample=None)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 4), padding=0)
        self.block2 = BasicBlock2D(64, 128, stride=2, downsample=None)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)
        self.block3 = BasicBlock2D(128, 256, stride=2, downsample=None)
        self.pool3 = nn.AvgPool2d(kernel_size=(4, 4), stride=1, padding=0)
        self.drop_out = nn.Dropout(0.5)

        self.linear = nn.Linear(256, num_classes)


    def forward(self, inputs):
        """
        Arguments:
            inputs: dict, with feature name as keys and 4d tensors as values.
                each 4d tensor has shape (B, T, H, W). A list of 3D tensors with shape (T, H, W) is
                also allowed.
        """
        assert isinstance(inputs, dict)

        embeddings = nn.ModuleDict()
        for name, feat in SPATIAL_FEATURES._asdict().items():
            if name not in include:
                continue
            feat_type = str(feat.type).split('.')[-1]
            if feat_type == 'CATEGORICAL':
                embeddings[name] = CategoricalEmbedding(
                    category_size=feat.scale,
                    embedding_dim=embedding_dims.get(name),
                    name=name,
                )
            elif feat_type == 'SCALAR':
                embeddings[name] = ScalarEmbedding(
                    embedding_dim=embedding_dims.get(name),
                    name=name
                )
            else:
                raise NotImplementedError
            cnn_channel_size += embedding_dims.get(name)

        embedded = []
        for name, x in inputs.items():
            emb_out = embeddings[name](x)
            embedded.append(emb_out.float())
        embedded = torch.cat(embedded, dim=2)       # (B, T, cnn_channel_size, H, W)
        embedded2 = torch.squeeze(embedded, dim=1)  # (B, cnn_channel_size, H, W)
#        embedded = embedded.permute(0, 2, 1, 3, 4)  # (B, cnn_channel_size, T, H, W)

        x = self.conv1(embedded)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.pool1(x)
        x = self.drop_out(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.drop_out(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.drop_out(x)

        logits = self.linear(x.view(x.size(0), -1))

        return logits

class ResNet2D_ck(nn.Module):
    """Add class docstring."""
    def __init__(self, embedding_dims=None, num_classes=2, include=['visibility_map','player_relative', 'unit_type']):
        super(ResNet2D, self).__init__()
        
        self.cnn_channel_size = cnn_channel_size
        
        self.conv1 = nn.Conv2d(self.cnn_channel_size, 32, kernel_size=7, stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.block1 = BasicBlock2D(32, 64, stride=1, downsample=None)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 4), padding=0)
        self.block2 = BasicBlock2D(64, 128, stride=2, downsample=None)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)
        self.block3 = BasicBlock2D(128, 256, stride=2, downsample=None)
        self.pool3 = nn.AvgPool2d(kernel_size=(4, 4), stride=1, padding=0)
        self.drop_out = nn.Dropout(0.5)

        self.linear = nn.Linear(256, num_classes)

    def forward(self, inputs):
        """
        Arguments:
            inputs: dict, with feature name as keys and 4d tensors as values.
                each 4d tensor has shape (B, T, H, W). A list of 3D tensors with shape (T, H, W) is
                also allowed.
        """
        assert isinstance(inputs, dict)
        
        x = self.conv1(embedded)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.pool1(x)
        x = self.drop_out(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.drop_out(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = self.drop_out(x)

        logits = self.linear(x.view(x.size(0), -1))

        return logits
        
#model = ResNet2D_ck(**model_configs)
#print(f"Model has {count_parameters(model):,} trainable parameters.")
#
#
##check model architecture
#print("Model's state_dict:")
#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
