# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.convolutional import BasicBlock3D, BasicBlock2D, VggBlock3D
from layers.embeddings import ScalarEmbedding
from layers.embeddings import CategoricalEmbedding
from layers.convolutional import SimpleConv
from layers.recurrent import SimpleGRU
from layers.attention import VaswaniAttention
from layers.non_local_embedded_gaussian import NONLocalBlock3D
# from layers.non_local_dot_product import NONLocalBlock3D
# from layers.non_local_gaussian import NONLocalBlock3D
# from layers.non_local_concatenation import NONLocalBlock3D

from features.custom_features import SPATIAL_FEATURES

class ResNet3D(nn.Module):
    """Add class docstring."""
    def __init__(self, embedding_dims=None, num_classes=2, include=['visibility_map', 'player_relative', 'unit_type']):
        super(ResNet3D, self).__init__()

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

        self.conv1 = nn.Conv3d(self.cnn_channel_size, 32, 7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.block1 = BasicBlock3D(32, 64, stride=1, downsample=None)
        self.pool1 = nn.MaxPool3d(kernel_size=(5, 4, 4), padding=0)
        self.block2 = BasicBlock3D(64, 128, stride=2, downsample=None)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), padding=0)
        self.block3 = BasicBlock3D(128, 256, stride=2, downsample=None)
        self.pool3 = nn.AvgPool3d(kernel_size=(5, 4, 4), stride=1, padding=0)
        # self.block4 = BasicBlock3D(256, 50, stride=1, downsample=None)
        # self.non_block2 = NONLocalBlock3D(in_channels=50, sub_sample=True, bn_layer=True)                        
        # self.pool4 = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=1, padding=0)        
        self.drop_out = nn.Dropout(0.5)

        self.linear = nn.Linear(256, self.num_classes)

    def forward(self, inputs):
        """
        Arguments:
            inputs: dict, with feature name as keys and 4d tensors as values.
                each 4d tensor has shape (B, T, H, W). A list of 3D tensors with shape (T, H, W) is
                also allowed.
        """
        assert isinstance(inputs, dict)

        embedded = []
        for name, x in inputs.items():
            emb_out = self.embeddings[name](x)
            embedded.append(emb_out.float())
        embedded = torch.cat(embedded, dim=2)       # (B, T, cnn_channel_size, H, W)
        embedded = embedded.permute(0, 2, 1, 3, 4)  # (B, cnn_channel_size, T, H, W)

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
        # x = self.block4(x)
        # x = self.non_block2(x)
        # x = self.pool4(x)
        # x = self.drop_out(x)

        logits = self.linear(x.view(x.size(0), -1))

        return logits
    
class ResNet2D(nn.Module):
    """Add class docstring."""
    def __init__(self, embedding_dims=None, num_classes=2, include=['visibility_map','player_relative', 'unit_type', 'graphic']):
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

        embedded = []
        # all_data = []
        # graphic_data = inputs['graphic']
        # del(inputs['graphic'])
        for name, x in inputs.items():
            # emb_2d= self.embeddings[name](x).unsqueeze(1)
            emb_out = self.embeddings[name](x.unsqueeze(1))
            # emb_out = self.embeddings[name](x)            
            embedded.append(emb_out.float())
        embedded = torch.cat(embedded, dim=2)       # (B, T, cnn_channel_size, H, W)
        embedded = torch.squeeze(embedded, dim=1)  # (B, cnn_channel_size, H, W)
        # embedded = embedded.permute(0, 2, 1, 3, 4)  # (B, cnn_channel_size, T, H, W)
        # all_data.append(graphic_data)
        # all_data.append(embedded)
        
        # embedded = torch.cat(all_data, dim=1)

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


class ResNet3D_non_local(nn.Module):
    """Add class docstring."""
    def __init__(self, embedding_dims=None, num_classes=2, include=['visibility_map', 'player_relative', 'unit_type']):
        super(ResNet3D_non_local, self).__init__()

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

        self.conv1 = nn.Conv3d(self.cnn_channel_size, 32, 7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.block1 = BasicBlock3D(32, 64, stride=1, downsample=None)        
        self.pool1 = nn.MaxPool3d(kernel_size=(5, 4, 4), padding=0)
        self.block2 = BasicBlock3D(64, 128, stride=2, downsample=None)
        # self.non_block1 = NONLocalBlock3D(in_channels=128, sub_sample=True, bn_layer=True) 
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), padding=0)
        self.block3 = BasicBlock3D(128, 256, stride=2, downsample=None)
        self.non_block2 = NONLocalBlock3D(in_channels=256, sub_sample=True, bn_layer=True)                
        self.pool3 = nn.AvgPool3d(kernel_size=(5, 4, 4), stride=1, padding=0)
        self.drop_out = nn.Dropout(0.5)

        self.linear = nn.Linear(256, self.num_classes)

    def forward(self, inputs):
        """
        Arguments:
            inputs: dict, with feature name as keys and 4d tensors as values.
                each 4d tensor has shape (B, T, H, W). A list of 3D tensors with shape (T, H, W) is
                also allowed.
        """
        assert isinstance(inputs, dict)

        embedded = []
        for name, x in inputs.items():
            emb_out = self.embeddings[name](x)
            embedded.append(emb_out.float())
        embedded = torch.cat(embedded, dim=2)       # (B, T, cnn_channel_size, H, W)
        embedded = embedded.permute(0, 2, 1, 3, 4)  # (B, cnn_channel_size, T, H, W)

        x = self.conv1(embedded)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.pool1(x)
        x = self.drop_out(x)
        x = self.block2(x)
        # x = self.non_block1(x)
        x = self.pool2(x)
        x = self.drop_out(x)
        x = self.block3(x)
        x = self.non_block2(x)
        x = self.pool3(x)
        x = self.drop_out(x)

        logits = self.linear(x.view(x.size(0), -1))

        return logits
    
class ResNet3D_post(nn.Module):
    """Add class docstring."""
    def __init__(self, embedding_dims=None, num_classes=2, include=['visibility_map', 'player_relative', 'unit_type']):
        super(ResNet3D_post, self).__init__()

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

        self.conv1 = nn.Conv3d(self.cnn_channel_size, 32, 7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.block1 = BasicBlock3D(32, 64, stride=1, downsample=None)
        self.pool1 = nn.MaxPool3d(kernel_size=(5, 2, 2), padding=0)
        self.block2 = BasicBlock3D(64, 128, stride=2, downsample=None)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), padding=0)
        self.block3 = BasicBlock3D(128, 256, stride=2, downsample=None)
        self.non_block2 = NONLocalBlock3D(in_channels=256, sub_sample=True, bn_layer=True)               
        self.pool3 = nn.AvgPool3d(kernel_size=(5, 2, 2), stride=1, padding=0)
    
        self.drop_out = nn.Dropout(0.5)

        self.linear = nn.Linear(256, self.num_classes)

    def forward(self, inputs):
        """
        Arguments:
            inputs: dict, with feature name as keys and 4d tensors as values.
                each 4d tensor has shape (B, T, H, W). A list of 3D tensors with shape (T, H, W) is
                also allowed.
        """
        assert isinstance(inputs, dict)

        embedded = []
        for name, x in inputs.items():
            emb_out = self.embeddings[name](x)
            embedded.append(emb_out.float())
        embedded = torch.cat(embedded, dim=2)       # (B, T, cnn_channel_size, H, W)
        embedded = embedded.permute(0, 2, 1, 3, 4)  # (B, cnn_channel_size, T, H, W)

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
        x = self.non_block2(x)
        x = self.pool3(x)
        x = self.drop_out(x)

        logits = self.linear(x.view(x.size(0), -1))

        return logits    
        
    
class VGGnet(nn.Module):
    """Add class docstring."""
    def __init__(self, embedding_dims=None, num_classes=2, include=['visibility_map', 'player_relative', 'unit_type']):
        super(VGGnet, self).__init__()

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

        self.conv1 = nn.Conv3d(self.cnn_channel_size, 32, 7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 16, 3, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 16, 3, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn3 = nn.BatchNorm3d(16)

        self.block1 = VggBlock3D(16, 32, stride=1, downsample=None, layer_count=2)
        self.block2 = VggBlock3D(32, 64, stride=1, downsample=None, layer_count=2)
        self.block3 = VggBlock3D(64, 128, stride=1, downsample=None, layer_count=2)
        self.block4 = VggBlock3D(128, 128, stride=1, downsample=None, layer_count=2)        
        # self.block5 = VggBlock3D(256, 256, stride=1, downsample=None, layer_count=3)                
        self.drop_out = nn.Dropout(0.5)
        self.avgpool = nn.AvgPool3d(kernel_size=(3, 3, 3), stride=1, padding=0)

        self.linear1 = nn.Linear(4096, 4096)
        self.linear2 = nn.Linear(4096, 1000)
        self.linear3 = nn.Linear(1000, self.num_classes)

    def forward(self, inputs):
        """
        Arguments:
            inputs: dict, with feature name as keys and 4d tensors as values.
                each 4d tensor has shape (B, T, H, W). A list of 3D tensors with shape (T, H, W) is
                also allowed.
        """
        assert isinstance(inputs, dict)

        embedded = []
        for name, x in inputs.items():
            emb_out = self.embeddings[name](x)
            embedded.append(emb_out.float())
        embedded = torch.cat(embedded, dim=2)       # (B, T, cnn_channel_size, H, W)
        embedded = embedded.permute(0, 2, 1, 3, 4)  # (B, cnn_channel_size, T, H, W)

        x = self.conv1(embedded)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        # x = self.avgpool(x)        
        x = self.drop_out(x)
        x = self.block2(x)
        x = self.avgpool(x)        
        x = self.drop_out(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = self.drop_out(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = self.drop_out(x)
        # x = self.block5(x)
        # x = self.avgpool(x)
        # x = self.drop_out(x)
        
        x = self.linear1(x.view(x.size(0), -1))
        x = self.linear2(x)
        logits = self.linear3(x)

        return logits
    
class SlowFastNet(nn.Module):
    """Add class docstring."""
    def __init__(self, embedding_dims=None, num_classes=2, include=['visibility_map', 'player_relative', 'unit_type']):
        super(SlowFastNet, self).__init__()

        if embedding_dims is None:
            self.embedding_dims = {
                'height_map': 10,
                'visibility_map': 10,
                'player_relative': 10,
                'unit_type': 50
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

        self.conv1 = nn.Conv3d(self.cnn_channel_size, 32, 7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.block1 = BasicBlock3D(32, 64, stride=1, downsample=None)        
        self.pool1 = nn.MaxPool3d(kernel_size=(5, 4, 4), padding=0)
        self.block2 = BasicBlock3D(64, 128, stride=2, downsample=None)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), padding=0)
        # self.non_block1 = NONLocalBlock3D(in_channels=128) 
        self.block3 = BasicBlock3D(128, 256, stride=2, downsample=None)
        self.non_block1 = NONLocalBlock3D(in_channels=256, bn_layer=False)                
        self.pool3 = nn.AvgPool3d(kernel_size=(1, 4, 4), stride=1, padding=0)
        self.drop_out = nn.Dropout(0.5)

        self.linear = nn.Linear(288, self.num_classes)
        
        self.block1_2 = BasicBlock3D(32, 8, stride=1, downsample=None)
        self.block2_2 = BasicBlock3D(8, 16, stride=2, downsample=None)
        self.block3_2 = BasicBlock3D(16, 32, stride=2, downsample=None)
        self.non_block2 = NONLocalBlock3D(in_channels=32, bn_layer=False)     
        self.pool3_2 = nn.AvgPool3d(kernel_size=(4, 4, 4), stride=1, padding=0)
        
    def forward(self, inputs):
        """
        Arguments:
            inputs: dict, with feature name as keys and 4d tensors as values.
                each 4d tensor has shape (B, T, H, W). A list of 3D tensors with shape (T, H, W) is
                also allowed.
        """
        assert isinstance(inputs, dict)

        embedded = []
        for name, x in inputs.items():
            emb_out = self.embeddings[name](x)
            embedded.append(emb_out.float())
        embedded = torch.cat(embedded, dim=2)       # (B, T, cnn_channel_size, H, W)
        embedded = embedded.permute(0, 2, 1, 3, 4)  # (B, cnn_channel_size, T, H, W)
        
        x1 = []
        x2 = []
        for i in range(embedded.shape[2]):
            x_ = embedded[:,:,i,:,:]
            x_ = x_.unsqueeze(dim=2)
            if (i+1) % 5 == 0:
                x1.append(x_)
            else:
                x2.append(x_)
                
        x1 = torch.cat(x1, dim=2)
        x2 = torch.cat(x2, dim=2)        

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.block1(x1)
        x1 = self.pool1(x1)
        x1 = self.drop_out(x1)
        x1 = self.block2(x1)
        x1 = self.pool2(x1)
        x1 = self.drop_out(x1)
        x1 = self.block3(x1)
        x1 = self.non_block1(x1)

        x1 = self.pool3(x1)
        x1 = self.drop_out(x1)
        
        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = F.relu(x2)
        x2 = self.maxpool(x2)

        x2 = self.block1_2(x2)
        x2 = self.pool1(x2)
        x2 = self.drop_out(x2)
        x2 = self.block2_2(x2)
        x2 = self.pool2(x2)
        x2 = self.drop_out(x2)
        x2 = self.block3_2(x2)
        x2 = self.non_block2(x2)

        x2 = self.pool3_2(x2)
        x2 = self.drop_out(x2)    
        
        x = []
        x.append(x1)
        x.append(x2)
        x = torch.cat(x, dim=1)        

        logits = self.linear(x.view(x.size(0), -1))

        return logits

#model_configs = {
#    'num_classes': 2,
#    'include': ['visibility_map', 'player_relative', 'unit_type']
#}
#
#from utils_sc.misc import count_parameters
#model = ResNet3D_CAM(**model_configs)
##print(f"Model has {count_parameters(model):,} trainable parameters.")
##check model architecture
#print("Model's state_dict:")
#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
