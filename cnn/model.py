import os
import sys
import time
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
import utils
    

class Node(nn.Module):
    def __init__(self, x_id, x_op, y_id, y_op, x_shape, y_shape, out_filters, stride=1, drop_path_keep_prob=None,
                 layer_id=0, num_layers=0, num_steps=0):
        super(Node, self).__init__()
        self.out_filters = out_filters
        self.stride = stride
        self.drop_path_keep_prob = drop_path_keep_prob
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.x_id = x_id
        self.y_id = y_id
        
        x_stride = stride if x_id in [0, 1] else 1
        self.x_op = nn.Sequential(OPERATIONS[x_op](out_filters, x_stride, True))
        if x_op in [0, 1]:
            pass
        elif x_op in [2, 3]:
            if x_shape[-1] != out_filters:
                self.x_op.add_module('pool_conv', ReLUConvBN(x[-1], out_filters, 1, 1, 0))
        else:
            if x_stride  > 1:
                assert x_stride == 2
                self.x_op.add_module('id_fact_reduce', FactorizedReduce(x_shape[-1], out_filters))
            if x_shape[-1] != out_filters:
                self.x_op.add_module('id_conv', ReLUConvBN(x[-1], out_filters, 1, 1, 0))
        out_x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, out_filters]

        y_stride = stride if y_id in [0, 1] else 1
        self.y_op = nn.Sequential(OPERATIONS[y_op](out_filters, y_stride, True))
        if y_op in [0, 1]:
            pass
        elif y_op in [2, 3]:
            if y_shape[-1] != out_filters:
                self.y_op.add_module('pool_conv', ReLUConvBN(y[-1], out_filters, 1, 1, 0))
        else:
            if y_stride > 1:
                assert y_stride == 2
                self.y_op.add_module('id_fact_reduce', FactorizedReduce(y_shape[-1], out_filters))
            if y_shape[-1] != out_filters:
                self.y_op.add_module('id_conv', ReLUConvBN(y[-1], out_filters, 1, 1, 0))
        out_y_shape = [y_shape[0] // y_stride, y_shape[1] // y_stride, out_filters]
        
        assert out_x_shape[0] == out_y_shape[0] and out_x_shape[1] == out_y_shape[1]
        self.out_shape = [out_x_shape[0], out_x_shape[1], out_x_shape[2]]
        
    def forward(self, x, y, step):
        x = self.x_op(x)
        if self.x_id in [0, 1, 2, 3] and self.drop_path_keep_prob is not None and self.training:
            x = apply_drop_path(x, self.drop_path_keep_prob, self.layer_id, self.num_layers, step, self.num_steps)
        y = self.y_op(y)
        if self.y_id in [0, 1, 2, 3] and self.drop_path_keep_prob is not None and self.training:
            y = apply_drop_path(y, self.drop_path_keep_prob, self.layer_id, self.num_layers, step, self.num_steps)
        print(x.shape,y.shape,self.x_id,self.y_id,self.x_op,self.y_op)
        out = x + y
        return out
    

class Cell(nn.Module):
    def __init__(self, arch, prev_layers, out_filters, reduction, layer_id, num_layers, num_steps, drop_path_keep_prob=None):
        super(Cell, self).__init__()
        print(prev_layers)
        assert len(prev_layers) == 2
        self.arch = arch
        self.reduction = reduction
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleList()
        self.num_nodes = len(arch) // 4
        self.used = [0] * (self.num_nodes + 2)
        
        # maybe calibrate size
        layers = [list(prev_layers[0]), list(prev_layers[1])]
        self.maybe_calibrate_size = MaybeCalibrateSize(layers, out_filters)
        layers = self.maybe_calibrate_size.out_shape

        self.layer_base = ReLUConvBN(layers[1][-1], out_filters, 1, 1, 0)
        layers[1][-1] = out_filters
        
        stride = 2 if self.reduction else 1
        for i in range(self.num_nodes):
            x_id, x_op, y_id, y_op = arch[4*i], arch[4*i+1], arch[4*i+2], arch[4*i+3]
            x_shape, y_shape = layers[x_id], layers[y_id]
            node = Node(x_id, x_op, y_id, y_op, x_shape, y_shape, out_filters, stride, drop_path_keep_prob, layer_id, num_layers, num_steps)
            self.ops.append(node)
            self.used[x_id] += 1
            self.used[y_id] += 1
            layers.append(node.out_shape)
        
        self.concat = []
        for i, c in enumerate(self.used):
            if self.used[i] == 0:
                self.concat.append(i)
        out_hw = min([shape[0] for i, shape in enumerate(layers) if self.used[i] == 0])
        self.out_shape = [out_hw, out_hw, out_filters * len(self.concat)]
    
    def forward(self, s0, s1, step):
        s0, s1 = self.maybe_calibrate_size(s0, s1)
        s1 = self.layer_base(s1)
        states = [s0, s1]
        for i in range(self.num_nodes):
            x_id = self.arch[4*i]
            y_id = self.arch[4*i+2]
            x = states[x_id]
            y = states[y_id]
            out = self.ops[i](x, y, step)
            states.append(out)
        return torch.cat([states[i] for i in self.concat], dim=1)
        

class NASNetwork(nn.Module):
    def __init__(self, num_layers, num_nodes, out_filters, keep_prob, drop_path_keep_prob, use_aux_head, num_steps, arch):
        super(NASNetwork, self).__init__()
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.out_filters = out_filters
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.num_steps = num_steps
        arch = list(map(int, arch.strip().split()))
        self.conv_arch = arch[:4 * self.num_nodes]
        self.reduc_arch = arch[4 * self.num_nodes:]
        self.criterion = nn.CrossEntropyLoss().cuda()
        
        self.num_layers = self.num_layers * 3
        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance, 2 * pool_distance + 1]
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1] #+ 1
        stem_multiplier = 3
        out_filters = stem_multiplier * self.out_filters
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_filters)
        )
        layers = [[32, 32, out_filters],[32, 32, out_filters]]
        out_filters = self.out_filters
        self.cells = nn.ModuleList()
        for i in range(self.num_layers+2):
            if i not in self.pool_layers:
                cell = Cell(self.conv_arch, layers, out_filters, False, i, self.num_layers+2, self.num_steps, self.drop_path_keep_prob)
            else:
                out_filters *= 2
                cell = Cell(self.reduc_arch, layers, out_filters, True, i, self.num_layers+2, self.num_steps, self.drop_path_keep_prob)
            self.cells.append(cell)
            layers = [layers[-1], cell.out_shape]
            
            if self.use_aux_head and i == self.aux_head_index:
                self.aux_head = AuxHead(layers[-1][-1])
        
        self.relu = nn.ReLU(inplace=False)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(layers[-1][-1], 10)
        
    
    def forward(self, input, step):
        aux_logits = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step)
            if i == self.aux_head_index and self.use_aux_head and self.training:
                aux_logits = self.aux_head(s1)
        out = self.relu(s1)
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits
    
    def loss(self, logits, target):
        return self.criterion(logits, target)
