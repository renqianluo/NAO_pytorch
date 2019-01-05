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
    def __init__(self, x_id, x_op, y_id, y_op, x_c, y_c, C_curr, stride=1, drop_path_keep_prob=None,
                 layer_id=0, num_layers=0, num_steps=0):
        super(Node, self).__init__()
        self.C = C_curr
        self.stride = stride
        self.drop_path_keep_prob = drop_path_keep_prob
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.x_id = x_id
        self.y_id = y_id
        self.x_op = nn.Sequential(OPERATIONS[x_op](C_curr, stride, True))
        self.y_op = nn.Sequential(OPERATIONS[y_op](C_curr, stride, True))
        
        x_stride = stride if x_id in [0, 1] else 1
        if x_op in [0, 1]:
            pass
        elif x_op in [2, 3]:
            if x_c != C_curr:
                self.x_op.add_module('pool_conv', ReLUConvBN(x_c, C_curr, 1, 1, 0))
        else:
            if x_stride  > 1:
                assert x_stride == 2
                self.x_op.add_module('id_fact_reduce', FactorizedReduce(x_c, C_curr))
            if x_c != C_curr:
                self.x_op.add_module('id_conv', ReLUConvBN(x_c, C_curr, 1, 1, 0))

        y_stride = stride if y_id in [0, 1] else 1
        if y_op in [0, 1]:
            pass
        elif y_op in [2, 3]:
            if y_c != C_curr:
                self.y_op.add_module('pool_conv', ReLUConvBN(y_c, C_curr, 1, 1, 0))
        else:
            if y_stride > 1:
                assert y_stride == 2
                self.y_op.add_module('id_fact_reduce', FactorizedReduce(y_c, C_curr))
            if y_c != C_curr:
                self.y_op.add_module('id_conv', ReLUConvBN(y_c, C_curr, 1, 1, 0))
        
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
    def __init__(self, arch, C_prev_prev, C_prev, C, reduction, reduction_prev, layer_id, num_layers, num_steps, drop_path_keep_prob=None):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)
        self.arch = arch
        self.reduction = reduction
        self.reduction_prev = reduction_prev
        self.layer_id = layer_id
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleList()
        self.num_nodes = len(arch) // 4
        self.used = [0] * (self.num_nodes + 2)
        
        # maybe calibrate size
        self.preprocess_x, self.preprocess_y = None, None
        if reduction_prev: # hw[0] != hw[1]
            self.preprocess_x = nn.Sequential(nn.ReLU(), FactorizedReduce(C_prev_prev, C))
        elif C_prev_prev != C:
            self.preprocess_x = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        if C_prev != C:
            self.preprocess_y = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        stride = 2 if self.reduction else 1
        for i in range(self.num_nodes):
            x_id, x_op, y_id, y_op = arch[4*i], arch[4*i+1], arch[4*i+2], arch[4*i+3]
            node = Node(x_id, x_op, y_id, y_op, C, C, C, stride, drop_path_keep_prob, layer_id, num_layers, num_steps)
            self.ops.append(node)
            self.used[x_id] += 1
            self.used[y_id] += 1
        self.concat = []
        for i, c in enumerate(self.used):
            if self.used[i] == 0:
                self.concat.append(i)

    
    def forward(self, s0, s1, step):
        if self.preprocess_x is not None:
            s0 = self.preprocess_x(s0)
        if self.preprocess_y is not None:
            s1 = self.preprocess_y(s1)
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
        self.C = out_filters
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
        C_curr = stem_multiplier * self.C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(self.num_layers+2):
            if i not in self.pool_layers:
                cell = Cell(self.conv_arch, C_prev_prev, C_prev, C_curr, False, reduction_prev, i, self.num_layers+2, self.num_steps, self.drop_path_keep_prob)
                reduction_prev = False
            else:
                C_curr *= 2
                cell = Cell(self.reduc_arch, C_prev_prev, C_prev, C_curr, True, reduction_prev, i, self.num_layers+2, self.num_steps, self.drop_path_keep_prob)
                reduction_prev = True
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, C_curr
            
            if self.use_aux_head and i == self.aux_head_index:
                self.aux_head = AuxHead(C_curr)
        
        self.relu = nn.ReLU(inplace=False)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(C_prev, 10)
        
    
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
