import os
import sys
import time
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import OPERATIONS, ReLUConvBN, MaybeCalibrateSize, FactorizedReduce, AuxHeadCIFAR, AuxHeadImageNet, apply_drop_path, FinalCombine
from torch import Tensor
import utils
    

class Node(nn.Module):
    def __init__(self, x_id, x_op, y_id, y_op, x_shape, y_shape, channels, stride=1, drop_path_keep_prob=None,
                 layer_id=0, layers=0, steps=0):
        super(Node, self).__init__()
        self.channels = channels
        self.stride = stride
        self.drop_path_keep_prob = drop_path_keep_prob
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.x_id = x_id
        self.y_id = y_id
        x_shape = list(x_shape)
        y_shape = list(y_shape)
        
        x_stride = stride if x_id in [0, 1] else 1
        if x_op == 0:
            self.x_op = OPERATIONS[x_op](channels, channels, 3, x_stride, 1)
            x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
        elif x_op == 1:
            self.x_op = OPERATIONS[x_op](channels, channels, 5, x_stride, 2)
            x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
        elif x_op == 2:
            self.x_op = OPERATIONS[x_op](3, stride=x_stride, padding=1, count_include_pad=False)
            x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, x_shape[-1]]
        elif x_op == 3:
            self.x_op = OPERATIONS[x_op](3, stride=x_stride, padding=1)
            x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, x_shape[-1]]
        elif x_op == 4:
            self.x_op = OPERATIONS[x_op]()
            if x_stride > 1:
                assert x_stride == 2
                self.x_op.add_module('id_fact_reduce', FactorizedReduce(x_shape[-1], channels))
                x_shape = [x_shape[0] // x_stride, x_shape[1] // x_stride, channels]
        
        y_stride = stride if y_id in [0, 1] else 1
        if y_op == 0:
            self.y_op = OPERATIONS[y_op](channels, channels, 3, y_stride, 1)
            y_shape = [y_shape[0] // y_stride, y_shape[1] // y_stride, channels]
        elif y_op == 1:
            self.y_op = OPERATIONS[y_op](channels, channels, 5, y_stride, 2)
            y_shape = [y_shape[0] // y_stride, y_shape[1] // y_stride, channels]
        elif y_op == 2:
            self.y_op = OPERATIONS[y_op](3, stride=y_stride, padding=1, count_include_pad=False)
            y_shape = [y_shape[0] // y_stride, y_shape[1] // y_stride, y_shape[-1]]
        elif y_op == 3:
            self.y_op = OPERATIONS[y_op](3, stride=y_stride, padding=1)
            y_shape = [y_shape[0] // y_stride, y_shape[1] // y_stride, y_shape[-1]]
        elif y_op == 4:
            self.y_op = OPERATIONS[y_op]()
            if y_stride > 1:
                assert y_stride == 2
                self.y_op.add_module('id_fact_reduce', FactorizedReduce(y_shape[-1], channels))
                y_shape = [y_shape[0] // y_stride, y_shape[1] // y_stride, channels]
        
        assert x_shape[0] == y_shape[0] and x_shape[1] == y_shape[1]
        self.out_shape = list(x_shape)
        
    def forward(self, x, y, step):
        x = self.x_op(x)
        if self.x_id in [0, 1, 2, 3] and self.drop_path_keep_prob is not None and self.training:
            x = apply_drop_path(x, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
        y = self.y_op(y)
        if self.y_id in [0, 1, 2, 3] and self.drop_path_keep_prob is not None and self.training:
            y = apply_drop_path(y, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
        #print(x.shape,y.shape,self.x_id,self.y_id,self.x_op,self.y_op)
        out = x + y
        return out
    

class Cell(nn.Module):
    def __init__(self, arch, prev_layers, channels, reduction, layer_id, layers, steps, drop_path_keep_prob=None):
        super(Cell, self).__init__()
        print(prev_layers)
        assert len(prev_layers) == 2
        self.arch = arch
        self.reduction = reduction
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleList()
        self.nodes = len(arch) // 4
        self.used = [0] * (self.nodes + 2)
        
        # maybe calibrate size
        prev_layers = [list(prev_layers[0]), list(prev_layers[1])]
        self.maybe_calibrate_size = MaybeCalibrateSize(prev_layers, channels)
        prev_layers = self.maybe_calibrate_size.out_shape
        self.layer_base = ReLUConvBN(prev_layers[1][-1], channels, 1, 1, 0)
        
        stride = 2 if self.reduction else 1
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4*i], arch[4*i+1], arch[4*i+2], arch[4*i+3]
            x_shape, y_shape = prev_layers[x_id], prev_layers[y_id]
            node = Node(x_id, x_op, y_id, y_op, x_shape, y_shape, channels, stride, drop_path_keep_prob, layer_id, layers, steps)
            self.ops.append(node)
            self.used[x_id] += 1
            self.used[y_id] += 1
            prev_layers.append(node.out_shape)
        
        self.concat = []
        for i, c in enumerate(self.used):
            if self.used[i] == 0:
                self.concat.append(i)
        out_hw = min([shape[0] for i, shape in enumerate(prev_layers) if self.used[i] == 0])
        self.final_combine = FinalCombine(prev_layers, out_hw, channels, self.concat)
        self.out_shape = [out_hw, out_hw, channels * len(self.concat)]
    
    def forward(self, s0, s1, step):
        s0, s1 = self.maybe_calibrate_size(s0, s1)
        s1 = self.layer_base(s1)
        states = [s0, s1]
        for i in range(self.nodes):
            x_id = self.arch[4*i]
            y_id = self.arch[4*i+2]
            x = states[x_id]
            y = states[y_id]
            out = self.ops[i](x, y, step)
            states.append(out)
        return self.final_combine(states)
        

class NASNetworkCIFAR(nn.Module):
    def __init__(self, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps, arch):
        super(NASNetworkCIFAR, self).__init__()
        self.classes = classes
        self.layers = layers
        self.nodes = nodes
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        arch = list(map(int, arch.strip().split()))
        self.conv_arch = arch[:4 * self.nodes]
        self.reduc_arch = arch[4 * self.nodes:]

        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.layers = self.layers * 3
        
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1] #+ 1
        stem_multiplier = 3
        channels = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        outs = [[32, 32, channels],[32, 32, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.layers+2):
            if i not in self.pool_layers:
                cell = Cell(self.conv_arch, outs, channels, False, i, self.layers+2, self.steps, self.drop_path_keep_prob)
            else:
                channels *= 2
                cell = Cell(self.reduc_arch, outs, channels, True, i, self.layers+2, self.steps, self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
            
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadCIFAR(outs[-1][-1], classes)
        
        #self.relu = nn.ReLU(inplace=False)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], 10)
        
        self.init_parameters()
    
    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() == 4:
                nn.init.kaiming_normal(w.data)
    
    def forward(self, input, step=None):
        aux_logits = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1)
        out = s1
        #out = self.relu(s1)
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits


class NASNetworkImageNet(nn.Module):
    def __init__(self, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps, arch):
        super(NASNetworkImageNet, self).__init__()
        self.classes = classes
        self.layers = layers
        self.nodes = nodes
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        arch = list(map(int, arch.strip().split()))
        self.conv_arch = arch[:4 * self.nodes]
        self.reduc_arch = arch[4 * self.nodes:]
        
        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.layers = self.layers * 3
        
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1]

        channels = self.channels
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        outs = [[112, 112, channels // 2], [56, 56, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.layers + 2):
            if i not in self.pool_layers:
                cell = Cell(self.conv_arch, outs, channels, False, i, self.layers + 2, self.steps,
                            self.drop_path_keep_prob)
            else:
                channels *= 2
                cell = Cell(self.reduc_arch, outs, channels, True, i, self.layers + 2, self.steps,
                            self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
            
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadImageNet(outs[-1][-1], classes)
        
        #self.relu = nn.ReLU(inplace=False)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], 10)
        
        self.init_parameters()
    
    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() == 4:
                nn.init.kaiming_normal(w.data)
    
    def forward(self, input, step=None):
        aux_logits = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1)
        
        out = s1
        #out = self.relu(s1)
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits
