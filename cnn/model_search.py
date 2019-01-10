import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *


class Node(nn.Module):
    def __init__(self, prev_layers, channels, stride=1, drop_path_keep_prob=None, node_id=0, layer_id=0, layers=0, steps=0):
        super(Node, self).__init__()
        self.channels = channels
        self.stride = stride
        self.drop_path_keep_prob = drop_path_keep_prob
        self.node_id = node_id
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.x_op = nn.ModuleList()
        self.y_op = nn.ModuleList()
        
        num_possible_inputs = node_id + 2
        
        # avg_pool
        self.x_avg_pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        # max_pool
        self.x_max_pool = nn.MaxPool2d(3, stride=1, padding=1)
        # sep_conv
        self.x_sep_conv_3 = WSSepConv(num_possible_inputs, channels, channels, 3, 1, 1)
        self.x_sep_conv_5 = WSSepConv(num_possible_inputs, channels, channels, 5, 1, 2)

        # avg_pool
        self.y_avg_pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        # max_pool
        self.y_max_pool = nn.MaxPool2d(3, stride=1, padding=1)
        # sep_conv
        self.y_sep_conv_3 = WSSepConv(num_possible_inputs, channels, channels, 3, 1, 1)
        self.y_sep_conv_5 = WSSepConv(num_possible_inputs, channels, channels, 5, 1, 2)
            
        self.out_shape = [prev_layers[0][0], prev_layers[0][1], channels]
        
    def forward(self, x, x_id, x_op, y, y_id, y_op, step):
        if x_op == 0:
            if x.size(1) != self.channels:
                x = self.x_conv(x, x_id)
            x = self.x_sep_conv_3(x, x_id)
        elif x_op == 1:
            if x.size(1) != self.channels:
                x = self.x_conv(x, x_id)
            x = self.x_sep_conv_5(x, x_id)
        elif x_op == 2:
            x = self.x_avg_pool(x)
            if x.size(1) != self.channels:
                x = self.x_avg_pool_conv(x, x_id)
        elif x_op == 3:
            x = self.x_max_pool(x)
            if x.size(1) != self.channels:
                x = self.x_max_pool_conv(x, x_id)
        else:
            assert x_op == 4
            if x.size(1) != self.channels:
                x = self.x_conv(x, x_id)
        
        if y_op == 0:
            if y.size(1) != self.channels:
                y = self.y_conv(y, y_id)
            y = self.y_sep_conv_3(y, y_id)
        elif y_op == 1:
            if y.size(1) != self.channels:
                y = self.y_conv(y, y_id)
            y = self.y_sep_conv_5(y, y_id)
        elif y_op == 2:
            y = self.y_avg_pool(y)
            if y.size(1) != self.channels:
                y = self.y_avg_pool_conv(y, y_id)
        elif y_op == 3:
            y = self.y_max_pool(y)
            if y.size(1) != self.channels:
                y = self.y_max_pool_conv(y, y_id)
        else:
            assert y_op == 4
            if y.size(1) != self.channels:
                y = self.x_conv(y, y_id)
         
        if x_op != 4 and self.drop_path_keep_prob is not None and self.training:
            x = apply_drop_path(x, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
        if y_op != 4 and self.drop_path_keep_prob is not None and self.training:
            y = apply_drop_path(y, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
            
        return x + y


class Cell(nn.Module):
    def __init__(self, prev_layers, nodes, channels, reduction, layer_id, layers, steps, drop_path_keep_prob=None):
        super(Cell, self).__init__()
        assert len(prev_layers) == 2
        print(prev_layers)
        self.reduction = reduction
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleList()
        self.nodes = nodes
        
        # maybe calibrate size
        prev_layers = [list(prev_layers[0]), list(prev_layers[1])]
        self.maybe_calibrate_size = MaybeCalibrateSize(prev_layers, channels)
        prev_layers = self.maybe_calibrate_size.out_shape

        stride = 2 if self.reduction else 1
        for i in range(self.nodes):
            node = Node(prev_layers, channels, stride, drop_path_keep_prob, i, layer_id, layers, steps)
            self.ops.append(node)
            prev_layers.append(node.out_shape)
        out_hw = min([shape[0] for i, shape in enumerate(prev_layers)])
        
        self.final_combine_conv = WSReLUConvBN(self.nodes+2, channels, channels, 1, 1, 0, False)
        
        self.out_shape = [out_hw, out_hw, channels]
        
    def forward(self, s0, s1, arch, step):
        s0, s1 = self.maybe_calibrate_size(s0, s1)
        states = [s0, s1]
        used = [0] * (self.nodes + 2)
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4*i], arch[4*i+1], arch[4*i+2], arch[4*i+3]
            used[x_id] += 1
            used[y_id] += 1
            out = self.ops[i](states[x_id], x_id, x_op, states[y_id], y_id, y_op, step)
            states.append(out)
        concat = []
        for i, c in enumerate(used):
            if used[i] == 0:
                concat.append(i)
                
        out = torch.cat([states[i] for i in concat], dim=1)
        out = self.final_combine_conv(out, concat)
        return out
    

class NASNetwork(nn.Module):
    def __init__(self, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps):
        super(NASNetwork, self).__init__()
        self.layers = layers
        self.nodes = nodes
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps

        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.total_layers = self.layers * 3 + 2
       
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1] #+ 1
        stem_multiplier = 3
        channels = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        outs = [[32, 32, channels], [32, 32, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        self.fac_recs = nn.ModuleList()
        for i in range(self.total_layers):
            if i not in self.pool_layers:
                cell = Cell(outs, self.nodes, channels, False, i, self.total_layers, self.steps, self.drop_path_keep_prob)
                outs = [outs[-1], cell.out_shape]
            else:
                channels *= 2
                fac_rec = FactorizedReduce(outs[-1][-1], channels)
                self.fac_recs.append(fac_rec)
                outs = [outs[-1], [outs[-1][0]//2, outs[-1][1]//2, channels]]
                cell = Cell(outs, self.nodes, channels, True, i, self.total_layers, self.steps, self.drop_path_keep_prob)
                outs = [outs[-1], cell.out_shape]
            self.cells.append(cell)
            
            if self.use_aux_head and i == self.aux_head_index:
                self.aux_head = AuxHead(outs[-1][-1])

        self.relu = nn.ReLU(inplace=False)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], 10)

        self.init_parameters()
        
    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal(w.data)

    def new(self):
        model_new = NASNetwork(
            self.layers, self.nodes, self.channels, self.keep_prob, self.drop_path_keep_prob,
            self.use_aux_head, self.steps)
        for x, y in zip(model_new.parameters(), self.parameters()):
            x.data.copy_(y.data)
        return model_new
    
    def forward(self, input, arch, step=None):
        aux_logits = None
        conv_arch, reduc_arch = arch
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                assert i in self.pool_layers
                fac_rec = self.fac_recs[self.pool_layers.index(i)]
                s0, s1 = s1, fac_rec(s1)
                s0, s1 = s1, cell(s0, s1, reduc_arch, step)
            else:
                s0, s1 = s1, cell(s0, s1, conv_arch, step)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.aux_head(s1)
        out = self.relu(s1)
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits
    
    def loss(self, logits, target):
        return self.criterion(logits, target)
