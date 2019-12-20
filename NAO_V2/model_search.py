import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import OPERATIONS_search_small, OPERATIONS_search_middle, WSReLUConvBN, FactorizedReduce, MaybeCalibrateSize, AuxHeadCIFAR, AuxHeadImageNet, apply_drop_path


class Node(nn.Module):
    def __init__(self, search_space, prev_layers, channels, stride, drop_path_keep_prob=None, node_id=0, layer_id=0, layers=0, steps=0):
        super(Node, self).__init__()
        self.search_space = search_space
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
        
        if search_space == 'small':
            OPERATIONS = OPERATIONS_search_small
        elif search_space == 'middle':
            OPERATIONS = OPERATIONS_search_middle
        else:
            OPERATIONS = OPERATIONS_search_small
        
        for k, v in OPERATIONS.items():
            self.x_op.append(v(num_possible_inputs, channels, channels, stride, True))
            self.y_op.append(v(num_possible_inputs, channels, channels, stride, True))
 
        self.out_shape = [prev_layers[0][0]//stride, prev_layers[0][1]//stride, channels]
        
    def forward(self, x, x_id, x_op, y, y_id, y_op, step, bn_train=False):
        stride = self.stride if x_id in [0, 1] else 1
        x = self.x_op[x_op](x, x_id, stride, bn_train)
        stride = self.stride if y_id in [0, 1] else 1
        y = self.y_op[y_op](y, y_id, stride, bn_train)
        
        X_DROP = False
        Y_DROP = False
        if self.search_space == 'small':
            if x_op not in [4] and self.drop_path_keep_prob is not None and self.training:
                X_DROP = True
            if y_op not in [4] and self.drop_path_keep_prob is not None and self.training:
                Y_DROP = True
        elif self.search_space == 'middle':
            if x_op not in [0, 1] and self.drop_path_keep_prob is not None and self.training:
                X_DROP = True
            if y_op not in [0, 1] and self.drop_path_keep_prob is not None and self.training:
                Y_DROP = True
        if X_DROP:
            x = apply_drop_path(x, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
        if Y_DROP:
            y = apply_drop_path(y, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
            
        return x + y


class Cell(nn.Module):
    def __init__(self, search_space, prev_layers, nodes, channels, reduction, layer_id, layers, steps, drop_path_keep_prob=None):
        super(Cell, self).__init__()
        self.search_space = search_space
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
            node = Node(search_space, prev_layers, channels, stride, drop_path_keep_prob, i, layer_id, layers, steps)
            self.ops.append(node)
            prev_layers.append(node.out_shape)
        out_hw = min([shape[0] for i, shape in enumerate(prev_layers)])

        if reduction:
            self.fac_1 = FactorizedReduce(prev_layers[0][-1], channels, prev_layers[0])
            self.fac_2 = FactorizedReduce(prev_layers[1][-1], channels, prev_layers[1])
        self.final_combine_conv = WSReLUConvBN(self.nodes+2, channels, channels, 1)
        
        self.out_shape = [out_hw, out_hw, channels]
        
    def forward(self, s0, s1, arch, step, bn_train=False):
        s0, s1 = self.maybe_calibrate_size(s0, s1, bn_train=bn_train)
        states = [s0, s1]
        used = [0] * (self.nodes + 2)
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4*i], arch[4*i+1], arch[4*i+2], arch[4*i+3]
            used[x_id] += 1
            used[y_id] += 1
            out = self.ops[i](states[x_id], x_id, x_op, states[y_id], y_id, y_op, step, bn_train=bn_train)
            states.append(out)
        concat = []
        for i, c in enumerate(used):
            if used[i] == 0:
                concat.append(i)
        
        # Notice that in reduction cell, 0, 1 might be concated and they might have to be factorized
        if self.reduction:
            if 0 in concat:
                states[0] = self.fac_1(states[0])
            if 1 in concat:
                states[1] = self.fac_2(states[1])
        out = torch.cat([states[i] for i in concat], dim=1)
        out = self.final_combine_conv(out, concat, bn_train=bn_train)
        return out
    

class NASWSNetworkCIFAR(nn.Module):
    def __init__(self, args, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps):
        super(NASWSNetworkCIFAR, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.classes = classes
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
            self.aux_head_index = self.pool_layers[-1]
        stem_multiplier = 3
        channels = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        outs = [[32, 32, channels], [32, 32, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.total_layers):
            # normal cell
            if i not in self.pool_layers:
                cell = Cell(self.search_space, outs, self.nodes, channels, False, i, self.total_layers, self.steps, self.drop_path_keep_prob)
            # reduction cell
            else:
                channels *= 2
                cell = Cell(self.search_space, outs, self.nodes, channels, True, i, self.total_layers, self.steps, self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
            
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadCIFAR(outs[-1][-1], classes)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)

        self.init_parameters()
        
    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def new(self):
        model_new = NASWSNetworkCIFAR(
            self.search_space, self.classes, self.layers, self.nodes, self.channels,
            self.keep_prob, self.drop_path_keep_prob, self.use_aux_head, self.steps)
        for x, y in zip(model_new.parameters(), self.parameters()):
            x.data.copy_(y.data)
        return model_new
    
    def forward(self, input, arch, step=None, bn_train=False):
        aux_logits = None
        conv_arch, reduc_arch = arch
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                assert i in self.pool_layers
                s0, s1 = s1, cell(s0, s1, reduc_arch, step, bn_train=bn_train)
            else:
                s0, s1 = s1, cell(s0, s1, conv_arch, step, bn_train=bn_train)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1, bn_train=bn_train)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits


class NASWSNetworkImageNet(nn.Module):
    def __init__(self, args, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps):
        super(NASWSNetworkImageNet, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.classes = classes
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
            self.aux_head_index = self.pool_layers[-1]
        
        channels = self.channels
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        
        outs = [[112, 112, channels // 2], [56, 56, channels]]
        self.cells = nn.ModuleList()
        for i in range(self.total_layers):
            if i not in self.pool_layers:
                cell = Cell(self.search_space, outs, self.nodes, channels, False, i, self.total_layers, self.steps,
                            self.drop_path_keep_prob)
                outs = [outs[-1], cell.out_shape]
            else:
                channels *= 2
                cell = Cell(self.search_space, outs, self.nodes, channels, True, i, self.total_layers, self.steps,
                            self.drop_path_keep_prob)
                outs = [outs[-1], cell.out_shape]
            self.cells.append(cell)
            
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadImageNet(outs[-1][-1], classes)
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)
        
        self.init_parameters()
    
    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)
    
    def new(self):
        model_new = NASWSNetworkImageNet(
            self.search_space, self.classes, self.layers, self.nodes, self.channels, 
            self.keep_prob, self.drop_path_keep_prob,self.use_aux_head, self.steps)
        for x, y in zip(model_new.parameters(), self.parameters()):
            x.data.copy_(y.data)
        return model_new
    
    def forward(self, input, arch, step=None, bn_train=False):
        aux_logits = None
        conv_arch, reduc_arch = arch
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                assert i in self.pool_layers
                s0, s1 = s1, cell(s0, s1, reduc_arch, step, bn_train=bn_train)
            else:
                s0, s1 = s1, cell(s0, s1, conv_arch, step, bn_train=bn_train)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1, bn_train=bn_train)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits
