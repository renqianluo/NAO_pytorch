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
    def __init__(self, C):
        super(Node, self).__init__()
        self.branch1 = nn.ModuleList()
        self.branch2 = nn.ModuleList()
        num_op = len(OPERATIONS)
        for op_id in range(num_op):
            op1 = OPERATIONS[op_id](C, 1, False)
            op2 = OPERATIONS[op_id](C, 1, False)
            #if op_id in [2,3]:
            #  op1 = nn.Sequential(op1, nn.BatchNorm2d(C, affine=False))
            #  op2 = nn.Sequential(op2, nn.BatchNorm2d(C, affine=False))
            self.branch1.append(op1)
            self.branch2.append(op2)

    def forward(self, x, x_op, y, y_op):
        return self.branch1[x_op](x) + self.branch2[y_op](y)


class Cell(nn.Module):
    def __init__(self, num_nodes, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.num_cells = num_nodes
        self.reduction = reduction
        #self.maybe_calibrate_size = MaybeCalibrateSize(C_prev_prev, C_prev, C)
        # bellow is same to maybecalibratesize in tf
        if reduction:
            self.preprocess_x = FactorizedReduce(C_prev_prev, C, affine=False)
        #else:
        #    self.preprocess_x = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        #self.preprocess_y = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self.ops = nn.ModuleList()
        for i in range(self.num_nodes):
            op = Node(C)
            self.ops.append(op)
    
    def forward(self, s0, s1, arch):
        if self.reduction:
            s0 = self.preprocess_x(s0)
        #s1 = self.preprocess_y(s1)
    
        assert len(arch) == 4 * self.num_nodes
        states = [s0, s1]
        for i in range(self.num_nodes):
            x_id, x_op, y_id, y_op = arch[4*i], arch[4*i+1], arch[4*i+2], arch[4*i+3]
            out = self.ops[i](states[x_id], x_op, states[y_id], y_op)
            states.append(out)
        return torch.cat(states, dim=1)
    

class NASNetwork(nn.Module):
    def __init__(self, num_layers, num_nodes, out_filters, keep_prob, drop_path_keep_prob, use_aux_head, num_epochs, num_train_batches):
        super(NASNetwork, self).__init__()
        logging.info("-" * 80)
        logging.info("Build model")
        self.num_layers = num_layers
        self.num_cells = num_nodes
        self.out_filters = out_filters
        self.keep_prob = keep_prob
        self.use_aux_head = use_aux_head
        self.num_epochs = num_epochs
        self.num_train_batches = num_train_batches
        self.num_train_steps = self.num_epochs * self.num_train_batches
        self.drop_path_keep_prob = drop_path_keep_prob
        self.criterion = nn.CrossEntropyLoss().cuda()
        
        if self.drop_path_keep_prob is not None:
            assert num_epochs is not None, "Need num_epochs to drop_path"
    
        self.num_layers = self.num_layers * 3
        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance, 2 * pool_distance + 1]
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1] #+ 1
        stem_multiplier = 3
        C_curr = stem_multiplier * self.out_filters
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
    
        self.cells = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.out_filters
        reduction_prev = False
        for i in range(self.num_layers+2):
            if i not in self.pool_layers:
                cell = Cell(self.num_nodes, C_prev_prev, C_prev, C_curr, False, reduction_prev)
                reduction_prev = False
            else:
                C_curr *= 2
                cell1 = FactorizedReduce(C_prev, C_curr, affine=False)
                C_prev_prev, C_prev = C_prev, C_curr
                cell2 = Cell(self.num_nodes, C_prev_prev, C_prev, C_curr, True, reduction_prev)
                reduction_prev = True
                cell = nn.Sequential(cell1, cell2)
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, C_curr
            
            if i == self.aux_head_index:
                self.aux_head = AuxHead(C_curr)

        self.relu = nn.ReLU(inplace=False)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(C_prev, 10)
    
    def forward(self, x, arch):
        aux_logits = None
        conv_arch, reduc_arch = arch
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                s0, s1 = s1, cell(s0, s1, reduc_arch)
            else:
                s0, s1 = s1, cell(s0, s1, conv_arch)
            if i == self.aux_head_index and self.use_aux_head and self.training:
                aux_logits = self.aux_head(s1)
        out = self.relu(s1)
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits
    
    def loss(self, logits, target):
        return self.criterion(logits, target)
    
    
def train(train_queue, model):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)
        model.optimizer.zero_grad()
        logits, aux_logits = model(input)
        loss = model.criterion(logits, target) + 0.4 * model.criterion(aux_logits, target)
        nn.utils.clip_grad_norm(model.parameters(), model.grad_clip)
        model.optimizer.step()
        prec1, prec5 = utils.accuracy(logits,target, topk=(1,5))
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)
  
        if step % 100 == 0 :
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

def valid(valid_queue, model, arch_pool):
    model.eval()
    arch_pool_valid_acc = []
    for arch in arch_pool:
        input, target = next(iter(valid_queue))
        input = Variable(input, volatile=True).cuda()
        logits, _ = model(input, arch)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        arch_pool_valid_acc.append(prec1)
    return arch_pool_valid_acc