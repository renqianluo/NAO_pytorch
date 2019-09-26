import os
import sys
import glob
import time
import copy
import random
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model import NASNetworkCIFAR, NASNetworkImageNet
from model_search import NASWSNetworkCIFAR, NASWSNetworkImageNet
from operations import OPERATIONS_search_small, OPERATIONS_search_middle
from controller import NAO

parser = argparse.ArgumentParser(description='NAO Search')

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100, imagenet'])
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--search_space', type=str, default='small', choices=['small', 'middle'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--child_batch_size', type=int, default=64)
parser.add_argument('--child_eval_batch_size', type=int, default=500)
parser.add_argument('--child_epochs', type=int, default=50)
parser.add_argument('--child_layers', type=int, default=3)
parser.add_argument('--child_nodes', type=int, default=5)
parser.add_argument('--child_channels', type=int, default=20)
parser.add_argument('--child_cutout_size', type=int, default=None)
parser.add_argument('--child_grad_bound', type=float, default=5.0)
parser.add_argument('--child_lr_max', type=float, default=0.025)
parser.add_argument('--child_lr_min', type=float, default=0.001)
parser.add_argument('--child_keep_prob', type=float, default=1.0)
parser.add_argument('--child_drop_path_keep_prob', type=float, default=0.9)
parser.add_argument('--child_l2_reg', type=float, default=3e-4)
parser.add_argument('--child_use_aux_head', action='store_true', default=False)
parser.add_argument('--child_arch_pool', type=str, default=None)
parser.add_argument('--child_lr', type=float, default=0.1)
parser.add_argument('--child_label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--child_gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--child_decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--controller_seed_arch', type=int, default=1000)
parser.add_argument('--controller_expand', type=int, default=10)
parser.add_argument('--controller_new_arch', type=int, default=300)
parser.add_argument('--controller_encoder_layers', type=int, default=1)
parser.add_argument('--controller_encoder_hidden_size', type=int, default=64)
parser.add_argument('--controller_encoder_emb_size', type=int, default=32)
parser.add_argument('--controller_mlp_layers', type=int, default=0)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)
parser.add_argument('--controller_decoder_layers', type=int, default=1)
parser.add_argument('--controller_decoder_hidden_size', type=int, default=64)
parser.add_argument('--controller_source_length', type=int, default=40)
parser.add_argument('--controller_encoder_length', type=int, default=20)
parser.add_argument('--controller_decoder_length', type=int, default=40)
parser.add_argument('--controller_encoder_dropout', type=float, default=0)
parser.add_argument('--controller_mlp_dropout', type=float, default=0.1)
parser.add_argument('--controller_decoder_dropout', type=float, default=0)
parser.add_argument('--controller_l2_reg', type=float, default=0)
parser.add_argument('--controller_encoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_decoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_epochs', type=int, default=1000)
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_optimizer', type=str, default='adam')
parser.add_argument('--controller_grad_bound', type=float, default=5.0)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def get_builder(dataset):
    if dataset == 'cifar10':
        return build_cifar10
    elif dataset == 'cifar100':
        return build_cifar100
    else:
        return build_imagenet
    

def build_cifar10(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    epoch = kwargs.pop('epoch')
    ratio = kwargs.pop('ratio')
    train_transform, valid_transform = utils._data_transforms_cifar10(args.child_cutout_size)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)

    num_train = len(train_data)
    assert num_train == len(valid_data)
    indices = list(range(num_train))    
    split = int(np.floor(ratio * num_train))
    np.random.shuffle(indices)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.child_eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=16)
    
    model = NASWSNetworkCIFAR(args, 10, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                       args.child_use_aux_head, args.steps)
    model = model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.child_epochs, args.child_lr_min, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def build_cifar100(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    epoch = kwargs.pop('epoch')
    ratio = kwargs.pop('ratio')
    train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_size)
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=valid_transform)

    num_train = len(train_data)
    assert num_train == len(valid_data)
    indices = list(range(num_train))    
    split = int(np.floor(ratio * num_train))
    np.random.shuffle(indices)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.child_eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=16)
    
    model = NASWSNetworkCIFAR(args, 100, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                       args.child_use_aux_head, args.steps)
    model = model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.child_epochs, args.child_lr_min, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def build_imagenet(model_state_dict, optimizer_state_dict, **kwargs):
    ratio = kwargs.pop('ratio')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if args.zip_file:
        logging.info('Loading data from zip file')
        traindir = os.path.join(args.data, 'train.zip')
        if args.lazy_load:
            train_data = utils.ZipDataset(traindir, train_transform)
        else:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryZipDataset(traindir, train_transform, num_workers=32)
    else:
        logging.info('Loading data from directory')
        traindir = os.path.join(args.data, 'train')
        if args.lazy_load:
            train_data = dset.ImageFolder(traindir, train_transform)
        else:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryDataset(traindir, train_transform, num_workers=32)
       
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(ratio * num_train))
    train_indices = sorted(indices[:split])
    valid_indices = sorted(indices[split:])

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_indices),
        pin_memory=True, num_workers=16)
    
    model = NASWSNetworkImageNet(args, 1000, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob,
                       args.child_drop_path_keep_prob, args.child_use_aux_head, args.steps)
    model = model.cuda()
    train_criterion = CrossEntropyLabelSmooth(1000, args.child_label_smooth).cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.child_decay_period, gamma=args.child_gamma)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def get_scheduler(optimizer, dataset):
    if 'cifar' in dataset:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.child_epochs, args.child_lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.child_decay_period, gamma=args.child_gamma)
    return scheduler


def child_train(train_queue, model, optimizer, global_step, arch_pool, arch_pool_prob, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()

        optimizer.zero_grad()
        # sample an arch to train
        arch = utils.sample_arch(arch_pool, arch_pool_prob)
        logits, aux_logits = model(input, arch, global_step)
        global_step += 1
        loss = criterion(logits, target)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, target)
            loss += 0.4 * aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
        optimizer.step()
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        
        if (step+1) % 100 == 0:
            logging.info('Train %03d loss %e top1 %f top5 %f', step+1, objs.avg, top1.avg, top5.avg)
            logging.info('Arch: %s', ' '.join(map(str, arch[0] + arch[1])))

    return top1.avg, objs.avg, global_step


def child_valid(valid_queue, model, arch_pool, criterion):
    valid_acc_list = []
    with torch.no_grad():
        model.eval()
        for i, arch in enumerate(arch_pool):
            # for step, (input, target) in enumerate(valid_queue):
            inputs, targets = next(iter(valid_queue))
            inputs = inputs.cuda()
            targets = targets.cuda()
                
            logits, _ = model(inputs, arch, bn_train=True)
            loss = criterion(logits, targets)
                
            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            valid_acc_list.append(prec1.data/100)
            
            if (i+1) % 100 == 0:
                logging.info('Valid arch %s\n loss %.2f top1 %f top5 %f', ' '.join(map(str, arch[0] + arch[1])), loss, prec1, prec5)
        
    return valid_acc_list


def train_and_evaluate_top_on_cifar10(archs, train_queue, valid_queue):
    res = []
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for i, arch in enumerate(archs):
        objs.reset()
        top1.reset()
        top5.reset()
        logging.info('Train and evaluate the {} arch'.format(i+1))
        model = NASNetworkCIFAR(args, 10, args.child_layers, args.child_nodes, args.child_channels, 0.6, 0.8,
                        True, args.steps, arch)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr_max,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, args.child_lr_min)
        global_step = 0
        for e in range(10):
            scheduler.step()
            for step, (input, target) in enumerate(train_queue):
                input = input.cuda().requires_grad_()
                target = target.cuda()

                optimizer.zero_grad()
                # sample an arch to train
                logits, aux_logits = model(input, global_step)
                global_step += 1
                loss = train_criterion(logits, target)
                if aux_logits is not None:
                    aux_loss = train_criterion(aux_logits, target)
                    loss += 0.4 * aux_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
                optimizer.step()
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('Train epoch %03d %03d loss %e top1 %f top5 %f', e+1, step+1, objs.avg, top1.avg, top5.avg)
        objs.reset()
        top1.reset()
        top5.reset()
        with torch.no_grad():
            model.eval()
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()
            
                logits, _ = model(input)
                loss = eval_criterion(logits, target)
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)
        res.append(top1.avg)
    return res


def train_and_evaluate_top_on_cifar100(archs, train_queue, valid_queue):
    res = []
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for i, arch in enumerate(archs):
        objs.reset()
        top1.reset()
        top5.reset()
        logging.info('Train and evaluate the {} arch'.format(i+1))
        model = NASNetworkCIFAR(args, 100, args.child_layers, args.child_nodes, args.child_channels, 0.6, 0.8,
                        True, args.steps, arch)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr_max,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, args.child_lr_min)
        global_step = 0
        for e in range(10):
            scheduler.step()
            for step, (input, target) in enumerate(train_queue):
                input = input.cuda().requires_grad_()
                target = target.cuda()

                optimizer.zero_grad()
                # sample an arch to train
                logits, aux_logits = model(input, global_step)
                global_step += 1
                loss = train_criterion(logits, target)
                if aux_logits is not None:
                    aux_loss = train_criterion(aux_logits, target)
                    loss += 0.4 * aux_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
                optimizer.step()
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('Train %3d %03d loss %e top1 %f top5 %f', e+1, step+1, objs.avg, top1.avg, top5.avg)
        objs.reset()
        top1.reset()
        top5.reset()
        with torch.no_grad():
            model.eval()
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()
            
                logits, _ = model(input)
                loss = eval_criterion(logits, target)
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)
        res.append(top1.avg)
    return res


def train_and_evaluate_top_on_imagenet(archs, train_queue, valid_queue):
    res = []
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for i, arch in enumerate(archs):
        objs.reset()
        top1.reset()
        top5.reset()
        logging.info('Train and evaluate the {} arch'.format(i+1))
        model = NASNetworkImageNet(args, 1000, args.child_layers, args.child_nodes, args.child_channels, 1.0, 1.0,
                        True, args.steps, arch)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )
        for step, (input, target) in enumerate(train_queue):
            input = input.cuda().requires_grad_()
            target = target.cuda()

            optimizer.zero_grad()
            # sample an arch to train
            logits, aux_logits = model(input, step)
            loss = train_criterion(logits, target)
            if aux_logits is not None:
                aux_loss = train_criterion(aux_logits, target)
                loss += 0.4 * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
            optimizer.step()
            
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
            
            if (step+1) % 100 == 0:
                logging.info('Train %03d loss %e top1 %f top5 %f', step+1, objs.avg, top1.avg, top5.avg)
            if step+1 == 500:
                break

        objs.reset()
        top1.reset()
        top5.reset()
        with torch.no_grad():
            model.eval()
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()
            
                logits, _ = model(input)
                loss = eval_criterion(logits, target)
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)
        res.append(top1.avg)
    return res


def nao_train(train_queue, model, optimizer):
    objs = utils.AvgrageMeter()
    mse = utils.AvgrageMeter()
    nll = utils.AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample['encoder_input']
        encoder_target = sample['encoder_target']
        decoder_input = sample['decoder_input']
        decoder_target = sample['decoder_target']
        
        encoder_input = encoder_input.cuda()
        encoder_target = encoder_target.cuda().requires_grad_()
        decoder_input = decoder_input.cuda()
        decoder_target = decoder_target.cuda()
        
        optimizer.zero_grad()
        predict_value, log_prob, arch = model(encoder_input, decoder_input)
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))
        loss = args.controller_trade_off * loss_1 + (1 - args.controller_trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.controller_grad_bound)
        optimizer.step()
        
        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)
        
    return objs.avg, mse.avg, nll.avg


def nao_valid(queue, model):
    inputs = []
    targets = []
    predictions = []
    archs = []
    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(queue):
            encoder_input = sample['encoder_input']
            encoder_target = sample['encoder_target']
            decoder_target = sample['decoder_target']
            
            encoder_input = encoder_input.cuda()
            encoder_target = encoder_target.cuda()
            decoder_target = decoder_target.cuda()
            
            predict_value, logits, arch = model(encoder_input)
            n = encoder_input.size(0)
            inputs += encoder_input.data.squeeze().tolist()
            targets += encoder_target.data.squeeze().tolist()
            predictions += predict_value.data.squeeze().tolist()
            archs += arch.data.squeeze().tolist()
    pa = utils.pairwise_accuracy(targets, predictions)
    hd = utils.hamming_distance(inputs, archs)
    return pa, hd


def nao_infer(queue, model, step, direction='+'):
    new_arch_list = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = sample['encoder_input']
        encoder_input = encoder_input.cuda()
        model.zero_grad()
        new_arch = model.generate_new_arch(encoder_input, step, direction=direction)
        new_arch_list.extend(new_arch.data.squeeze().tolist())
    return new_arch_list


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    if args.dataset == 'cifar10':
        args.num_class = 10
    elif args.dataset == 'cifar100':
        args.num_class = 100
    else:
        args.num_class = 10
    
    if args.search_space == 'small':
        OPERATIONS = OPERATIONS_search_small
    elif args.search_space == 'middle':
        OPERATIONS = OPERATIONS_search_middle
    args.child_num_ops = len(OPERATIONS)
    args.controller_encoder_vocab_size = 1 + ( args.child_nodes + 2 - 1 ) + args.child_num_ops
    args.controller_decoder_vocab_size = args.controller_encoder_vocab_size
    args.steps = int(np.ceil(45000 / args.child_batch_size)) * args.child_epochs

    logging.info("args = %s", args)
    
    if args.child_arch_pool is not None:
        logging.info('Architecture pool is provided, loading')
        with open(args.child_arch_pool) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            child_arch_pool = archs
    elif os.path.exists(os.path.join(args.output_dir, 'arch_pool')):
        logging.info('Architecture pool is founded, loading')
        with open(os.path.join(args.output_dir, 'arch_pool')) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            child_arch_pool = archs
    else:
        child_arch_pool = None

    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_fn(ratio=0.9, epoch=-1)

    nao = NAO(
        args.controller_encoder_layers,
        args.controller_encoder_vocab_size,
        args.controller_encoder_hidden_size,
        args.controller_encoder_dropout,
        args.controller_encoder_length,
        args.controller_source_length,
        args.controller_encoder_emb_size,
        args.controller_mlp_layers,
        args.controller_mlp_hidden_size,
        args.controller_mlp_dropout,
        args.controller_decoder_layers,
        args.controller_decoder_vocab_size,
        args.controller_decoder_hidden_size,
        args.controller_decoder_dropout,
        args.controller_decoder_length,
    )
    nao = nao.cuda()
    logging.info("Encoder-Predictor-Decoder param size = %fMB", utils.count_parameters_in_MB(nao))

    
    if child_arch_pool is None:
        logging.info('Architecture pool is not provided, randomly generating now')
        child_arch_pool = utils.generate_arch(args.controller_seed_arch, args.child_nodes, args.child_num_ops)  # [[[conv],[reduc]]]
    arch_pool = []
    arch_pool_valid_acc = []
    for i in range(4):
        logging.info('Iteration %d', i)

        child_arch_pool_prob = []
        for arch in child_arch_pool:
            if args.dataset == 'cifar10':
                tmp_model = NASNetworkCIFAR(args, args.num_class, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                    args.child_use_aux_head, args.steps, arch)
            elif args.dataset == 'cifar100':
                tmp_model = NASNetworkCIFAR(args, args.num_class, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                    args.child_use_aux_head, args.steps, arch)
            else:
                tmp_model = NASNetworkImageNet(args, args.num_class, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob,
                    args.child_drop_path_keep_prob, args.child_use_aux_head, args.steps, arch)
            child_arch_pool_prob.append(utils.count_parameters_in_MB(tmp_model))
            del tmp_model
        
        step = 0
        scheduler = get_scheduler(optimizer, args.dataset)
        for epoch in range(1, args.child_epochs + 1):
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            # sample an arch to train
            train_acc, train_obj, step = child_train(train_queue, model, optimizer, step, child_arch_pool, child_arch_pool_prob, train_criterion)
            logging.info('train_acc %f', train_acc)

        logging.info("Evaluate seed archs")
        arch_pool += child_arch_pool
        arch_pool_valid_acc = child_valid(valid_queue, model, arch_pool, eval_criterion)

        arch_pool_valid_acc_sorted_indices = np.argsort(arch_pool_valid_acc)[::-1]
        arch_pool = [arch_pool[i] for i in arch_pool_valid_acc_sorted_indices]
        arch_pool_valid_acc = [arch_pool_valid_acc[i] for i in arch_pool_valid_acc_sorted_indices]
        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(i)), 'w') as fa:
            with open(os.path.join(args.output_dir, 'arch_pool.perf.{}'.format(i)), 'w') as fp:
                for arch, perf in zip(arch_pool, arch_pool_valid_acc):
                    arch = ' '.join(map(str, arch[0] + arch[1]))
                    fa.write('{}\n'.format(arch))
                    fp.write('{}\n'.format(perf))
        if i == 3:
            break
                            
        # Train Encoder-Predictor-Decoder
        logging.info('Train Encoder-Predictor-Decoder')
        encoder_input = list(map(lambda x: utils.parse_arch_to_seq(x[0]) + utils.parse_arch_to_seq(x[1]), arch_pool))
        # [[conv, reduc]]
        min_val = min(arch_pool_valid_acc)
        max_val = max(arch_pool_valid_acc)
        encoder_target = [(i - min_val) / (max_val - min_val) for i in arch_pool_valid_acc]

        if args.controller_expand:
            dataset = list(zip(encoder_input, encoder_target))
            n = len(dataset)
            ratio = 0.9
            split = int(n*ratio)
            np.random.shuffle(dataset)
            encoder_input, encoder_target = list(zip(*dataset))
            train_encoder_input = list(encoder_input[:split])
            train_encoder_target = list(encoder_target[:split])
            valid_encoder_input = list(encoder_input[split:])
            valid_encoder_target = list(encoder_target[split:])
            for _ in range(args.controller_expand-1):
                for src, tgt in zip(encoder_input[:split], encoder_target[:split]):
                    a = np.random.randint(0, args.child_nodes)
                    b = np.random.randint(0, args.child_nodes)
                    src = src[:4 * a] + src[4 * a + 2:4 * a + 4] + \
                            src[4 * a:4 * a + 2] + src[4 * (a + 1):20 + 4 * b] + \
                            src[20 + 4 * b + 2:20 + 4 * b + 4] + src[20 + 4 * b:20 + 4 * b + 2] + \
                            src[20 + 4 * (b + 1):]
                    train_encoder_input.append(src)
                    train_encoder_target.append(tgt)
        else:
            train_encoder_input = encoder_input
            train_encoder_target = encoder_target
            valid_encoder_input = encoder_input
            valid_encoder_target = encoder_target
        logging.info('Train data: {}\tValid data: {}'.format(len(train_encoder_input), len(valid_encoder_input)))

        nao_train_dataset = utils.NAODataset(train_encoder_input, train_encoder_target, True, swap=True if args.controller_expand is None else False)
        nao_valid_dataset = utils.NAODataset(valid_encoder_input, valid_encoder_target, False)
        nao_train_queue = torch.utils.data.DataLoader(
            nao_train_dataset, batch_size=args.controller_batch_size, shuffle=True, pin_memory=True)
        nao_valid_queue = torch.utils.data.DataLoader(
            nao_valid_dataset, batch_size=args.controller_batch_size, shuffle=False, pin_memory=True)
        nao_optimizer = torch.optim.Adam(nao.parameters(), lr=args.controller_lr, weight_decay=args.controller_l2_reg)
        for nao_epoch in range(1, args.controller_epochs+1):
            nao_loss, nao_mse, nao_ce = nao_train(nao_train_queue, nao, nao_optimizer)
            logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", nao_epoch, nao_loss, nao_mse, nao_ce)
            if nao_epoch % 100 == 0:
                pa, hs = nao_valid(nao_valid_queue, nao)
                logging.info("Evaluation on valid data")
                logging.info('epoch %04d pairwise accuracy %.6f hamming distance %.6f', nao_epoch, pa, hs)

        # Generate new archs
        new_archs = []
        max_step_size = 50
        predict_step_size = 0
        top100_archs = list(map(lambda x: utils.parse_arch_to_seq(x[0]) + utils.parse_arch_to_seq(x[1]), arch_pool[:100]))
        nao_infer_dataset = utils.NAODataset(top100_archs, None, False)
        nao_infer_queue = torch.utils.data.DataLoader(
            nao_infer_dataset, batch_size=len(nao_infer_dataset), shuffle=False, pin_memory=True)
        while len(new_archs) < args.controller_new_arch:
            predict_step_size += 1
            logging.info('Generate new architectures with step size %d', predict_step_size)
            new_arch = nao_infer(nao_infer_queue, nao, predict_step_size, direction='+')
            for arch in new_arch:
                if arch not in encoder_input and arch not in new_archs:
                    new_archs.append(arch)
                if len(new_archs) >= args.controller_new_arch:
                    break
            logging.info('%d new archs generated now', len(new_archs))
            if predict_step_size > max_step_size:
                break

        child_arch_pool = list(map(lambda x: utils.parse_seq_to_arch(x), new_archs))  # [[[conv],[reduc]]]
        logging.info("Generate %d new archs", len(child_arch_pool))

    logging.info('Finish Searching')
    logging.info('Reranking top 5 architectures')
    # reranking top 5
    top_archs = arch_pool[:5]
    if args.dataset == 'cifar10':
        top_archs_perf = train_and_evaluate_top_on_cifar10(top_archs, train_queue, valid_queue)
    elif args.dataset == 'cifar100':
        top_archs_perf = train_and_evaluate_top_on_cifar100(top_archs, train_queue, valid_queue)
    else:
        top_archs_perf = train_and_evaluate_top_on_imagenet(top_archs, train_queue, valid_queue)
    top_archs_sorted_indices = np.argsort(top_archs_perf)[::-1]
    top_archs = [top_archs[i] for i in top_archs_sorted_indices]
    top_archs_perf = [top_archs_perf[i] for i in top_archs_sorted_indices]
    with open(os.path.join(args.output_dir, 'arch_pool.final'), 'w') as fa:
        with open(os.path.join(args.output_dir, 'arch_pool.perf.final'), 'w') as fp:
            for arch, perf in zip(top_archs, top_archs_perf):
                arch = ' '.join(map(str, arch[0] + arch[1]))
                fa.write('{}\n'.format(arch))
                fp.write('{}\n'.format(perf))
  

if __name__ == '__main__':
    main()
