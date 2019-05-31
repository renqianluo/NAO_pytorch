import os
import sys
import glob
import time
import copy
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
from torch.autograd import Variable
from model_search import NASNetworkCIFAR, NASNetworkImageNet

parser = argparse.ArgumentParser(description='NAO CIFAR-10')

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100, imagenet'])
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--preload', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=160)
parser.add_argument('--eval_batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--nodes', type=int, default=5)
parser.add_argument('--channels', type=int, default=20)
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--lr_max', type=float, default=0.025)
parser.add_argument('--lr_min', type=float, default=0.001)
parser.add_argument('--keep_prob', type=float, default=1.0)
parser.add_argument('--drop_path_keep_prob', type=float, default=1.0)
parser.add_argument('--l2_reg', type=float, default=3e-4)
parser.add_argument('--use_aux_head', action='store_true', default=False)
parser.add_argument('--eval_epochs', type=str, default='30')
parser.add_argument('--arch_pool', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
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


def build_cifar10(model_state_dict, optimizer_state_dict, **kwargs):
    epoch = kwargs.pop('epoch')
    train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_size)
    train_data = dset.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data_path, train=False, download=True, transform=train_transform)
    
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=True, pin_memory=True, num_workers=16)
    
    model = NASNetworkCIFAR(10, args.layers, args.nodes, args.channels, args.keep_prob,
                            args.drop_path_keep_prob,
                            args.use_aux_head, args.steps)
    model = model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr_max,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr_min, epoch)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def build_cifar100(model_state_dict, optimizer_state_dict, **kwargs):
    epoch = kwargs.pop('epoch')
    train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_size)
    train_data = dset.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data_path, train=False, download=True, transform=train_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=True, pin_memory=True, num_workers=16)

    model = NASNetworkCIFAR(100, args.layers, args.nodes, args.channels, args.keep_prob,
                            args.drop_path_keep_prob,
                            args.use_aux_head, args.steps)
    model = model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr_max,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr_min, epoch)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def build_imagenet(model_state_dict, optimizer_state_dict, **kwargs):
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
        validdir = os.path.join(args.data, 'validation.zip')
        if args.preload:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryZipDataset(traindir, train_transform, num_workers=32)
            valid_data = utils.InMemoryZipDataset(validdir, valid_transform, num_workers=32)
        else:
            train_data = utils.ZipDataset(traindir, train_transform)
            valid_data = utils.ZipDataset(validdir, valid_transform)
    else:
        logging.info('Loading data from directory')
        traindir = os.path.join(args.data, 'train')
        validdir = os.path.join(args.data, 'validation')
        if args.preload:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryDataset(traindir, train_transform, num_workers=32)
            valid_data = utils.InMemoryDataset(validdir, valid_transform, num_workers=32)
        else:
            train_data = dset.ImageFolder(traindir, train_transform)
            valid_data = dset.ImageFolder(validdir, valid_transform)
    
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True, num_workers=16)
    
    model = NASNetworkImageNet(1000, args.layers, args.nodes, args.channels, args.keep_prob,
                               args.drop_path_keep_prob, args.use_aux_head, args.steps)
    model = model.cuda()
    train_criterion = CrossEntropyLabelSmooth(1000, args.label_smooth).cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def train(train_queue, model, optimizer, global_step, arch_pool, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)
        
        optimizer.zero_grad()
        # sample an arch to train
        arch = utils.sample_arch(arch_pool)
        logits, aux_logits = model(input, arch, global_step)
        global_step += 1
        loss = criterion(logits, target)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, target)
            loss += 0.4 * aux_loss
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_bound)
        optimizer.step()
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)
        
        if step % 100 == 0:
            logging.info('train %03d loss %e top1 %f top5 %f', step, objs.avg, top1.avg, top5.avg)
            logging.info('arch: %s', ' '.join(map(str, arch[0]+arch[1])))
    
    return top1.avg, objs.avg, global_step


def valid(valid_queue, model, arch_pool, criterion):
    valid_acc_list = []
    for arch in arch_pool:
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda(async=True)
        
            logits, _ = model(input, arch)
            loss = criterion(logits, target)
        
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data[0], n)
            top1.update(prec1.data[0], n)
            top5.update(prec5.data[0], n)
        
            if step % 100 == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        valid_acc_list.append(top1.avg)
    return valid_acc_list


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
        
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    args.steps = int(np.ceil(50000 / args.batch_size)) * args.epochs

    logging.info("Args = %s", args)
    
    if args.arch_pool is not None:
        logging.info('Architecture pool is provided, loading')
        with open(args.arch_pool) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            arch_pool = archs
    
    eval_epochs = eval(args.eval_epochs)
    _, model_state_dict, epoch, step, optimizer_state_dict = utils.load(args.output_dir)
    
    # Train child model
    assert arch_pool is not None
    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_fn(
        model_state_dict, optimizer_state_dict, epoch=epoch)

    eval_points = utils.generate_eval_points(eval_epochs, 0, args.epochs)
    step = 0
    while epoch < args.epochs:
        epoch += 1
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        # sample an arch to train
        train_acc, train_obj, step = train(train_queue, model, optimizer, step, arch_pool, train_criterion)
        logging.info('train_acc %f', train_acc)
    
        # Evaluate seed archs
        if epoch not in eval_points:
            continue
            
        valid_accuracy_list = valid(valid_queue, model, arch_pool, eval_criterion)

        # Output archs and evaluated error rate
        with open(os.path.join(args.output_dir, 'arch_pool.{}.perf'.format(epoch)), 'w') as f:
            for arch, perf in zip(arch_pool, valid_accuracy_list):
                arch = ' '.join(map(str, arch[0] + arch[1]))
                f.write('arch: {}\tvalid acc: {}\n'.format(arch, perf))
        utils.save(args.output_dir, args, model, epoch, step, optimizer)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), args.lr_min)
      

if __name__ == '__main__':
    main()