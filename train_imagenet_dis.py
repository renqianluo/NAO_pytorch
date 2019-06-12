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
import random
import distributed_utils
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch import Tensor
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model import NASNetworkImageNet

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'])
parser.add_argument('--data', type=str, default='data/imagenet')
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False) # best practice: do not use lazy_load. when using zip_file, do not use lazy_load
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--arch', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--nodes', type=int, default=5)
parser.add_argument('--channels', type=int, default=48)
parser.add_argument('--use_aux_head', action='store_true', default=False)
parser.add_argument('--keep_prob', type=float, default=1.0)
parser.add_argument('--drop_path_keep_prob', type=float, default=1.0)
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--l2_reg', type=float, default=3e-5)
parser.add_argument('--seed', type=int, default=0)

# distributed
parser.add_argument('--distributed_init_method', type=str, default=None)
parser.add_argument('--distributed_world_size', type=int, default=max(1, torch.cuda.device_count()))
parser.add_argument('--distributed_rank', type=int, default=0)
parser.add_argument('--distributed_backend', type=str, default='nccl')
parser.add_argument('--distributed_port', default=-1, type=int)
parser.add_argument('--device_id', '--local_rank', default=0, type=int)
parser.add_argument('--distributed_no_spawn', action='store_true')
parser.add_argument('--ddp_backend', default='c10d', type=str, choices=['c10d', 'no_c10d'])
parser.add_argument('--bucket_cap_mb', default=25, type=int, metavar='MB')
parser.add_argument('--fix_batches_to_gpus', action='store_true')
parser.add_argument('--philly_vc', type=str, default=None)

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


def train(train_queue, model, optimizer, global_step, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()
    
        optimizer.zero_grad()
        logits, aux_logits = model(input, global_step)
        global_step += 1
        loss = criterion(logits, target)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, target)
            loss += 0.4 * aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
    
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
    
        if (step+1) % 100 == 0:
            logging.info('train %03d loss %e top1 %f top5 %f', step+1, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, global_step


def valid(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()
        
            logits, _ = model(input)
            loss = criterion(logits, target)
        
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
        
            if (step+1) % 100 == 0:
                logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg


def build_imagenet(args, model_state_dict, optimizer_state_dict, init_distributed=False, epoch=-1):
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
        validdir = os.path.join(args.data, 'valid.zip')
        if args.lazy_load:
            train_data = utils.ZipDataset(traindir, train_transform)
            valid_data = utils.ZipDataset(validdir, valid_transform)
        else:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryZipDataset(traindir, train_transform, num_workers=32)
            valid_data = utils.InMemoryZipDataset(validdir, valid_transform, num_workers=32)
    else:
        logging.info('Loading data from directory')
        traindir = os.path.join(args.data, 'train')
        validdir = os.path.join(args.data, 'valid')
        if args.lazy_load:
            train_data = dset.ImageFolder(traindir, train_transform)
            valid_data = dset.ImageFolder(validdir, valid_transform)
        else:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryDataset(traindir, train_transform, num_workers=32)
            valid_data = utils.InMemoryDataset(validdir, valid_transform, num_workers=32)
    
    logging.info('Found %d in training data', len(train_data))
    logging.info('Found %d in validation data', len(valid_data))

    args.steps = int(np.ceil(len(train_data) / (args.batch_size))) * torch.cuda.device_count() * args.epochs
    args.batch_size = args.batch_size * args.distributed_world_size
    args.lr = args.batch_size * args.distributed_world_size


    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)

    if init_distributed:
        import socket
        args.distributed_rank = distributed_utils.distributed_init(args)
        print('| initialized host {} as rank {}'.format(socket.gethostbyname(), args.distributed_rank))

    model = NASNetworkImageNet(args, 1000, args.layers, args.nodes, args.channels, args.keep_prob,
                       args.drop_path_keep_prob, args.use_aux_head, args.steps, args.arch)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info("multi adds = %fM", model.multi_adds / 1000000)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    model = model.cuda()
    
    train_criterion = CrossEntropyLabelSmooth(1000, args.label_smooth).cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, args.gamma, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def main(args, init_distributed=False):
    if not torch.cuda.is_available():
        logging.info('No GPU found!')
        sys.exit(1)
    
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    
    logging.info("Args = %s", args)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        
    _, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1 = utils.load(args.output_dir)
    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_imagenet(args, model_state_dict, optimizer_state_dict, init_distributed=init_distributed, epoch=epoch-1)

    logging.info('Training on {} GPUs'.format(args.distributed_world_size))

    while epoch < args.epochs:
        scheduler.step()
        if init_distributed:
            torch.distributed.barrier()
        logging.info('rank {} epoch {} lr {}'.format(args.distributed_rank, epoch, scheduler.get_lr()[0]))
        train_acc, train_obj, step = train(train_queue, model, optimizer, step, train_criterion)
        logging.info('rank {} train_acc {}'.format(args.distributed_rank, train_acc))
        valid_acc_top1, valid_acc_top5, valid_obj = valid(valid_queue, model, eval_criterion)
        logging.info('rank {} valid_acc_top1 {}'.format(args.distributed_rank, valid_acc_top1))
        logging.info('rank {} valid_acc_top5 {}'.format(args.distributed_rank, valid_acc_top5))

        epoch += 1
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        tmp_file = '{}.tmp'.format(epoch)
        if distributed_utils.is_master(args):
            utils.save(args.output_dir, args, model, epoch, step, optimizer, best_acc_top1, is_best)
            print('Master rank, completed saving ckpt {}'.format(epoch))
            time.sleep(5)
            with open(tmp_file, 'w') as fout:
                fout.write('I have completed saving {}'.format(epoch))
            time.sleep(5)
        else:
            while not os.path.isfile(tmp_file):
                time.sleep(5)
        if args.distributed_world_size > 1:
            torch.distributed.barrier()
        if distributed_utils.is_master(args):
            os.remove(tmp_file)


def distributed_main(i, args):
    args.device_id = i
    if args.distributed_rank is None:
        args.distributed_rank = i
    main(args, init_distributed=True)


def cli_main(args):
    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)
    
    if args.distributed_init_method is not None:
        if args.philly_vc is not None and not 'tcp' in args.distributed_init_method:
            distributed_utils.setup_init_philly_shared_system(args)
        print('| running distributed main')
        distributed_main(args.device_id, args)
    
    elif args.distributed_world_size > 1:
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| Note: you may get better performance with: --ddp-backend=no_c10d')
        print('| running multiprocessing main')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, main),
            nproc=args.distributed_world_size
        )
    
    else:
        print('| running single GPU main')
        main(args)

if __name__ == '__main__':
    cli_main(args)
