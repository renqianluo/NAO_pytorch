import os
import sys
import glob
import time
import copy
import math
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
from model import NASNetworkCIFAR
from operations import OPERATIONS_CIFAR
from controller import NAO

parser = argparse.ArgumentParser(description='NAO Search')

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100'])
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--child_batch_size', type=int, default=128)
parser.add_argument('--child_eval_batch_size', type=int, default=500)
parser.add_argument('--child_budget', type=int, default=100)
parser.add_argument('--child_layers', type=int, default=3)
parser.add_argument('--child_nodes', type=int, default=5)
parser.add_argument('--child_channels', type=int, default=20)
parser.add_argument('--child_cutout_size', type=int, default=None)
parser.add_argument('--child_grad_bound', type=float, default=5.0)
parser.add_argument('--child_lr_max', type=float, default=0.025)
parser.add_argument('--child_lr_min', type=float, default=0.001)
parser.add_argument('--child_keep_prob', type=float, default=0.6)
parser.add_argument('--child_drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--child_l2_reg', type=float, default=3e-4)
parser.add_argument('--child_use_aux_head', action='store_true', default=False)
parser.add_argument('--child_arch_pool', type=str, default=None)
parser.add_argument('--child_label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--child_adaptive_budget', action='store_true', default=False)
parser.add_argument('--controller_seed_arch', type=int, default=100)
parser.add_argument('--controller_expand', type=int, default=10)
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
parser.add_argument('--controller_encoder_vocab_size', type=int, default=None)
parser.add_argument('--controller_decoder_vocab_size', type=int, default=None)
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


def build_task(ratio=0.9):
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
    
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    return train_queue, valid_queue, train_criterion, eval_criterion


def child_estimate(train_queue, valid_queue, arch_pool, train_criterion, eval_criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    model_sizes = list(map(lambda x:utils.count_parameters_in_MB(
        NASNetworkCIFAR(args, args.num_class, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                       args.child_use_aux_head, args.child_budget, x)), arch_pool))
    if args.child_adaptive_budget:
        min_size = min(model_sizes)
        budgets = list(map(lambda x:math.ceil(args.child_budget * (x / min_size)), model_sizes))
    else:
        budgets = list(map(lambda x:args.child_budget, model_sizes))
    valid_acc = []
    
    for i, (arch, model_size, budget) in enumerate(zip(arch_pool, model_sizes, budgets)):
        step = 0
        model = NASNetworkCIFAR(args, args.num_class, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                       args.child_use_aux_head, budget, arch)
        model = model.cuda()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr_max,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )

        logging.info('%d arch: %s', i+1, ' '.join(map(str, arch[0] + arch[1])))
        logging.info('Size: {}MB'.format(model_size))
        logging.info('Training budget: {}'.format(budget))
        model.train()
        while True:
            for inputs, targets in train_queue:
                inputs = inputs.cuda()
                targets = targets.cuda()

                optimizer.zero_grad()
                logits, aux_logits = model(inputs, step)
                loss = train_criterion(logits, targets)
                if aux_logits is not None:
                    aux_loss = train_criterion(aux_logits, targets)
                    loss += 0.4 * aux_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
                optimizer.step()
                
                prec1 = utils.accuracy(logits, targets, topk=(1,))[0]
                n = inputs.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                
                step += 1

                if step == budget:
                    break
            if step == budget:
                break
        logging.info('Training loss %e, top1 acc %f', objs.avg, top1.avg)

        top1.reset()
        with torch.no_grad():
            model.eval()
            for inputs, targets in valid_queue:
                inputs = inputs.cuda()
                targets = targets.cuda()
                logits, _ = model(inputs, arch)
                loss = eval_criterion(logits, targets)
                
                prec1 = utils.accuracy(logits, targets, topk=(1,))[0]
                n = inputs.size(0)
                top1.update(prec1.data, n)
            logging.info('Valid top1 acc %f', top1.avg)
            valid_acc.append(top1.avg)

    return valid_acc


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


def nao_data_preprocess(encoder_input, encoder_target):
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
    return train_encoder_input, train_encoder_target, valid_encoder_input, valid_encoder_target


def train_nao(model, encoder_input, encoder_target):
    train_encoder_input, train_encoder_target, valid_encoder_input, valid_encoder_target = nao_data_preprocess(encoder_input, encoder_target)
    logging.info('Train data: {}\tValid data: {}'.format(len(train_encoder_input), len(valid_encoder_input)))
    nao_train_dataset = utils.NAODataset(train_encoder_input, train_encoder_target, True, swap=True if args.controller_expand is None else False)
    nao_valid_dataset = utils.NAODataset(valid_encoder_input, valid_encoder_target, False)
    nao_train_queue = torch.utils.data.DataLoader(
        nao_train_dataset, batch_size=args.controller_batch_size, shuffle=True, pin_memory=True)
    nao_valid_queue = torch.utils.data.DataLoader(
        nao_valid_dataset, batch_size=args.controller_batch_size, shuffle=False, pin_memory=True)
    nao_optimizer = torch.optim.Adam(model.parameters(), lr=args.controller_lr, weight_decay=args.controller_l2_reg)
    for nao_epoch in range(1, args.controller_epochs+1):
        nao_loss, nao_mse, nao_ce = nao_train(nao_train_queue, model, nao_optimizer)
        logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", nao_epoch, nao_loss, nao_mse, nao_ce)
        if nao_epoch % 100 == 0:
            pa, hs = nao_valid(nao_valid_queue, model)
            logging.info("Evaluation on valid data")
            logging.info('epoch %04d pairwise accuracy %.6f hamming distance %.6f', nao_epoch, pa, hs)
    return model


def generate_synthetic_nao_data(model, exclude=[], maxn=1000):
    synthetic_encoder_input = []
    synthetic_encoder_target = []
    while len(synthetic_encoder_input) < maxn:
        synthetic_arch = utils.generate_arch(1, args.child_nodes, args.child_num_ops)[0]
        synthetic_arch = utils.parse_arch_to_seq(synthetic_arch[0]) + utils.parse_arch_to_seq(synthetic_arch[1])
        if synthetic_arch not in exclude and synthetic_arch not in synthetic_encoder_input:
            synthetic_encoder_input.append(synthetic_arch)
    
    nao_synthetic_dataset = utils.NAODataset(synthetic_encoder_input, None, False)      
    nao_synthetic_queue = torch.utils.data.DataLoader(nao_synthetic_dataset, batch_size=len(nao_synthetic_dataset), shuffle=False, pin_memory=True)

    with torch.no_grad():
        model.eval()
        for sample in nao_synthetic_queue:
            encoder_input = sample['encoder_input'].cuda()
            _, _, _, predict_value = model.encoder(encoder_input)
            synthetic_encoder_target += predict_value.data.squeeze().tolist()
    assert len(synthetic_encoder_input) == len(synthetic_encoder_target)
    return synthetic_encoder_input, synthetic_encoder_target


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
    cudnn.benchmark = False
    cudnn.deterministic = True

    if args.dataset == 'cifar10':
        args.num_class = 10
    elif args.dataset == 'cifar100':
        args.num_class = 100
    else:
        args.num_class = 10
    args.child_num_ops = len(OPERATIONS_CIFAR)
    args.controller_encoder_vocab_size = 1 + ( args.child_nodes + 2 - 1 ) + args.child_num_ops
    args.controller_decoder_vocab_size = args.controller_encoder_vocab_size
    args.steps = int(np.ceil(45000 / args.child_batch_size)) * args.child_budget

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

    train_queue, valid_queue, train_criterion, eval_criterion = build_task(ratio=0.9)

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

        # Estimate seed archs
        child_arch_pool_valid_acc = child_estimate(train_queue, valid_queue, child_arch_pool, train_criterion, eval_criterion)

        arch_pool += child_arch_pool
        arch_pool_valid_acc += child_arch_pool_valid_acc

        arch_pool_valid_acc_sorted_indices = np.argsort(arch_pool_valid_acc)[::-1]
        arch_pool = list(map(lambda x:arch_pool[x], arch_pool_valid_acc_sorted_indices))
        arch_pool_valid_acc = list(map(lambda x:arch_pool_valid_acc[x], arch_pool_valid_acc_sorted_indices))
        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(i)), 'w') as fa:
            with open(os.path.join(args.output_dir, 'arch_pool.perf.{}'.format(i)), 'w') as fp:
                for arch, perf in zip(arch_pool, arch_pool_valid_acc):
                    arch = ' '.join(map(str, arch[0] + arch[1]))
                    fa.write('{}\n'.format(arch))
                    fp.write('{}\n'.format(perf))
        if i == 3:
            with open(os.path.join(args.output_dir, 'arch_pool.final'), 'w') as fa:
                with open(os.path.join(args.output_dir, 'arch_pool.perf.final'), 'w') as fp:
                    for arch, perf in zip(arch_pool, arch_pool_valid_acc):
                        arch = ' '.join(map(str, arch[0] + arch[1]))
                        fa.write('{}\n'.format(arch))
                        fp.write('{}\n'.format(perf))
            break
                            
        # Train Encoder-Predictor-Decoder
        logging.info('Training Encoder-Predictor-Decoder')
        encoder_input = list(map(lambda x: utils.parse_arch_to_seq(x[0]) + utils.parse_arch_to_seq(x[1]), arch_pool))
        # [[conv, reduc]]
        min_val = min(arch_pool_valid_acc)
        max_val = max(arch_pool_valid_acc)
        encoder_target = list(map(lambda x: (x - min_val) / (max_val - min_val), arch_pool_valid_acc))

        # Pre-train NAO
        logging.info('Pre-train NAO')
        nao = train_nao(nao, encoder_input, encoder_target)
        logging.info('Finish pre-training NAO')
        # Generate synthetic data
        logging.info('Generate synthetic data for NAO')
        synthetic_encoder_input, synthetic_encoder_target = generate_synthetic_nao_data(nao, encoder_input, args.controller_seed_arch * 10 - len(arch_pool))
        all_encoder_input = encoder_input + synthetic_encoder_input
        all_encoder_target = encoder_target + synthetic_encoder_target
        # Train NAO
        logging.info('Train NAO')
        nao = train_nao(nao, all_encoder_input, all_encoder_target)
        logging.info('Finish training NAO')

        # Generate new archs
        new_archs = []
        max_step_size = 50
        predict_step_size = 0
        top100_archs = list(map(lambda x: utils.parse_arch_to_seq(x[0]) + utils.parse_arch_to_seq(x[1]), arch_pool[:100]))
        nao_infer_dataset = utils.NAODataset(top100_archs, None, False)
        nao_infer_queue = torch.utils.data.DataLoader(
            nao_infer_dataset, batch_size=len(nao_infer_dataset), shuffle=False, pin_memory=True)
        while len(new_archs) < args.controller_seed_arch:
            predict_step_size += 1
            logging.info('Generate new architectures with step size %d', predict_step_size)
            new_arch = nao_infer(nao_infer_queue, nao, predict_step_size, direction='+')
            for arch in new_arch:
                if arch not in encoder_input and arch not in new_archs:
                    new_archs.append(arch)
                if len(new_archs) >= args.controller_seed_arch:
                    break
            logging.info('%d new archs generated now', len(new_archs))
            if predict_step_size > max_step_size:
                break

        child_arch_pool = list(map(lambda x: utils.parse_seq_to_arch(x), new_archs))  # [[[conv],[reduc]]]
        logging.info("Generate %d new archs", len(child_arch_pool))

    logging.info('Finish Searching')
  

if __name__ == '__main__':
    main()
