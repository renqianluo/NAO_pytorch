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
import torch.backends.cudnn as cudnn
from controller import NAO


parser = argparse.ArgumentParser()
# Basic model parameters.
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--controller_seed_arch', type=int, default=1000)
parser.add_argument('--controller_new_arch', type=int, default=300)
parser.add_argument('--controller_random_arch', type=int, default=100)
parser.add_argument('--controller_replace', action='store_true', default=False)
parser.add_argument('--controller_encoder_layers', type=int, default=1)
parser.add_argument('--controller_encoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_encoder_emb_size', type=int, default=48)
parser.add_argument('--controller_mlp_layers', type=int, default=3)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)
parser.add_argument('--controller_decoder_layers', type=int, default=1)
parser.add_argument('--controller_decoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_source_length', type=int, default=40)
parser.add_argument('--controller_encoder_length', type=int, default=20)
parser.add_argument('--controller_decoder_length', type=int, default=40)
parser.add_argument('--controller_encoder_dropout', type=float, default=0)
parser.add_argument('--controller_mlp_dropout', type=float, default=0.1)
parser.add_argument('--controller_decoder_dropout', type=float, default=0)
parser.add_argument('--controller_l2_reg', type=float, default=1e-4)
parser.add_argument('--controller_encoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_decoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_epochs', type=int, default=1000)
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_optimizer', type=str, default='adam')
parser.add_argument('--controller_grad_bound', type=float, default=5.0)
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

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
    pa = utils.AvgrageMeter()
    hs = utils.AvgrageMeter()
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
            pairwise_acc = utils.pairwise_accuracy(encoder_target.data.squeeze().tolist(),
                                                predict_value.data.squeeze().tolist())
            hamming_dis = utils.hamming_distance(decoder_target.data.squeeze().tolist(), arch.data.squeeze().tolist())
            pa.update(pairwise_acc, n)
            hs.update(hamming_dis, n)
    return pa.avg, hs.avg


def nao_infer(queue, model, step):
    new_arch_list = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = sample['encoder_input']
        encoder_input = encoder_input.cuda()
        model.zero_grad()
        new_arch = model.generate_new_arch(encoder_input, step)
        new_arch_list.extend(new_arch.data.squeeze().tolist())
    return new_arch_list

def main():
    arch_pool = utils.generate_arch(args.controller_seed_arch, 5, 5)
    valid_arch_pool = utils.generate_arch(100, 5, 5)
    train_encoder_input = list(map(lambda x: utils.parse_arch_to_seq(x[0], 2) + utils.parse_arch_to_seq(x[1], 2), arch_pool))
    valid_encoder_input = list(map(lambda x: utils.parse_arch_to_seq(x[0], 2) + utils.parse_arch_to_seq(x[1], 2), valid_arch_pool))
    train_encoder_target = [np.random.random() for i in range(args.controller_seed_arch)]
    valid_encoder_target = [np.random.random() for i in range(100)]
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
    logging.info("param size = %fMB", utils.count_parameters_in_MB(nao))
    nao = nao.cuda()
    nao_train_dataset = utils.NAODataset(train_encoder_input, train_encoder_target, True, swap=True)
    nao_valid_dataset = utils.NAODataset(valid_encoder_input, valid_encoder_target, False)
    nao_train_queue = torch.utils.data.DataLoader(
        nao_train_dataset, batch_size=args.controller_batch_size, shuffle=True, pin_memory=True)
    nao_valid_queue = torch.utils.data.DataLoader(
        nao_valid_dataset, batch_size=len(nao_valid_dataset), shuffle=False, pin_memory=True)
    nao_optimizer = torch.optim.Adam(nao.parameters(), lr=args.controller_lr, weight_decay=args.controller_l2_reg)
    for nao_epoch in range(1, args.controller_epochs + 1):
        nao_loss, nao_mse, nao_ce = nao_train(nao_train_queue, nao, nao_optimizer)
        if nao_epoch % 10 == 0:
            logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", nao_epoch, nao_loss, nao_mse, nao_ce)
        if nao_epoch % 100 == 0:
            pa, hs = nao_valid(nao_valid_queue, nao)
            logging.info("Evaluation on training data")
            logging.info('epoch %04d pairwise accuracy %.6f hamming distance %.6f', nao_epoch, pa, hs)

    new_archs = []
    max_step_size = 100
    predict_step_size = 0
    top100_archs = list(map(lambda x: utils.parse_arch_to_seq(x[0], 2) + utils.parse_arch_to_seq(x[1], 2), arch_pool[:100]))
    nao_infer_dataset = utils.NAODataset(top100_archs, None, False)
    nao_infer_queue = torch.utils.data.DataLoader(nao_infer_dataset, batch_size=len(nao_infer_dataset), shuffle=False, pin_memory=True)
    while len(new_archs) < args.controller_new_arch:
        predict_step_size += 1
        logging.info('Generate new architectures with step size %d', predict_step_size)
        new_arch = nao_infer(nao_infer_queue, nao, predict_step_size)
        for arch in new_arch:
            if arch not in train_encoder_input and arch not in new_archs:
                new_archs.append(arch)
            if len(new_archs) >= args.controller_new_arch:
                break
        logging.info('%d new archs generated now', len(new_archs))
        if predict_step_size > max_step_size:
            break
            # [[conv, reduc]]
    new_archs = list(map(lambda x: utils.parse_seq_to_arch(x, 2), new_archs))  # [[[conv],[reduc]]]
    num_new_archs = len(new_archs)
    logging.info("Generate %d new archs", num_new_archs)
    new_arch_pool = arch_pool + new_archs + utils.generate_arch(args.controller_random_arch, 5, 5)
    logging.info("Totally %d archs now to train", len(new_arch_pool))
    arch_pool = new_arch_pool


if __name__ == '__main__':
    main()