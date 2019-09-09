import os
import sys
import glob
import time
import copy
import logging
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import utils
from controller import NAO


parser = argparse.ArgumentParser()
# Basic model parameters.
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--new_arch', type=int, default=300)
parser.add_argument('--encoder_layers', type=int, default=1)
parser.add_argument('--encoder_hidden_size', type=int, default=64)
parser.add_argument('--encoder_emb_size', type=int, default=32)
parser.add_argument('--mlp_layers', type=int, default=0)
parser.add_argument('--mlp_hidden_size', type=int, default=200)
parser.add_argument('--decoder_layers', type=int, default=1)
parser.add_argument('--decoder_hidden_size', type=int, default=64)
parser.add_argument('--source_length', type=int, default=40)
parser.add_argument('--encoder_length', type=int, default=20)
parser.add_argument('--decoder_length', type=int, default=40)
parser.add_argument('--encoder_dropout', type=float, default=0)
parser.add_argument('--mlp_dropout', type=float, default=0)
parser.add_argument('--decoder_dropout', type=float, default=0)
parser.add_argument('--l2_reg', type=float, default=0.0)
parser.add_argument('--encoder_vocab_size', type=int, default=12)
parser.add_argument('--decoder_vocab_size', type=int, default=12)
parser.add_argument('--max_step_size', type=int, default=100)
parser.add_argument('--trade_off', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--expand', type=int, default=None)
parser.add_argument('--iteration', type=int, default=0)
parser.add_argument('--generate_topk', type=int, default=100)
parser.add_argument('--remain_topk', type=int, default=100)
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
        loss = args.trade_off * loss_1 + (1 - args.trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()
        
        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)
    
    return objs.avg, mse.avg, nll.avg


def nao_valid(queue, model):
    pa = utils.AvgrageMeter()
    hs = utils.AvgrageMeter()
    mse = utils.AvgrageMeter()
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
            mse.update(F.mse_loss(predict_value.data.squeeze(), encoder_target.data.squeeze()), n)
            pa.update(pairwise_acc, n)
            hs.update(hamming_dis, n)
    return mse.avg, pa.avg, hs.avg


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
        logging.info('No GPU found!')
        sys.exit(1)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    logging.info("Args = %s", args)

    nao = NAO(
        args.encoder_layers,
        args.encoder_vocab_size,
        args.encoder_hidden_size,
        args.encoder_dropout,
        args.encoder_length,
        args.source_length,
        args.encoder_emb_size,
        args.mlp_layers,
        args.mlp_hidden_size,
        args.mlp_dropout,
        args.decoder_layers,
        args.decoder_vocab_size,
        args.decoder_hidden_size,
        args.decoder_dropout,
        args.decoder_length,
    )
    logging.info("param size = %fMB", utils.count_parameters_in_MB(nao))
    nao = nao.cuda()

    with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(args.iteration))) as f:
        arch_pool = f.read().splitlines()
        arch_pool = list(map(utils.build_dag, arch_pool))
    with open(os.path.join(args.output_dir, 'arch_pool.{}.perf'.format(args.iteration))) as f:
        arch_pool_valid_acc = f.read().splitlines()
        arch_pool_valid_acc = list(map(float, arch_pool_valid_acc))

    logging.info('Training Encoder-Predictor-Decoder')
    train_encoder_input = list(map(lambda x: utils.parse_arch_to_seq(x[0], 2) + utils.parse_arch_to_seq(x[1], 2), arch_pool))
    min_val = min(arch_pool_valid_acc)
    max_val = max(arch_pool_valid_acc)
    train_encoder_target = [(i - min_val) / (max_val - min_val) for i in arch_pool_valid_acc]

    if args.expand is not None:
        buffer1, buffer2 = [], []
        for _ in range(args.expand-1):
            for src, tgt in zip(train_encoder_input, train_encoder_target):
                a = np.random.randint(0, 5)
                b = np.random.randint(0, 5)
                src = src[:4 * a] + src[4 * a + 2:4 * a + 4] + \
                        src[4 * a:4 * a + 2] + src[4 * (a + 1):20 + 4 * b] + \
                        src[20 + 4 * b + 2:20 + 4 * b + 4] + src[20 + 4 * b:20 + 4 * b + 2] + \
                        src[20 + 4 * (b + 1):]
                buffer1.append(src)
                buffer2.append(tgt)
        train_encoder_input += buffer1
        train_encoder_target += buffer2

    nao_train_dataset = utils.NAODataset(train_encoder_input, train_encoder_target, True, swap=True)
    nao_valid_dataset = utils.NAODataset(train_encoder_input, train_encoder_target, False)
    nao_train_queue = torch.utils.data.DataLoader(
        nao_train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    nao_valid_queue = torch.utils.data.DataLoader(
        nao_valid_dataset, batch_size=len(nao_valid_dataset), shuffle=False, pin_memory=True)
    nao_optimizer = torch.optim.Adam(nao.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    for nao_epoch in range(1, args.epochs + 1):
        nao_loss, nao_mse, nao_ce = nao_train(nao_train_queue, nao, nao_optimizer)
        if nao_epoch % 10 == 0 or nao_epoch == 1:
            logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", nao_epoch, nao_loss, nao_mse, nao_ce)
        if nao_epoch % 100 == 0 or nao_epoch == 1:
            mse, pa, hs = nao_valid(nao_train_queue, nao)
            logging.info("Evaluation on train data")
            logging.info('epoch %04d mse %.6f pairwise accuracy %.6f hamming distance %.6f', nao_epoch, mse, pa, hs)
            mse, pa, hs = nao_valid(nao_valid_queue, nao)
            logging.info("Evaluation on valid data")
            logging.info('epoch %04d mse %.6f pairwise accuracy %.6f hamming distance %.6f', nao_epoch, mse, pa, hs)

    new_archs = []
    predict_step_size = 0
    top_archs = list(map(lambda x: utils.parse_arch_to_seq(x[0], 2) + utils.parse_arch_to_seq(x[1], 2), arch_pool[:args.generate_topk]))
    nao_infer_dataset = utils.NAODataset(top_archs, None, False)
    nao_infer_queue = torch.utils.data.DataLoader(nao_infer_dataset, batch_size=len(nao_infer_dataset), shuffle=False, pin_memory=True)
        
    while len(new_archs) < args.new_arch:
        predict_step_size += 1
        logging.info('Generate new architectures with step size %d', predict_step_size)
        new_arch = nao_infer(nao_infer_queue, nao, predict_step_size, direction='+')
        for arch in new_arch:
            if arch not in train_encoder_input and arch not in new_archs:
                new_archs.append(arch)
            if len(new_archs) >= args.new_arch:
                break
        logging.info('%d new archs generated now', len(new_archs))
        if predict_step_size > args.max_step_size:
            break

    logging.info("Generate %d new archs", len(new_archs))
    new_arch_pool = list(map(lambda x: utils.parse_seq_to_arch(x, 2), new_archs))
    new_arch_pool = new_arch_pool + arch_pool[:args.remain_topk]
    with open(os.path.join(args.output_dir, 'new_arch_pool.{}'.format(args.iteration)), 'w') as f:
        for arch in new_arch_pool:
            arch = ' '.join(map(str, arch[0] + arch[1]))
            f.write('{}\n'.format(arch))
    logging.info('Finish training!')


if __name__ == '__main__':
    main()