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
from torch.autograd import Variable
from model_search import NASNetwork
from calculate_params import calculate_params
from controller import NAO

parser = argparse.ArgumentParser(description='NAO CIFAR-10')

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--child_sample_policy', type=str, default=None)
parser.add_argument('--child_batch_size', type=int, default=128)
parser.add_argument('--child_eval_batch_size', type=int, default=128)
parser.add_argument('--child_epochs', type=int, default=150)
parser.add_argument('--child_layers', type=int, default=5)
parser.add_argument('--child_nodes', type=int, default=5)
parser.add_argument('--child_channels', type=int, default=20)
parser.add_argument('--child_cutout_size', type=int, default=None)
parser.add_argument('--child_grad_bound', type=float, default=5.0)
parser.add_argument('--child_lr_max', type=float, default=0.025)
parser.add_argument('--child_lr_min', type=float, default=0.001)
parser.add_argument('--child_keep_prob', type=float, default=0.8)
parser.add_argument('--child_drop_path_keep_prob', type=float, default=1.0)
parser.add_argument('--child_l2_reg', type=float, default=3e-4)
parser.add_argument('--child_use_aux_head', action='store_true', default=False)
parser.add_argument('--child_eval_epochs', type=str, default='30')
parser.add_argument('--child_arch_pool', type=str, default=None)
parser.add_argument('--controller_seed_arch', type=int, default=1000)
parser.add_argument('--controller_encoder_layers', type=int, default=1)
parser.add_argument('--controller_encoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_encoder_emb_size', type=int, default=32)
parser.add_argument('--controller_mlp_layers', type=int, default=3)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)
parser.add_argument('--controller_decoder_layers', type=int, default=1)
parser.add_argument('--controller_decoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_source_length', type=int, default=40)
parser.add_argument('--controller_encoder_length', type=int, default=20)
parser.add_argument('--controller_decoder_length', type=int, default=40)
parser.add_argument('--controller_encoder_dropout', type=float, default=0.1)
parser.add_argument('--controller_mlp_dropout', type=float, default=0.1)
parser.add_argument('--controller_decoder_dropout', type=float, default=0.0)
parser.add_argument('--controller_l2_reg', type=float, default=1e-4)
parser.add_argument('--controller_encoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_decoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_epochs', type=int, default=1000)
parser.add_argument('--controller_eval_frequency', type=int, default=10)#[TODO rm this inrelease]
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_optimizer', type=str, default='adam')
parser.add_argument('--controller_grad_bound', type=float, default=5.0)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def child_train(train_queue, model, optimizer, global_step, arch_pool, criterion):
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
            logging.info('arch: %s', ' '.join(map(str, arch[0] + arch[1])))

    return top1.avg, objs.avg, global_step


def nao_train(encoder_input, encoder_target, decoder_target, model, parallel_model, optimizer, params, epoch):
    logging.info('Training Encoder-Predictor-Decoder')
    step = 0
    start_time = time.time()
    train_epochs = params['train_epochs']
    for e in range(1, train_epochs + 1):
        # prepare data
        N = len(encoder_input)
        if params['shuffle']:
            data = list(zip(encoder_input, encoder_target, decoder_target))
            np.random.shuffle(data)
            encoder_input, encoder_target, decoder_target = zip(*data)
        decoder_input = torch.cat((torch.LongTensor([[SOS_ID]] * N), torch.LongTensor(encoder_input)[:, :-1]), dim=1)
        
        encoder_train_input = controller_batchify(torch.LongTensor(encoder_input), params['batch_size'], cuda=True)
        encoder_train_target = controller_batchify(torch.Tensor(encoder_target), params['batch_size'], cuda=True)
        decoder_train_input = controller_batchify(torch.LongTensor(decoder_input), params['batch_size'], cuda=True)
        decoder_train_target = controller_batchify(torch.LongTensor(decoder_target), params['batch_size'], cuda=True)
        
        epoch += 1
        total_loss = 0
        mse = 0
        cse = 0
        batch = 0
        while batch < encoder_train_input.size(0):
            model.train()
            optimizer.zero_grad()
            encoder_train_input_batch = controller_get_batch(encoder_train_input, batch, evaluation=False)
            encoder_train_target_batch = controller_get_batch(encoder_train_target, batch, evaluation=False)
            decoder_train_input_batch = controller_get_batch(decoder_train_input, batch, evaluation=False)
            decoder_train_target_batch = controller_get_batch(decoder_train_target, batch, evaluation=False)
            predict_value, log_prob, arch = parallel_model(encoder_train_input_batch, decoder_train_input_batch)
            loss_1 = F.mse_loss(predict_value.squeeze(), encoder_train_target_batch.squeeze())
            loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_train_target_batch.view(-1))
            loss = params['trade_off'] * loss_1 + (1 - params['trade_off']) * loss_2
            mse += loss_1.data
            cse += loss_2.data
            total_loss += loss.data
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), params['max_gradient_norm'])
            optimizer.step()
            
            step += 1
            LOG = 100
            if step % LOG == 0:
                elapsed = time.time() - start_time
                cur_loss = total_loss[0] / LOG
                mse = mse[0] / LOG
                cse = cse[0] / LOG
                logging.info('| epoch {:6d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                             'mse {:5.6f} | cross entropy {:5.6f} | loss {:5.6f}'.format(
                    e, batch + 1, len(encoder_train_input), optimizer.param_groups[0]['lr'],
                       elapsed * 1000 / LOG, mse, cse, cur_loss))
                total_loss = 0
                mse = 0
                cse = 0
                start_time = time.time()
            batch += 1
        # [TODO rm this inrelease]
        if e % params['eval_frequency'] == 0:
            evaluate(encoder_input, encoder_target, decoder_target, model, parallel_model, params, epoch)
        if e % params['save_frequency'] == 0:
            save_checkpoint(model, optimizer, epoch, params['model_dir'])
            logging.info('Saving Model!')
    return epoch


def nao_valid(encoder_input, encoder_target, decoder_target, model, parallel_model, params, epoch):
    encoder_test_input = controller_batchify(torch.LongTensor(encoder_input), params['batch_size'], cuda=True)
    i = 0
    predict_value_list = []
    arch_list = []
    test_start_time = time.time()
    
    while i < encoder_test_input.size(0):
        model.eval()
        encoder_test_input_batch = controller_get_batch(encoder_test_input, i, evaluation=True)
        predict_value, logits, arch = parallel_model(encoder_test_input_batch)
        predict_value_list.extend(predict_value.data.squeeze().tolist())
        arch_list.extend(arch.data.squeeze().tolist())
        i += 1
    
    ground_truth_perf_list = encoder_target
    ground_truth_arch_list = decoder_target
    
    pairwise_acc = pairwise_accuracy(ground_truth_perf_list, predict_value_list)
    hamming_dis = hamming_distance(ground_truth_arch_list, arch_list)
    
    test_time = time.time() - test_start_time
    logging.info("Evaluation on training data\n")
    logging.info('| epoch {:3d} | pairwise accuracy {:<6.6f} | hamming distance {:<6.6f} | {:<6.2f} secs'.format(
        epoch, pairwise_acc, hamming_dis, test_time))


def nao_infer(encoder_input, model, parallel_model, params):
    logging.info(
        'Generating new architectures using gradient descent with step size {}'.format(params['predict_lambda']))
    logging.info('Preparing data')
    encoder_infer_input = controller_batchify(torch.LongTensor(encoder_input), params['batch_size'], cuda=True)
    
    new_arch_list = []
    for i in range(encoder_infer_input.size(0)):
        model.eval()
        model.zero_grad()
        encoder_infer_input_batch = controller_get_batch(encoder_infer_input, i, evaluation=False)
        new_arch = parallel_model.generate_new_arch(encoder_infer_input_batch, params['predict_lambda'])
        new_arch_list.extend(new_arch.data.squeeze().tolist())
    return new_arch_list


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    
    args.steps = int(np.ceil(50000 / args.child_batch_size)) * args.child_epochs

    logging.info("args = %s", args)
    if args.child_arch_pool is not None:
        logging.info('Architecture pool is provided, loading')
        with open(args.child_arch_pool) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            args.child_arch_pool = archs
    if  os.path.exists(os.path.join(args.output_dir, 'arch_pool')):
        logging.info('Architecture pool is founded, loading')
        with open(os.path.join(args.output_dir, 'arch_pool')) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            args.child_arch_pool = archs

    args.child_eval_epochs = eval(args.child_eval_epochs)
    train_transform, valid_transform = utils._data_transforms_cifar10(args.child_cutout_size)
    train_data = dset.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(0.9 * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=16)

    model = NASNetwork(args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob,
                       args.child_drop_path_keep_prob, args.child_use_aux_head, args.steps)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.child_epochs), eta_min=args.child_lr_min)

    # Train child model
    if args.child_arch_pool is None:
        arch_pool = utils.generate_arch(args.controller_seed_arch, args.child_nodes, 5)  # [[[conv],[reduc]]]
        args.child_arch_pool = arch_pool
        args.child_arch_pool_prob = None
    else:
        if args.child_sample_policy == 'uniform':
            args.child_arch_pool_prob = None
        elif args.child_sample_policy == 'params':
            args.child_arch_pool_prob = calculate_params(args.child_arch_pool)
        elif args.child_sample_policy == 'valid_performance':
            args.child_arch_pool_prob = child_valid(valid_queue, model, args.child_arch_pool)
        elif args.child_sample_policy == 'predicted_performance':
            encoder_input = list(map(lambda x: utils.parse_arch_to_seq(x[0], 2) + \
                                               utils.parse_arch_to_seq(x[1], 2), args.child_arch_pool))
            predicted_error_rate = controller.test(args, encoder_input)
            args.child_arch_pool_prob = [1 - i[0] for i in predicted_error_rate]
        else:
            raise ValueError('Child model arch pool sample policy is not provided!')

    for epoch in range(1, args.child_epochs + 1):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        # sample an arch to train
        train_acc, train_obj, step = child_train(train_queue, model, optimizer, step, args.child_arch_pool, criterion)
        logging.info('train_acc %f', train_acc)
    
        if isinstance(args.child_eval_epochs, int):
            if epoch % args.child_eval_every_epochs != 0:
                continue
        else:
            if epoch not in args.child_eval_epochs:
                continue
        # Evaluate seed archs
        valid_accuracy_list = child_valid(valid_queue, model, args.child_arch_pool, criterion)

        # Output archs and evaluated error rate
        old_archs = args.child_arch_pool
        old_archs_perf = [1 - i for i in valid_accuracy_list]

        old_archs_sorted_indices = np.argsort(old_archs_perf)
        old_archs = np.array(old_archs)[old_archs_sorted_indices].tolist()
        old_archs_perf = np.array(old_archs_perf)[old_archs_sorted_indices].tolist()
        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(epoch)), 'w') as fa:
            with open(os.path.join(args.output_dir, 'arch_pool.perf.{}'.format(epoch)), 'w') as fp:
                with open(os.path.join(args.output_dir, 'arch_pool'), 'w') as fa_latest:
                    with open(os.path.join(args.output_dir, 'arch_pool.perf'), 'w') as fp_latest:
                        for arch, perf in zip(old_archs, old_archs_perf):
                            arch = ' '.join(map(str, arch[0] + arch[1]))
                            fa.write('{}\n'.format(arch))
                            fa_latest.write('{}\n'.format(arch))
                            fp.write('{}\n'.format(perf))
                            fp_latest.write('{}\n'.format(perf))

        # Train Encoder-Predictor-Decoder
        encoder_input = list(map(lambda x: utils.parse_arch_to_seq(x[0], 2) + \
                                           utils.parse_arch_to_seq(x[1], 2), old_archs))
        # [[conv, reduc]]
        min_val = min(old_archs_perf)
        max_val = max(old_archs_perf)
        encoder_target = [(i - min_val) / (max_val - min_val) for i in old_archs_perf]
        decoder_target = copy.copy(encoder_input)

        nao = NAO(
        
        )
        model = model.cuda()
        args.controller_batches_per_epoch = np.ceil(len(encoder_input) / args.controller_batch_size)
        # if clean controller model
        controller.train(args, encoder_input, encoder_target, decoder_target)

        # Generate new archs
        # old_archs = old_archs[:450]
        new_archs = []
        max_step_size = 100
        args.controller_predict_lambda = 0
        top100_archs = list(map(lambda x: utils.parse_arch_to_seq(x[0], branch_length) + \
                                          utils.parse_arch_to_seq(x[1], branch_length), old_archs[:100]))
        while len(new_archs) < 500:
            args.controller_predict_lambda += 1
            new_arch = controller.predict(args, top100_archs)
            for arch in new_arch:
                if arch not in encoder_input and arch not in new_archs:
                    new_archs.append(arch)
                if len(new_archs) >= 500:
                    break
            logging.info('{} new archs generated now'.format(len(new_archs)))
            if args.controller_predict_lambda > max_step_size:
                break
                # [[conv, reduc]]
        new_archs = list(map(lambda x: utils.parse_seq_to_arch(x, branch_length), new_archs))  # [[[conv],[reduc]]]
        num_new_archs = len(new_archs)
        logging.info("Generate {} new archs".format(num_new_archs))
        new_arch_pool = old_archs[:len(old_archs) - (num_new_archs + 50)] + new_archs + utils.generate_arch(50, 5, 5)
        logging.info("Totally {} archs now to train".format(len(new_arch_pool)))
        args.child_arch_pool = new_arch_pool
        with open(os.path.join(args.child_model_dir, 'arch_pool'), 'w') as f:
            for arch in new_arch_pool:
                arch = ' '.join(map(str, arch[0] + arch[1]))
                f.write('{}\n'.format(arch))
  

if __name__ == '__main__':
  main()
