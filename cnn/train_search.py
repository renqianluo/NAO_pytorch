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
from model_search import train as child_train
from model_search import valid as child_valid
from calculate_params import calculate_params
import controller

parser = argparse.ArgumentParser(description='NAO CIFAR-10')

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--eval_dataset', type=str, default='valid', choices=['valid', 'test', 'both'])
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--child_sample_policy', type=str, default=None)
parser.add_argument('--child_batch_size', type=int, default=128)
parser.add_argument('--child_eval_batch_size', type=int, default=128)
parser.add_argument('--child_num_epochs', type=int, default=150)
parser.add_argument('--child_lr_dec_every', type=int, default=100)
parser.add_argument('--child_num_layers', type=int, default=5)
parser.add_argument('--child_num_cells', type=int, default=5)
parser.add_argument('--child_out_filters', type=int, default=20)
parser.add_argument('--child_out_filters_scale', type=int, default=1)
parser.add_argument('--child_num_branches', type=int, default=5)
parser.add_argument('--child_num_aggregate', type=int, default=None)
parser.add_argument('--child_num_replicas', type=int, default=None)
parser.add_argument('--child_lr_T_0', type=int, default=None)
parser.add_argument('--child_lr_T_mul', type=int, default=None)
parser.add_argument('--child_cutout_size', type=int, default=None)
parser.add_argument('--child_grad_bound', type=float, default=5.0)
parser.add_argument('--child_lr_max', type=float, default=None)
parser.add_argument('--child_lr_min', type=float, default=None)
parser.add_argument('--child_keep_prob', type=float, default=0.5)
parser.add_argument('--child_drop_path_keep_prob', type=float, default=1.0)
parser.add_argument('--child_l2_reg', type=float, default=1e-4)
parser.add_argument('--child_use_aux_heads', action='store_true', default=False)
parser.add_argument('--child_sync_replicas', action='store_true', default=False)
parser.add_argument('--child_lr_cosine', action='store_true', default=False)
parser.add_argument('--child_eval_every_epochs', type=str, default='30')
parser.add_argument('--child_arch_pool', type=str, default=None)
parser.add_argument('--controller_num_seed_arch', type=int, default=1000)
parser.add_argument('--controller_encoder_num_layers', type=int, default=1)
parser.add_argument('--controller_encoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_encoder_emb_size', type=int, default=32)
parser.add_argument('--controller_mlp_num_layers', type=int, default=0)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)
parser.add_argument('--controller_decoder_num_layers', type=int, default=1)
parser.add_argument('--controller_decoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_source_length', type=int, default=60)
parser.add_argument('--controller_encoder_length', type=int, default=20)
parser.add_argument('--controller_decoder_length', type=int, default=60)
parser.add_argument('--controller_encoder_dropout', type=float, default=0.1)
parser.add_argument('--controller_mlp_dropout', type=float, default=0.1)
parser.add_argument('--controller_decoder_dropout', type=float, default=0.0)
parser.add_argument('--controller_weight_decay', type=float, default=1e-4)
parser.add_argument('--controller_encoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_decoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_train_epochs', type=int, default=300)
parser.add_argument('--controller_eval_frequency', type=int, default=10)#[TODO rm this inrelease]
parser.add_argument('--controller_save_frequency', type=int, default=10)
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_optimizer', type=str, default='adam')
parser.add_argument('--controller_start_decay_step', type=int, default=100)
parser.add_argument('--controller_decay_steps', type=int, default=1000)
parser.add_argument('--controller_decay_factor', type=float, default=0.9)
parser.add_argument('--controller_attention', action='store_true', default=False)
parser.add_argument('--controller_max_gradient_norm', type=float, default=5.0)
parser.add_argument('--controller_time_major', action='store_true', default=False)
parser.add_argument('--controller_symmetry', action='store_true', default=False)
parser.add_argument('--controller_predict_beam_width', type=int, default=0)
parser.add_argument('--controller_predict_lambda', type=float, default=1)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


CIFAR_CLASSES = 10

def train(args):
    args = get_params(args)
    branch_length = args.controller_source_length // 2 // 5 // 2
    args.child_eval_every_epochs = eval(args.child_eval_every_epochs)
  
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = 45000
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=16)
  
    # Train child model
    if args.arch_pool is None:
        arch_pool = utils.generate_arch(args.controller_num_seed_arch, args.child_num_cells, 5) #[[[conv],[reduc]]]
        args.child_arch_pool = arch_pool
        args.child_arch_pool_prob = None
    else:
        if args.child_sample_policy == 'uniform':
            args.child_arch_pool_prob = None
        elif args.child_sample_policy == 'params':
            args.child_arch_pool_prob = calculate_params(args.child_arch_pool)
        elif args.child_sample_policy == 'valid_performance':
            args.child_arch_pool_prob = child_valid(valid_queue, child_model, args.child_arch_pool)
        elif args.child_sample_policy == 'predicted_performance':
            encoder_input = list(map(lambda x: utils.parse_arch_to_seq(x[0], branch_length) + \
                                           utils.parse_arch_to_seq(x[1], branch_length), args.child_arch_pool))
            predicted_error_rate = controller.test(args, encoder_input)
            args.child_arch_pool_prob = [1-i[0] for i in predicted_error_rate]
        else:
            raise ValueError('Child model arch pool sample policy is not provided!')
          
  
    child_model = NASNetwork(args)
    child_model = child_model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(child_model))
    
    optimizer = torch.optim.SGD(
        child_model.parameters(),
        args.child_lr_max,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )
  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.child_num_epochs), eta_min=args.child_lr_min)
    
    #[TODO] controller
    
    for epoch in range(1, args.num_epochs+1):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        # sample an arch to train
        child_train(train_queue, child_model)

        if isinstance(args.child_eval_every_epochs, int):
            if epoch % args.child_eval_every_epochs != 0:
                continue
        else:
            if epoch not in args.child_eval_every_epochs:
                continue
    
        # Evaluate seed archs
        valid_accuracy_list = child_valid(valid_queue, child_model, args.child_arch_pool)

        # Output archs and evaluated error rate
        old_archs = args.child_arch_pool
        old_archs_perf = [1 - i for i in valid_accuracy_list]
    
        old_archs_sorted_indices = np.argsort(old_archs_perf)
        old_archs = np.array(old_archs)[old_archs_sorted_indices].tolist()
        old_archs_perf = np.array(old_archs_perf)[old_archs_sorted_indices].tolist()
        with open(os.path.join(args.child_model_dir, 'arch_pool.{}'.format(epoch)), 'w') as fa:
            with open(os.path.join(args.child_model_dir, 'arch_pool.perf.{}'.format(epoch)), 'w') as fp:
                with open(os.path.join(args.child_model_dir, 'arch_pool'), 'w') as fa_latest:
                    with open(os.path.join(args.child_model_dir, 'arch_pool.perf'),'w') as fp_latest:
                        for arch, perf in zip(old_archs, old_archs_perf):
                            arch = ' '.join(map(str, arch[0] + arch[1]))
                            fa.write('{}\n'.format(arch))
                            fa_latest.write('{}\n'.format(arch))
                            fp.write('{}\n'.format(perf))
                            fp_latest.write('{}\n'.format(perf))
            
        if epoch >= args.child_num_epochs:
            break
  
        # Train Encoder-Predictor-Decoder
        encoder_input = list(map(lambda x : utils.parse_arch_to_seq(x[0], branch_length) + \
                                          utils.parse_arch_to_seq(x[1], branch_length), old_archs))
        #[[conv, reduc]]
        min_val = min(old_archs_perf)
        max_val = max(old_archs_perf)
        encoder_target = [(i - min_val)/(max_val - min_val) for i in old_archs_perf]
        decoder_target = copy.copy(encoder_input)
        args.controller_batches_per_epoch = np.ceil(len(encoder_input) / args.controller_batch_size)
        #if clean controller model
        controller.train(args, encoder_input, encoder_target, decoder_target)
        
        # Generate new archs
        #old_archs = old_archs[:450]
        new_archs = []
        max_step_size = 100
        args.controller_predict_lambda = 0
        top100_archs = list(map(lambda x : utils.parse_arch_to_seq(x[0], branch_length) + \
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
                #[[conv, reduc]]
        new_archs = list(map(lambda x: utils.parse_seq_to_arch(x, branch_length), new_archs)) #[[[conv],[reduc]]]
        num_new_archs = len(new_archs)
        logging.info("Generate {} new archs".format(num_new_archs))
        new_arch_pool = old_archs[:len(old_archs)-(num_new_archs+50)] + new_archs + utils.generate_arch(50, 5, 5)
        logging.info("Totally {} archs now to train".format(len(new_arch_pool)))
        args.child_arch_pool = new_arch_pool
        with open(os.path.join(args.child_model_dir, 'arch_pool'), 'w') as f:
            for arch in new_arch_pool:
                arch = ' '.join(map(str, arch[0] + arch[1]))
                f.write('{}\n'.format(arch))
      

def get_params(args):
    args.child_model_dir = os.path.join(args.output_dir, 'child')
    args.controller_model_dir = os.path.join(args.output_dir, 'controller')
    if args.child_arch_pool is not None:
        logging.info('Architecture pool is provided, loading')
        with open(args.child_arch_pool) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            args.arch_pool = archs
    if os.path.exists(os.path.join(args.child_model_dir, 'arch_pool')):
        logging.info('Found arch_pool in child model dir, loading')
        with open(os.path.join(args.child_model_dir, 'arch_pool')) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            args.arch_pool = archs
    return args


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)
  train(args)
  

if __name__ == '__main__':
  main()
