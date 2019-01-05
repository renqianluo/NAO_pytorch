import os
import numpy as np
import logging
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable

B=5

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
      

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
    
    
def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout_size is not None:
        train_transform.transforms.append(Cutout(args.cutout_size))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)
      

def save(model_path, args, model, epoch, step, optimizer):
    state_dict = {
        'args': args,
        'model': model.state_dict() if model else {},
        'epoch': epoch,
        'step': step,
        'optimizer': optimizer.state_dict()
    }
    filename = os.path.join(model_path, 'checkpoint{}.pt'.format(epoch))
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    torch.save(state_dict(), model_path)
    shutil.copyfile(filename, newest_filename)
  

def load(model_path):
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    if not os.path.exists(newest_filename):
        return None, None, 0, 0, None
    state_dict = torch.load(newest_filename)
    args = state_dict['args']
    model_state_dict = state_dict['model']
    epoch = state_dict['epoch']
    step = state_dict['step']
    optimizer_state_dict = state_dict['optimizer']
    return args, model_state_dict, epoch, step, optimizer_state_dict

  
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makekdirs(path)
        print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def sample_arch(arch_pool, prob=None):
    if prob is not None:
        logging.info('Arch pool prob is provided, sampling according to the prob')
        index = np.random.multinomial(1, prob)
    else:
        index = np.random.uniform([], minval=0, maxval=N, dtype=tf.int32)
    arch = arch_pool[index]
    conv_arch = arch[0]
    reduc_arch = arch[1]
    return conv_arch, reduc_arch


def generate_arch(n, num_nodes, num_ops=7):
    def _get_arch():
        arch = []
        for i in range(2, num_nodes+2):
            p1 = np.random.randint(0, i)
            op1 = np.random.randint(0, num_ops)
            p2 = np.random.randint(0, i)
            op2 = np.random.randint(0 ,num_ops)
            arch.extend([p1, op1, p2, op2])
        return arch
    archs = [[_get_arch(), _get_arch()] for i in range(n)] #[[[conv],[reduc]]]
    return archs

def build_dag(arch):
    if arch is None:
        return None, None
    # assume arch is the format [idex, op ...] where index is in [0, 5] and op in [0, 10]
    arch = list(map(int, arch.strip().split()))
    length = len(arch)
    conv_dag = arch[:length//2]
    reduc_dag = arch[length//2:]
    return conv_dag, reduc_dag

def parse_arch_to_seq(cell, branch_length):
    assert branch_length in [2, 3]
    seq = []
    def _parse_op(op):
        if op == 0:
            return 7, 12
        if op == 1:
            return 8, 11
        if op == 2:
            return 8, 12
        if op == 3:
            return 9, 11
        if op == 4:
            return 10, 11

    for i in range(B):
        prev_node1 = cell[4*i]+1
        prev_node2 = cell[4*i+2]+1
        if branch_length == 2:
            op1 = cell[4*i+1] + 7
            op2 = cell[4*i+3] + 7
            seq.extend([prev_node1, op1, prev_node2, op2])
        else:
            op11, op12 = _parse_op(cell[4*i+1])
            op21, op22 = _parse_op(cell[4*i+3])
            seq.extend([prev_node1, op11, op12, prev_node2, op21, op22]) #nopknopk
    return seq

def parse_seq_to_arch(seq, branch_length):
    n = len(seq)
    assert branch_length in [2, 3]
    assert n // 2 // 5 // 2 == branch_length
    def _parse_cell(cell_seq):
        cell_arch = []
        def _recover_op(op1, op2):
            if op1 == 7:
                return 0
            if op1 == 8:
                if op2 == 11:
                    return 1
                if op2 == 12:
                    return 2
            if op1 == 9:
                return 3
            if op1 == 10:
                return 4
        if branch_length == 2:
            for i in range(B):
                p1 = cell_seq[4*i] - 1
                op1 = cell_seq[4*i+1] - 7
                p2 = cell_seq[4*i+2] - 1
                op2 = cell_seq[4*i+3] - 7
                cell_arch.extend([p1, op1, p2, op2])
            return cell_arch
        else:
            for i in range(B):
                p1 = cell_seq[6*i] - 1
                op11 = cell_seq[6*i+1]
                op12 = cell_seq[6*i+2]
                op1 = _recover_op(op11, op12)
                p2 = cell_seq[6*i+3] - 1
                op21 = cell_seq[6*i+4]
                op22 = cell_seq[6*i+5]
                op2 = _recover_op(op21, op22)
                cell_arch.extend([p1, op1, p2, op2])
            return cell_arch
    conv_seq = seq[:n//2]
    reduc_seq = seq[n//2:]
    conv_arch = _parse_cell(conv_seq)
    reduc_arch = _parse_cell(reduc_seq)
    arch = [conv_arch, reduc_arch]
    return arch


def pairwise_accuracy(la, lb):
    N = len(la)
    assert N == len(lb)
    total = 0
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            if la[i] >= la[j] and lb[i] >= lb[j]:
                count += 1
            if la[i] < la[j] and lb[i] < lb[j]:
                count += 1
            total += 1
    return float(count) / total

def hamming_distance(la, lb):
    N = len(la)
    assert N == len(lb)
  
    def _hamming_distance(s1, s2):
        n = len(s1)
        assert n == len(s2)
        c = 0
        for i, j in zip(s1, s2):
            if i != j:
                c += 1
        return c
  
    dis = 0
    for i in range(N):
        line1 = la[i]
        line2 = lb[i]
        dis += _hamming_distance(line1, line2)
    return dis / N
