import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


OPERATIONS = {
    0: lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine), # sep conv 3 x 3
    1: lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine), # sep conv 5 x 5
    2: lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False), # avg pool 3 x 3
    3: lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1), # max pool 3x 3
    4: lambda C, stride, affine: Identity(), # identity
}

def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
    layer_ratio = float(layer_id+1) / (layers)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
    step_ratio = float(step + 1) / steps
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
    if drop_path_keep_prob < 1.:
        noise_shape = [x.size(0), 1, 1, 1]
        mask = Variable(torch.cuda.FloatTensor(*noise_shape).bernoulli_(drop_path_keep_prob))
        x = x / drop_path_keep_prob * mask
    return x


class AuxHead(nn.Module):
    def __init__(self, C_in):
        super(AuxHead, self).__init__()
        self.ops = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C_in, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Linear(768, 10)
        
    def forward(self, x):
        x = self.ops(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    
    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.path1 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.path2 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x):
        path1 = x
        path2 = F.pad(x, (0, 1, 0, 1), "constant", 0)[:, :, 1:, 1:]
        out = torch.cat([self.path1(path1), self.path2(path2)], dim=1)
        out = self.bn(out)
        return out


class MaybeCalibrateSize(nn.Module):
    def __init__(self, layers, out_filters, affine=True):
        super(MaybeCalibrateSize, self).__init__()
        self.preprocess_x, self.preprocess_y = None, None
        hw = [layer[0] for layer in layers]
        c = [layer[-1] for layer in layers]
        
        x_out_shape = [hw[0], hw[1], c[0]]
        y_out_shape = [hw[1], hw[1], c[1]]
        if hw[0] != hw[1]:
            assert hw[0] == 2 * hw[1]
            self.preprocess_x = nn.Sequential(nn.ReLU(), FactorizedReduce(c[0], out_filters))
            x_out_shape = [hw[1], hw[1], out_filters]
        elif c[0] != out_filters:
            self.preprocess_x = ReLUConvBN(c[0], out_filters, 1, 1, 0)
            x_out_shape = [hw[0], hw[0], out_filters]
        if c[1] != out_filters:
            self.preprocess_y = ReLUConvBN(layers[1][-1], out_filters, 1, 1, 0)
            y_out_shape = [hw[1], hw[1], out_filters]
            
        self.out_shape = [x_out_shape, y_out_shape]
    
    def forward(self, s0, s1):
        if self.preprocess_x is not None:
            s0 = self.preprocess_x(s0)
        if self.preprocess_y is not None:
            s1 = self.preprocess_y(s1)
        return [s0, s1]