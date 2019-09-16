import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

INPLACE=False
BIAS=False


def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
    layer_ratio = float(layer_id+1) / (layers)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
    step_ratio = float(step + 1) / steps
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
    if drop_path_keep_prob < 1.:
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(drop_path_keep_prob).cuda()
        #x.div_(drop_path_keep_prob)
        #x.mul_(mask)
        x = x / drop_path_keep_prob * mask
    return x


class AuxHeadCIFAR(nn.Module):
    def __init__(self, C_in, classes):
        """assuming input size 8x8"""
        super(AuxHeadCIFAR, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)
        
    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxHeadImageNet(nn.Module):
    def __init__(self, C_in, classes):
        """input should be in [B, C, 7, 7]"""
        super(AuxHeadImageNet, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)
    
    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):
        super(ReLUConvBN, self).__init__()
        self.relu = nn.ReLU(inplace=INPLACE)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.multi_adds = C_in * C_out * kernel_size * kernel_size * (shape[0] // stride) * (shape[0] // stride)
    
    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Conv, self).__init__()
        if isinstance(kernel_size, int):
            self.ops = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=affine)
            )
        else:
            assert isinstance(kernel_size, tuple)
            k1, k2 = kernel_size[0], kernel_size[1]
            self.ops = nn.Sequential(
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_out, C_out, (k1, k2), stride=(1, stride), padding=padding[0], bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=INPLACE),
                nn.Conv2d(C_out, C_out, (k2, k1), stride=(stride, 1), padding=padding[1], bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x, bn_train=False):
        x = self.ops(x)
        return x
    
    
class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
        self.multi_adds = 2 * (shape[0] // stride) * (shape[1] // stride) * ( kernel_size * kernel_size * C_in + C_in * C_out)

    def forward(self, x):
        return self.op(x)


class DilSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, shape, affine=True):
        super(DilSepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
      )
        self.multi_adds = (shape[0] // stride) * (shape[1] // stride) * (kernel_size * kernel_size * C_in + C_in * C_out)

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.multi_adds = 0
    
    def forward(self, x):
        return x


class Zero(nn.Module):
  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride
    self.multi_adds = 0

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPool, self).__init__()
        self.op = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.multi_adds = 0
    
    def forward(self, x):
        return self.op(x)


class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, count_include_pad=False):
        super(AvgPool, self).__init__()
        self.op = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=count_include_pad)
        self.multi_adds = 0
    
    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, shape, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.path1 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.path2 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.multi_adds = 2 * 1 * 1 * C_in * C_out // 2 * (shape[0] // 2) * (shape[0] // 2)
    
    def forward(self, x):
        path1 = x
        path2 = F.pad(x, (0, 1, 0, 1), "constant", 0)[:, :, 1:, 1:]
        out = torch.cat([self.path1(path1), self.path2(path2)], dim=1)
        out = self.bn(out)
        return out


class MaybeCalibrateSize(nn.Module):
    def __init__(self, layers, channels, affine=True):
        super(MaybeCalibrateSize, self).__init__()
        self.channels = channels
        self.multi_adds = 0
        hw = [layer[0] for layer in layers]
        c = [layer[-1] for layer in layers]
        
        x_out_shape = [hw[0], hw[0], c[0]]
        y_out_shape = [hw[1], hw[1], c[1]]
        # previous reduction cell
        if hw[0] != hw[1]:
            assert hw[0] == 2 * hw[1]
            self.relu = nn.ReLU(inplace=INPLACE)
            self.preprocess_x = FactorizedReduce(c[0], channels, [hw[0], hw[0], c[0]], affine)
            x_out_shape = [hw[1], hw[1], channels]
            self.multi_adds += self.preprocess_x.multi_adds
        elif c[0] != channels:
            self.preprocess_x = ReLUConvBN(c[0], channels, 1, 1, 0, [hw[0], hw[0]], affine)
            x_out_shape = [hw[0], hw[0], channels]
            self.multi_adds += self.preprocess_x.multi_adds
        if c[1] != channels:
            self.preprocess_y = ReLUConvBN(c[1], channels, 1, 1, 0, [hw[1], hw[1]], affine)
            y_out_shape = [hw[1], hw[1], channels]
            self.multi_adds += self.preprocess_y.multi_adds
            
        self.out_shape = [x_out_shape, y_out_shape]
    
    def forward(self, s0, s1):
        if s0.size(2) != s1.size(2):
            s0 = self.relu(s0)
            s0 = self.preprocess_x(s0)
        elif s0.size(1) != self.channels:
            s0 = self.preprocess_x(s0)
        if s1.size(1) != self.channels:
            s1 = self.preprocess_y(s1)
        return [s0, s1]


class FinalCombine(nn.Module):
    def __init__(self, layers, out_hw, channels, concat, affine=True):
        super(FinalCombine, self).__init__()
        self.out_hw = out_hw
        self.channels = channels
        self.concat = concat
        self.ops = nn.ModuleList()
        self.concat_fac_op_dict = {}
        self.multi_adds = 0
        for i in concat:
            hw = layers[i][0]
            if hw > out_hw:
                assert hw == 2 * out_hw and i in [0,1]
                self.concat_fac_op_dict[i] = len(self.ops)
                op = FactorizedReduce(layers[i][-1], channels, [hw, hw], affine)
                self.ops.append(op)
                self.multi_adds += op.multi_adds
        
    def forward(self, states):
        for i in self.concat:
            if i in self.concat_fac_op_dict:
                states[i] = self.ops[self.concat_fac_op_dict[i]](states[i])
        out = torch.cat([states[i] for i in self.concat], dim=1)
        return out


OPERATIONS_CIFAR = {
    0: lambda c_in, c_out, stride, shape, affine: Zero(stride),
    1: lambda c_in, c_out, stride, shape, affine: Identity() if stride == 1 else FactorizedReduce(c_in, c_out, shape, affine=affine),
    2: lambda c_in, c_out, stride, shape, affine: SepConv(c_in, c_out, 3, stride, 1, shape, affine),
    3: lambda c_in, c_out, stride, shape, affine: SepConv(c_in, c_out, 5, stride, 2, shape, affine),
    4: lambda c_in, c_out, stride, shape, affine: SepConv(c_in, c_out, 7, stride, 3, shape, affine),
    5: lambda c_in, c_out, stride, shape, affine: DilSepConv(c_in, c_out, 3, stride, 2, 2, shape, affine=affine),
    6: lambda c_in, c_out, stride, shape, affine: DilSepConv(c_in, c_out, 5, stride, 4, 2, shape, affine=affine),
    7: lambda c_in, c_out, stride, shape, affine: DilSepConv(c_in, c_out, 7, stride, 6, 2, shape, affine=affine),
    8: lambda c_in, c_out, stride, shape, affine: MaxPool(3, stride, 1), 
    9: lambda c_in, c_out, stride, shape, affine: AvgPool(3, stride, 1),
}