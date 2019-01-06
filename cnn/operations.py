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


class WSReLUConvBN(nn.Module):
    def __init__(self, num_possible_inputs, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(WSReLUConvBN, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.w = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, kernel_size, kernel_size)) for i in range(num_possible_inputs)])
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x, x_id):
        x = self.relu(x)
        if isinstance(x_id, int):
            w = self.w[x_id]
        else:
            assert isinstance(x_id, list)
            w = torch.cat([self.w[i] for i in x_id], dim=1)
        x = F.conv2d(x, w, stride=self.stride, padding=self.padding)
        x = self.bn(x)
        return x


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


class WSSepConv(nn.Module):
    def __init__(self, num_possible_inputs, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(WSSepConv, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.C_out = C_out
        self.C_in = C_in
        self.stride = stride
        self.padding = padding
        
        self.relu1 = nn.ReLU(inplace=False)
        self.W1_depthwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_in, 1, kernel_size, kernel_size)) for i in range(num_possible_inputs)])
        self.W1_pointwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, 1, 1)) for i in range(num_possible_inputs)])
        self.bn1 = nn.BatchNorm2d(C_in, affine=affine)

        self.relu2 = nn.ReLU(inplace=False)
        self.W2_depthwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_in, 1, kernel_size, kernel_size)) for i in range(num_possible_inputs)])
        self.W2_pointwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, 1, 1)) for i in range(num_possible_inputs)])
        self.bn2 = nn.BatchNorm2d(C_in, affine=affine)
    
    def forward(self, x, x_id):
        x = self.relu1(x)
        x = F.conv2d(x, self.W1_depthwise[x_id], stride=self.stride, padding=self.padding, groups=self.C_in)
        x = F.conv2d(x, self.W1_pointwise[x_id], padding=0)
        x = self.bn1(x)

        x = self.relu2(x)
        x = F.conv2d(x, self.W2_depthwise[x_id], padding=self.padding, groups=self.C_in)
        x = F.conv2d(x, self.W2_pointwise[x_id], padding=0)
        x = self.bn2(x)
        return x


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
    def __init__(self, layers, channels, affine=True):
        super(MaybeCalibrateSize, self).__init__()
        self.channels = channels
        hw = [layer[0] for layer in layers]
        c = [layer[-1] for layer in layers]
        
        x_out_shape = [hw[0], hw[1], c[0]]
        y_out_shape = [hw[1], hw[1], c[1]]
        if hw[0] != hw[1]:
            assert hw[0] == 2 * hw[1]
            self.preprocess_x = nn.Sequential(nn.ReLU(), FactorizedReduce(c[0], channels))
            x_out_shape = [hw[1], hw[1], channels]
        elif c[0] != channels:
            self.preprocess_x = ReLUConvBN(c[0], channels, 1, 1, 0)
            x_out_shape = [hw[0], hw[0], channels]
        if c[1] != channels:
            self.preprocess_y = ReLUConvBN(layers[1][-1], channels, 1, 1, 0)
            y_out_shape = [hw[1], hw[1], channels]
            
        self.out_shape = [x_out_shape, y_out_shape]
    
    def forward(self, s0, s1):
        if s0.size(2) != s1.size(2) or s0.size(1) != self.channels:
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
        self.ops = {}
        for i, layer in enumerate(layers):
            if i in concat:
                hw = layer[0]
                if hw > out_hw:
                    assert hw == 2 * out_hw
                    self.ops[i] = FactorizedReduce(layer[-1], channels)
        
    def forward(self, states):
        for i, state in enumerate(states):
            if i in self.concat:
                if state.size(2) > self.out_hw:
                    states[i] = self.ops[i](state)
        out = torch.cat([states[i] for i in self.concat], dim=1)
        return out