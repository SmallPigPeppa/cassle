import torch.nn as nn
import torch
import numpy as np


class Conv3x3_modified(nn.Module):
    """
    add bn into conv
    self.expansion_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    self.expansion_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    self.expansion_3x3 = conv3x3(in_planes, out_planes, stride=stride, groups=groups, dilation=dilation)
    self.expansion_3x3.weight.data.zero_()
    """

    def __init__(self, in_planes, out_planes, stride=1, groups=1, dilation=1, use_expansion=False):
        super(Conv3x3_modified, self).__init__()
        self.outplanes = out_planes
        self.conv2d_3x3 = nn.Sequential()
        self.conv2d_3x3.add_module('conv',
                                   conv3x3_bias(in_planes, out_planes, stride=stride, groups=groups, dilation=dilation))
        self.conv2d_3x3.add_module('bn', nn.BatchNorm2d(out_planes))

        self.expansion_3x3 = nn.Sequential()
        self.expansion_3x3.add_module('conv',
                                      conv3x3_nobias(in_planes, out_planes, stride=stride, groups=groups, dilation=dilation))
        self.expansion_3x3.add_module('bn', nn.BatchNorm2d(out_planes))

        self.use_expansion = use_expansion

    def forward(self, x):
        if not self.use_expansion:
            return self.conv2d_3x3(x)
        else:
            with torch.no_grad():
                out1 = self.conv2d_3x3(x)
            return self.expansion_3x3(x) + out1

    def set_expansion(self, use_expansion=True):
        self.use_expansion = use_expansion

    # @torch.no_grad()
    # def clean_expansion(self):
    #     # expansion_3x3.conv clean weight
    #     nn.init.constant_(self.expansion_3x3.conv.weight.data, 0)
    #     nn.init.constant_(self.expansion_3x3.conv.bias.data, 0)
    #     # expansion_3x3.bn clean weight
    #     nn.init.constant_(self.expansion_3x3.bn.weight, 1)
    #     nn.init.constant_(self.expansion_3x3.bn.bias, 0)
    #     # print('############running_mean#########')
    #     # print(self.expansion_3x3.bn.running_mean)
    #     # print('############running_var#########')
    #     # print(self.expansion_3x3.bn.running_var)
    #     nn.init.constant_(self.expansion_3x3.bn.running_mean, 0)
    #     nn.init.constant_(self.expansion_3x3.bn.running_var, 0)
    #
    #     # conv2d_3x3.bn clean weight
    #     nn.init.constant_(self.conv2d_3x3.bn.weight, 1)
    #     nn.init.constant_(self.conv2d_3x3.bn.bias, 0)
    #     nn.init.constant_(self.conv2d_3x3.bn.running_var, 1)
    #     nn.init.constant_(self.conv2d_3x3.bn.running_mean, 0)

    @torch.no_grad()
    def clean_expansion(self):
        self.conv2d_3x3.bn = nn.BatchNorm2d(self.outplanes,affine=False,track_running_stats=False)
        # self.conv2d_3x3.bn=nn.Identity()
        self.expansion_3x3.bn = nn.BatchNorm2d(self.outplanes)
        nn.init.constant_(self.expansion_3x3.conv.weight.data, 0)

        # nn.init.constant_(self.expansion_3x3.conv.bias.data, 0)


    @torch.no_grad()
    def re_parameterize(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv2d_3x3)
        expansion_kernel3x3, expansion_bias3x3 = self._fuse_bn_tensor(self.expansion_3x3)
        kernel = kernel3x3 + expansion_kernel3x3
        bias = bias3x3 + expansion_bias3x3
        self.conv2d_3x3.conv.weight.data = kernel
        self.conv2d_3x3.conv.bias.data = bias

        # expansion_3x3.conv expansion_3x3.bn clean weight
        # self.clean_expansion()

    # @torch.no_grad()
    # def re_parameterize(self):
    #     conv2d_3x3_bn_k, conv2d_3x3_bn_b = self._fuse_bn_tensor(self.conv2d_3x3.bn)
    #     expansion_3x3_k, expansion_3x3_b = self._fuse_bn_tensor(self.expansion_3x3)
    #     # print('expansion_3x3_k:',expansion_3x3_k.shape)
    #     # print('expansion_3x3_b:',expansion_3x3_b.shape)
    #     # print('conv2d_3x3_bn_b:', conv2d_3x3_bn_b.shape)
    #     # print('conv2d_3x3_bn_k:', conv2d_3x3_bn_k.shape)
    #     addtion_k = expansion_3x3_k / conv2d_3x3_bn_k.reshape(-1, 1, 1, 1)
    #     addtion_b = expansion_3x3_b / conv2d_3x3_bn_k
    #     self.conv2d_3x3.conv.weight.data += addtion_k
    #     self.conv2d_3x3.conv.bias.data += addtion_b

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


def conv3x3_bias(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)

def conv3x3_nobias(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3_modified(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return Conv3x3_modified(in_planes, out_planes, stride=stride,
                            groups=groups, dilation=dilation)
