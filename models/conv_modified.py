import torch.nn as nn
import torch
import numpy as np


class Conv3x3_mofied(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1, dilation=1,use_expansion=False):
        super(Conv3x3_mofied, self).__init__()
        # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.conv2d_3x3 = conv3x3(in_planes, out_planes, stride=stride, groups=groups, dilation=dilation)
        # self.expansion_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.expansion_3x3 = conv3x3(in_planes, out_planes, stride=stride, groups=groups, dilation=dilation)
        self.expansion_3x3.weight.data.zero_()
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

    def re_parameterize(self):
        # kernel = self.get_equivalent_kernel_bias()
        with torch.no_grad():
            self.conv2d_3x3.weight.data = self.conv2d_3x3.weight.data+self.expansion_3x3.weight.data
        self.expansion_3x3.weight.data.zero_()

    # def get_equivalent_kernel_bias(self):
    #     # bias no use
    #     kernel3x3, _ = self._fuse_bn_tensor(self.conv2d_3x3)
    #     kernel1x1, _ = self._fuse_bn_tensor(self.expansion_3x3)
    #     return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1)
    #
    # def _pad_1x1_to_3x3_tensor(self, kernel1x1):
    #     if kernel1x1 is None:
    #         return 0
    #     else:
    #         return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    #
    # def _fuse_bn_tensor(self, branch):
    #     if branch is None:
    #         return 0, 0
    #     if isinstance(branch, nn.Module):
    #         kernel = branch.weight
    #         return kernel, kernel
    #     if isinstance(branch, nn.Sequential):
    #         kernel = branch.conv.weight
    #         running_mean = branch.bn.running_mean
    #         running_var = branch.bn.running_var
    #         gamma = branch.bn.weight
    #         beta = branch.bn.bias
    #         eps = branch.bn.eps
    #     else:
    #         assert isinstance(branch, nn.BatchNorm2d)
    #         if not hasattr(self, 'id_tensor'):
    #             input_dim = self.in_channels // self.groups
    #             kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
    #             for i in range(self.in_channels):
    #                 kernel_value[i, i % input_dim, 1, 1] = 1
    #             self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
    #         kernel = self.id_tensor
    #         running_mean = branch.running_mean
    #         running_var = branch.running_var
    #         gamma = branch.weight
    #         beta = branch.bias
    #         eps = branch.eps
    #     std = (running_var + eps).sqrt()
    #     t = (gamma / std).reshape(-1, 1, 1, 1)
    #     return kernel * t, beta - running_mean * gamma / std


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3_modified(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return Conv3x3_mofied(in_planes, out_planes, stride=stride,
                           groups=groups,dilation=dilation)

def conv3x3_modified_padding(in_planes, out_planes, stride=1, groups=1, dilation=1,padding=1):
    conv_m=Conv3x3_mofied(in_planes, out_planes, stride=stride,
                           groups=groups,dilation=dilation)
    conv_m.conv2d_3x3=nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)
    return conv_m

class BasicBlock_Modified(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_Modified, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
