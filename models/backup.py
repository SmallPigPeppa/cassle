import torch.nn as nn
import torch
import numpy as np

class Conv2d_mofied(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_mode='zeros', use_expansion=False):
        super(Conv2d_mofied, self).__init__()
        self.conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.expansion_1x1 = conv1x1(in_planes, out_planes, stride=1)
        self.use_expansion = use_expansion

    def forward(self, x):
        if not self.use_expansion:
            return self.conv2d(x)
        else:
            with torch.no_grad():
                out1 = self.conv2d(x)
            return self.expansion_1x1(x) + out1

    def set_expansion(self, use_expansion=True):
        self.use_expansion = use_expansion

    def re_parameterize(self):
        kernel = self.get_equivalent_kernel_bias()
        self.conv2d.weight.data = kernel

    def get_equivalent_kernel_bias(self):
        # bias no use
        kernel3x3, _ = self._fuse_bn_tensor(self.conv2d)
        kernel1x1, _ = self._fuse_bn_tensor(self.expansion_1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Module):
            kernel = branch.weight
            return kernel, kernel
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