import torch.nn as nn
import torch
import numpy as np


# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


class Conv3x3_mofied(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1, dilation=1, use_expansion=False):
        super(Conv3x3_mofied, self).__init__()
        # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.conv2d_3x3 = conv3x3(in_planes, out_planes, stride=stride, groups=groups, dilation=dilation)
        self.expansion_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.expansion_1x1 = nn.Identity()
        # nn.init.constant_(self.expansion_1x1.weight.data, 0.0)
        # self.expansion_1x1.weight.data.zero_()

        self.use_expansion = use_expansion

    def set_expansion(self, use_expansion=True):
        self.use_expansion = use_expansion

    def re_param(self):
        kernel = self.get_equivalent_kernel_bias()
        self.conv2d_3x3.weight.data = kernel
        self.clean_expansion()

    def clean_expansion(self):
        nn.init.constant_(self.expansion_1x1.weight.data, 0.0)
        # self.expansion_1x1.weight.data.zero_()

    def forward(self, x):
        if not self.use_expansion:
            return self.conv2d_3x3(x)
        else:
            print('#########use expansion############')
            with torch.no_grad():
                out1 = self.conv2d_3x3(x)
            return self.expansion_1x1(x) + out1

    def get_equivalent_kernel_bias(self):
        # bias no use
        kernel3x3 = self.conv2d_3x3.weight
        kernel1x1 = self.expansion_1x1.weight
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


if __name__ == '__main__':
    x = torch.rand([4, 3, 32, 32])
    c1 = conv3x3(3, 4)
    # c2=Conv3x3_mofied(3,4)
    c2 = conv3x3(3, 4)
    print(c1(x)[0][0][0])
    # print(c2.conv2d_3x3(x)[0][0][0])
    print(c2(x)[0][0][0])
