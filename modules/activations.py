#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
Author: Zhongxi Qiu
FileName: activations.py
Time: 2020/09/13 15:42:15
Version: 1.0
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import Conv2d

__all__ = ["HSwish", "HSigmoid", "Mish", "FReLU", "FReLU_Light"]

class HSwish(nn.Module):
    '''
    Implementation of hswish activation function in "Searching for MobileNetV3"
    paper link:https://arxiv.org/pdf/1905.02244.pdf
    h-swish = x * relu6(x+3) / 6
    '''
    def __init__(self):
        super(HSwish, self).__init__()

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class HSigmoid(nn.Module):
    '''
    Implementation of hsigmoid activation function in "Searching for MobileNetV3"
    paper link:https://arxiv.org/pdf/1905.02244.pdf
    h-sigmoid = relu6(x+3) / 6
    '''
    def __init__(self):
        super(HSigmoid, self).__init__()

    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class Mish(nn.Module):
    '''
        Implementation of mish activation function in "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        paper link:
        mish = x * tanh(ln(1+e^x))
    '''
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        mish = x * F.tanh(F.softplus(x))
        return mish

class FReLU(nn.Module):
    '''
        implementation of FReLU in "FReLU-Funnel Activation for Visual Recognition"
        paper:https://arxiv.org/abs/2007.11824
        frelu = max(x, t(x))
    '''
    def __init__(self, in_ch):
        super(FReLU, self).__init__()
        self.frelu_conv = Conv2d(in_ch, in_ch, 3, stride=1, padding=1, groups=in_ch, nolinear=None)
    
    def forward(self, x):
        x1 = self.frelu_conv(x)
        return torch.max(x, x1)

class FReLU_Light(nn.Module):
    '''
    implementation of FReLU for light net in "FReLU-Funnel Activation for Visual Recognition"
    paper:https://arxiv.org/abs/2007.11824
    frelu = max(x, t(x))
    '''
    def __init__(self, in_ch):
        super(FReLU_Light, self).__init__()
        self.frelu_conv1 = Conv2d(in_ch, in_ch, (1, 3), stride=1, padding=(0, 1), groups=in_ch, nolinear=None)
        self.frelu_conv2 = Conv2d(in_ch, in_ch, (3, 1), stride=1, padding=(1, 0), groups=in_ch, nolinear=None)
    
    def forward(self, x):
        x1 = self.frelu_conv1(x)
        x2 = self.frelu_conv2(x)
        return torch.max(x, x1 + x2)

class Swish(nn.Module):
    '''
        Implementation of Swish.
        "Searching for Activation functions"<https://arxiv.org/abs/1710.05941>
        Swish = x * sigmoid(beta*x)
    '''
    def __init__(self, beta=1.0):
        '''
            Initialize this module, set the parameters for the object.
            @beta: float, the value of beta in the function 
        '''
        super(Swish, self).__init__()
        self.beta = beta
    
    def forward(self, x):
        '''
            Propogation, build the computer graph
            @x: tensor, the inputs
            @return: tensor, the result after propogation
        '''
        swish = x*F.sigmoid(x*self.beta)
        return swish