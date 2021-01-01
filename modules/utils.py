#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
Author: Zhongxi Qiu
FileName: utils.py
Time: 2020/09/13 11:39:00
Version: 1.0
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn 

__all__ = ["Conv2d", "Block", "SEModule", "BasicBlock"]

class Conv2d(nn.Module):
    '''
    Implementation of convolutional layers with bn and nolinear activation function
    '''
    def __init__(
        self, in_ch, out_ch, ksize, stride=1, padding=0, dilation=1,groups=1, bias=False,bn=nn.BatchNorm2d, nolinear=nn.ReLU(inplace=True)
    ):
        '''
            Initialize this module's object
            @in_ch: int, the channels of inputs
            @out_ch: int, the channels of output
            @ksize: [int,tuple], the size of the conv kernel
            @stride: [int, tuple], the stride of conv operation
            @padding: int, the size of padding for this operation
            @dilation: int, the atrous rate of the conv
            @groups: int, the number of groups for the conv
            @bias: boolean, whether use the bias
            @bn: nn.Module, the batch normalization operation
            @nolinear: nn.Module, the nolinear activation function
        '''
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = None
        if bn is not None:
            self.bn = bn(out_ch)
        self.nolinear = nolinear
    
    def forward(self, x):
        '''
            In the dynamic compute graph framework such as pytorch, which will bulid the graph in the forward part.
        '''
        net = self.conv(x)
        if self.bn is not None:
            net = self.bn(net)
        if self.nolinear is not None:
            net = self.nolinear(net)
        return net

class SEModule(nn.Module):
    '''
        Implementation of semodule in SENet and MobileNetV3, there we use 1x1 conv replace the linear layer.
        SENet:"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>
        MobileNetV3: "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>
    '''
    def __init__(self, in_ch, reduction=16, sigmoid=nn.Sigmoid(), bn=nn.BatchNorm2d, nolinear=nn.ReLU()):
        '''
            Initialize the module.
            @in_ch: int, the number of channels of input,
            @reduction: int, the coefficient of dimensionality reduction
            @sigmoid: nn.Module, the sigmoid function, in MobilenetV3 is H-Sigmoid and in SeNet is sigmoid
            @bn: nn.Module, the batch normalization moldule
            @nolinear: nn.Module, the nolinear function module
        '''
        super(SEModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            Conv2d(in_ch, in_ch // reduction, ksize=1, stride=1, padding=0, bn=bn, nolinear=nolinear),
            Conv2d(in_ch // reduction, in_ch, ksize=1, stride=1, padding=0, bn=bn, nolinear=sigmoid)
        )
    def forward(self, x):
        net = self.avgpool(x)
        net = self.fc(net)
        return net

class Block(nn.Module):
    '''
    Implementation the Bottleblock in ResNet. We also implementate the block that introduced in SENet.
    ResNet:"Deep Residual Learning for Image Recognition"<https://arxiv.org/pdf/1512.03385.pdf>
    SENet:"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>
    '''
    expansion = 4
    def __init__(self, in_ch, planes, stride=1, downsample=None, bn=nn.BatchNorm2d, nolinear=nn.ReLU, semodule=None):
        '''
            Initialize the module.
            @in_ch: int, the number of channels of input,
            @planes: int, the base channels for the block
            @stride: int, the stride of this block
            @downsample: nn.Module, the downsample part for the block
            @bn: nn.Module, the batch normalization moldule
            @nolinear: nn.Module, the nolinear function module
            @semodule: nn.Module, the Squeeze-and-Excitation module in SENet
        '''
        super(Block, self).__init__()
        self.conv1 = Conv2d(in_ch, planes, ksize=1, stride=1, padding=0, bn=bn, nolinear=nolinear)
        self.conv2 = Conv2d(planes, planes, ksize=3, stride=stride, padding=1, bn=bn, nolinear=nolinear)
        self.conv3 = Conv2d(planes, planes*self.expansion, ksize=1, stride=1, padding=0, bn=bn, nolinear=None)
        self.downsample = downsample
        self.se = semodule
        self.nolinear = nolinear
    
    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        if self.se is not None:
            net = self.se(net)
        if self.downsample is not None:
            x = self.downsample(x)
        net = net + x
        net = self.nolinear(net)
        return net

class BasicBlock(nn.Module):
    '''
    Implementation the Basicblock in ResNet. We also implementate the block that introduced in SENet.
    ResNet:"Deep Residual Learning for Image Recognition"<https://arxiv.org/pdf/1512.03385.pdf>
    SENet:"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>
    '''
    expansion = 1
    def __init__(self, in_ch, planes, stride=1, downsample=None, bn=nn.BatchNorm2d, nolinear=nn.ReLU, semodule=None):
        '''
            Initialize the module.
            @in_ch: int, the number of channels of input,
            @planes: int, the base channels for the block
            @stride: int, the stride of this block
            @downsample: nn.Module, the downsample part for the block
            @bn: nn.Module, the batch normalization moldule
            @nolinear: nn.Module, the nolinear function module
            @semodule: nn.Module, the Squeeze-and-Excitation module in SENet
        '''
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_ch, planes, ksize=3, stride=1, padding=1, bn=bn, nolinear=nolinear)
        self.conv2 = Conv2d(planes, planes, ksize=3, stride=stride, padding=1, bn=bn, nolinear=None)
        self.downsample = downsample
        self.se = semodule
        self.nolinear = nolinear
    
    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        if self.se is not None:
            net = self.se(net)
        if self.downsample is not None:
            x = self.downsample(x)
        net = net + x
        net = self.nolinear(net)
        return net