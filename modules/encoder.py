#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
Author: Zhongxi Qiu
FileName: encoder.py
Time: 2020/09/26 15:49:04
Version: 1.0
'''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch.nn as nn
from torch.nn import functional as F 

from .utils import *




class Encoder(nn.Module):
    '''
        Implementation of the Encoder in our framework.
    '''
    def __init__(self, in_ch, out_ch=2048, dilations=[1, 1, 1, 1], strides=[1, 1, 1, 2], bn=nn.BatchNorm2d, nolinear=nn.ReLU(inplace=True)):
        '''
            Initialize the module.
            @in_ch: int, the number of channels of inputs
            @out_ch: int, the number of channels of outputs
            @dilations: list, the rates of dilation for each stage
            @strides: list, the stride of each stage
            @bn: nn.Module, the batch normalization module.
            @nolinear: nn.Module, the nolinear function module
        '''
        super(Encoder, self).__init__()
        self.inplanes = 64
        self.conv1 = Conv2d(in_ch, self.inplanes, ksize=3, stride=2, padding=1, bn=bn, nolinear=nolinear)
        self.conv2 = Conv2d(self.inplanes, self.inplanes, ksize=3, stride=1, padding=1, bn=bn, nolinear=nolinear)
        self.conv3 = Conv2d(self.inplanes, self.inplanes, ksize=3, stride=1, padding=1, bn=bn, nolinear=nolinear)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self.__make_layer(Block, 64, 2, stride=strides[0], dilation=dilations[0], semodule=None, nolinear=nolinear, bn=bn)
        self.layer2 = self.__make_layer(Block, 128, 2, stride=strides[1], dilation=dilations[1], semodule=SEModule, nolinear=nolinear, bn=bn)
        self.layer3 = self.__make_layer(Block, 256, 2, stride=strides[2], dilation=dilations[2], semodule=SEModule, nolinear=nolinear, bn=bn)
        self.layer4 = self.__make_layer(Block, 512, 2, stride=strides[3], dilation=dilations[3], semodule=None, nolinear=nolinear, bn=bn)
        self.out_conv = Conv2d(self.inplanes, out_ch, ksize=1, stride=1, padding=0, nolinear=nolinear, bn=bn)


    def __make_layer(self, block, planes, blocks,stride=1, dilation=1, 
                            bn=nn.BatchNorm2d, nolinear=nn.ReLU(inplace=True), semodule=None, sigmoid=nn.Sigmoid()):
        '''
            Build the stage in the model.
            @block: nn.Module, the block module
            @planes: int, the base channels
            @stride: int, the stride for the first block in the stage
            @bn: nn.Module, the batch normalization module
            @nolinear: nn.Module, the nolinear function module
            @semodule, nn.Module, the Squeeze-and-Excitation module in SENet
            @sigmoid: nn.Module, sigmoid function for the SEModule
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample =  Conv2d(self.inplanes, planes * block.expansion, ksize=3, stride=stride, padding=1, nolinear=None, bn=bn)
        layers = []
        if semodule is not None:
            semodule = semodule(planes * block.expansion, sigmoid=sigmoid, bn=bn, nolinear=nolinear)
        layers.append(block(self.inplanes, planes,stride=stride, downsample=downsample, bn=bn, nolinear=nolinear, semodule=semodule))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1,  bn=bn, nolinear=nolinear, semodule=semodule))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.maxpool(net)
        net = self.layer1(net)
        net = self.layer2(net)
        net = self.layer3(net)
        net = self.layer4(net)
        net = self.out_conv(net)
        return net

if __name__ == "__main__":
    import torch
    net = Encoder(3)
    x = torch.randn((2, 3, 224, 224))
    net = net.cuda()
    x = x.cuda()
    with torch.no_grad():
        out = net(x)
        print(out.shape)