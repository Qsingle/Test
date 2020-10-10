#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
Author: Zhongxi Qiu
FileName: resnet.py
Time: 2020/10/09 09:35:55
Version: 1.0
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

from .utils import *

def get_layers(n_layers):
    blocks = []
    if n_layers == 18:
        blocks = [2, 2, 2, 2]
    elif n_layers == 34:
        blocks = [3, 4, 6, 3]
    elif n_layers == 50:
        blocks = [3, 4, 6, 3]
    elif n_layers == 101:
        blocks = [3, 4, 23, 3]
    elif n_layers == 152:
        blocks = [3, 8, 36, 3]
    else:
        raise ValueError("Unknown number of layers:{}".format(n_layers))
    return blocks

class ResNet(nn.Module):
    '''
        Implementation of the ResNet.
        ResNet:"Deep Residual Learning for Image Recognition"<https://arxiv.org/pdf/1512.03385.pdf>
    '''
    def __init__(self, in_ch, n_layers=18, num_classes=1000, light_head=False, bn=nn.BatchNorm2d, nolinear=nn.ReLU(inplace=True)):
        '''
            Initialize the module.
            @in_ch: int, the number of channels of inputs
            @n_layers: int, the number of layers
            @num_classes: int, the number  of classes that need predict
            @light_head: boolean, whether use conv3x3 replace the conv7x7
            @bn: nn.Module, the batch normalization module.
            @nolinear: nn.Module, the nolinear function module
        '''
        super(ResNet, self).__init__()
        self.inplanes = 64
        if light_head:
            self.conv1 = nn.Sequential(
                Conv2d(in_ch, self.inplanes, ksize=3, stride=1, padding=1, bn=bn, nolinear=nolinear),
                Conv2d(self.inplanes, self.inplanes, ksize=3, stride=2, padding=1, bn=bn, nolinear=nolinear),
                Conv2d(self.inplanes, self.inplanes, ksize=3, stride=1, padding=1, bn=bn, nolinear=nolinear)
            )
        else:
            self.conv1 = Conv2d(in_ch, self.inplanes, ksize=7, stride=2, padding=3, bn=bn, nolinear=nolinear)
        block = Block
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
        if n_layers < 50:
            block = BasicBlock
        blocks = get_layers(n_layers)
        self.layer1 = self.__make_layer(block, 64, blocks[0], stride=1, semodule=None)
        self.layer2 = self.__make_layer(block, 128, blocks[1], stride=2, semodule=None)
        self.layer3 = self.__make_layer(block, 256, blocks[2], stride=2, semodule=None)
        self.layer4 = self.__make_layer(block, 512, blocks[3], stride=2, semodule=None)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.inplanes, num_classes, bias=False),
        ) 

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
        net = self.maxpool(net)
        net = self.layer1(net)
        net = self.layer2(net)
        net = self.layer3(net)
        net = self.layer4(net)
        feature = net
        net = self.pool(net)
        net = self.fc(net)
        return net, feature

if __name__ == "__main__":
    x = torch.randn((1, 3, 224, 224))
    model = ResNet(3)
    model = model.cuda()
    x = x.cuda()
    with torch.no_grad():
        out=model(x)
        print(out.shape)