#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
Author: Zhongxi Qiu
FileName: decoder.py
Time: 2020/09/27 11:02:18
Version: 1.0
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

from .utils import *
from .activations import *

class UpBlock(nn.Module):
    '''
        Implementation of the upblock for decoder module
    '''
    def __init__(self, in_ch, out_ch, bn=nn.BatchNorm2d, nolinear=nn.ReLU(inplace=True)):
        '''
            Initialize the module.
            @in_ch: int, the number of channels for inputs
            @out_ch: int, the  number of channels for outputs
            @bn: nn.Module, the batch normalization module
            @nolinear: nn.Module, the nolinear function module 
        '''
        super(UpBlock, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.up_conv = Conv2d(in_ch, out_ch, ksize=1, stride=1, padding=0, bn=bn, nolinear=nolinear)
        nolinear = FReLU(out_ch)
        self.conv = BasicBlock(out_ch, out_ch, stride=1, bn=bn, nolinear=nolinear)
    
    def forward(self, x):
        net = self.up(x)
        net = self.up_conv(net)
        net = self.conv(net)
        return net

class Decoder(nn.Module):
    '''
        Implementation of the Decoder in our framework.
    '''
    def __init__(self, in_ch, out_ch, bn=nn.BatchNorm2d, nolinear=nn.ReLU(inplace=True)):
        '''
            Initialize the module.
            @in_ch: int, the number of channels for inputs
            @out_ch: int, the  number of channels for outputs
            @bn: nn.Module, the batch normalization module
            @nolinear: nn.Module, the nolinear function module 
        '''
        super(Decoder, self).__init__()
        self.up1 = UpBlock(in_ch, 512, bn=bn, nolinear=nolinear)
        self.up2 = UpBlock(512, 256, bn=bn, nolinear=nolinear)
        self.up3 = UpBlock(256, 128, bn=bn, nolinear=nolinear)
        self.up4 = UpBlock(128, 64, bn=bn, nolinear=nolinear)
        self.up5 = UpBlock(64, 64, bn=bn, nolinear=nolinear)
        self.out_conv = Conv2d(64, out_ch, ksize=1, stride=1, padding=0, bn=bn, nolinear=nolinear)
    
    def forward(self, x):
        net = self.up1(x)
        net = self.up2(net)
        net = self.up3(net)
        net = self.up4(net)
        net = self.up5(net)
        net = self.out_conv(net)
        return net

if __name__ == "__main__":
    from encoder import Encoder
    x = torch.randn((2, 3, 224, 224)).cuda()
    encoder = Encoder(3).cuda()
    decoder = Decoder(2048, 3).cuda()
    with torch.no_grad():
        out1 = encoder(x)
        out2 = decoder(out1)
        print(out1.shape)
        print(out2.shape)