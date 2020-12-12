#-*- coding:utf-8 -*-
#!/etc/env python
'''
   @Author:Zhongxi Qiu
   @File: loss_metrics.py
   @Time: 2020-12-12 20:17:07
   @Version:1.0
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class MissClassification(nn.Module):
    def __init__(self, average=True):
        '''
            The adversarial samples are not equal to the labels.
            args:
                average: bool, whether return the average
        '''
        super(MissClassification, self).__init__()
        self.average = average
    
    def forward(self, preds, labels):
        adv = preds != labels
        assert adv.shape != labels.shape, "The shape of adv are not equal to the labels, got {}/{}".format(adv.shape, labels.shape)
        if self.average:
            return adv.mean(dim=-1)
        else:
            return adv.sum()