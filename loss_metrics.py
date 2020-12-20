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
    def __init__(self):
        '''
            The adversarial samples are not equal to the labels.
            args:
                average: bool, whether return the average
        '''
        super(MissClassification, self).__init__()
    
    def forward(self, preds, labels):
        preds = torch.softmax(preds, dim=1)
        preds = torch.max(preds, dim=1)[1]
        adv = (preds != labels)
        #assert adv.shape == labels.shape, "The shape of adv are not equal to the labels, got {}/{}".format(adv.shape, labels.shape)
      
        return 1 - adv.sum() / adv.size()[0]
