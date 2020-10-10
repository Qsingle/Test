#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
Author: Zhongxi Qiu
FileName: train.py
Time: 2020/10/09 10:10:31
Version: 1.0
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as opt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as tsf
from sklearn.metrics import accuracy_score, f1_score

from modules.decoder import Decoder
from modules.encoder import Encoder
from modules.resnet import ResNet

img_size = 64
batch_size = 32
num_workers = 4
num_classes = 10
channels = 3
epochs = 32

train_transform = tsf.Compose([
    tsf.Resize((img_size, img_size)),
    tsf.RandomHorizontalFlip(),
    tsf.RandomVerticalFlip(),
    tsf.ToTensor(),
    tsf.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

val_transform =  tsf.Compose([
    tsf.Resize((img_size, img_size)),
    tsf.ToTensor(),
    tsf.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataset = CIFAR10("./data", train=True, transform=train_transform, download=True)
test_dataset = CIFAR10("./data", train=False, transform=val_transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

encoder = Encoder(in_ch=channels)
decoder = Decoder(in_ch=2048, out_ch=channels)

classifier = ResNet(channels, n_layers=50, num_classes=num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)
classifier.to(device)

encoder_opt = opt.Adam(encoder.parameters(), lr=0.01, weight_decay=5e-3)
decoder_opt = opt.Adam(decoder.parameters(), lr=0.01, weight_decay=5e-3)
classifier_opt = opt.Adam(classifier.parameters(), lr=0.01, weight_decay=5e-3)

c_loss = nn.CrossEntropyLoss()
a_loss = nn.CrossEntropyLoss()
hinge_loss = nn.L1Loss()
margin_loss = nn.MSELoss()
reconstruct_loss = nn.L1Loss()

for epoch in range(epochs):
    for x, label in train_loader:
        x = x.to(device, dtype=torch.float32)
        label = label.to(device)
        target = torch.randint(0, num_classes, label.size()).to(device)
        encoder.eval()
        decoder.eval()
        classifier_opt.zero_grad()
        en_z = encoder(x)
        de_x = decoder(en_z)
        adv_x = de_x + x
        c, f_c = classifier(adv_x)
        cla_loss = c_loss(c, label.long())
        m_loss = margin_loss(f_c, en_z)
        c_t_loss = cla_loss + m_loss
        c_t_loss.backward(retain_graph=True)
        classifier_opt.step()
        encoder.train()
        decoder.train()
        classifier.eval()
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        en_z = encoder(x)
        de_x = decoder(en_z)
        c, c_f = classifier(de_x + x)
        h_loss = hinge_loss(de_x + x, x)
        re = decoder(encoder(de_x + x))
        re_loss = reconstruct_loss(re, x)
        adv_loss = a_loss(c, target.long())
        total_loss = adv_loss + 0.5 * h_loss + 0.5*re_loss
        total_loss.backward(retain_graph=True)
        decoder_opt.step()
        encoder_opt.zero_grad()
       
    
    p_a = []
    p_p = []
    ls = []
    classifier.eval()
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for x, label in test_loader:
            for l in label:
                ls.append(l.cpu().numpy())
            x = x.to(device, dtype=torch.float32)
            label = label.to(device)
            en_z = encoder(x)
            de_x = decoder(en_z)
            adv_x = x + de_x
            adv_pred, _ = classifier(adv_x)
            adv_pred = F.softmax(adv_pred, dim=1)
            adv_pred = torch.max(adv_pred, dim=1)[1].detach().cpu().numpy()
            pred,_ = classifier(x)
            pred = torch.max(F.softmax(pred, dim=1), dim=1)[1].detach().cpu().numpy()
            for i in adv_pred:
                p_a.append(i)
            for i in pred:
                p_p.append(i)
    acc_u = accuracy_score(label, p_p)
    acc_a = accuracy_score(label, p_a)
    f1_u = f1_score(label, p_p, average="macro")
    f1_a = f1_score(label, p_a, average="macro")
    print("acc_u:{} f1_u:{} acc_a:{} f1_a:{}".format(acc_u, f1_u, acc_a, f1_a))