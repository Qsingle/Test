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
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from matplotlib import pyplot as plt
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
%pylab inline

from modules.decoder import Decoder
from modules.encoder import Encoder
from datasets import PALMClassifyDataset
# from modules.resnet import ResNet

img_size = 224
batch_size = 2
num_workers = 4
num_classes = 2
channels = 3
epochs = 64
alpha = 0.005
init_lr = 0.01 * batch_size / 256

# train_transform = tsf.Compose([
#     tsf.Resize((img_size, img_size)),
#     tsf.RandomHorizontalFlip(),
#     tsf.RandomVerticalFlip(),
#     tsf.ToTensor(),
#     tsf.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])

# val_transform =  tsf.Compose([
#     tsf.Resize((img_size, img_size)),
#     tsf.ToTensor(),
#     tsf.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])

# train_dataset = CIFAR10("./data", train=True, transform=train_transform, download=True)
# test_dataset = CIFAR10("./data", train=False, transform=val_transform, download=False)
#train_paths, val_paths, train_labels, val_labels = train_test_split(paths, labels, random_state=0, stratify=labels)
train_paths, train_labels = get_paths("./cat_dog/train", img_suffix=".jpg")
val_paths, val_labels = get_paths("./cat_dog/val", img_suffix='.jpg')
train_dataset = PALMClassifyDataset(train_paths, train_labels, augmentation=True, img_size=img_size)
test_dataset = PALMClassifyDataset(val_paths, val_labels, augmentation=False, img_size=img_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

encoder = Encoder(in_ch=channels, out_ch=2048)
decoder = Decoder(in_ch=2048, out_ch=channels)

#classifier = ResNet(channels, n_layers=50, num_classes=num_classes)
classifier = resnet18(pretrained=False, num_classes=num_classes, zero_init_residual=False)
# classifier.load_state_dict(torch.load("./best_classifier.pth", map_location="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)
classifier.to(device)

encoder_opt = opt.Adam(encoder.parameters(), lr=init_lr, weight_decay=5e-4)
decoder_opt = opt.Adam(decoder.parameters(), lr=init_lr, weight_decay=5e-4)
classifier_opt = opt.Adam(classifier.parameters(), lr=init_lr, weight_decay=5e-4)

c_loss = nn.CrossEntropyLoss()
a_loss = nn.CrossEntropyLoss()
hinge_loss = nn.MSELoss()
margin_loss = nn.MSELoss()
reconstruct_loss = nn.L1Loss()
best_acc = 0
best_f1 = 0
low_acc = np.inf
low_f1 = np.inf

for epoch in range(epochs):
    for x, label in train_loader:
        x = x.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long)
        target = torch.randint(0, num_classes, label.size()).to(device)        
        encoder.train()
        decoder.train()
        classifier.eval()
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        en_z = encoder(x)
        de_x = decoder(en_z)
        adv_x = x + de_x*alpha
        c = classifier(adv_x)
        h_loss = hinge_loss(adv_x, x)
        re = decoder(encoder(adv_x))
        re_loss = reconstruct_loss(re, x)
        adv_loss = a_loss(c, target.long())
        total_loss = 4*adv_loss +  h_loss + re_loss
        total_loss.backward(retain_graph=True)
        decoder_opt.step()
        encoder_opt.step()
        classifier.train()
        encoder.eval()
        decoder.eval()
        classifier_opt.zero_grad()
        c = classifier(x)
        cla_u_loss = c_loss(c, label)
        cla_u_loss.backward(retain_graph=True)
        classifier_opt.step()
        en_z = encoder(x)
        de_x = decoder(en_z)
        classifier_opt.zero_grad()
        adv_x = x + alpha*de_x
        c = classifier(adv_x)
        cla_loss = c_loss(c, label.long())
        c_t_loss = cla_loss
        c_t_loss.backward()
        classifier_opt.step()
        
        
    p_a = []
    p_p = []
    ls = []
    torch.save(classifier.state_dict(),"./classifier.pth")
    torch.save(encoder.state_dict(),"./encoder.pth")
    torch.save(decoder.state_dict(),"./decoder.pth")
    classifier.eval()
    encoder.eval()
    decoder.eval()
    show = True
    with torch.no_grad():
        for i, (x, label) in enumerate(test_loader):
            for l in label:
                ls.append(l.cpu().numpy())
            x = x.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)
            en_z = encoder(x)
            de_x = decoder(en_z)
            adv_x = x + de_x*alpha
            adv_pred = classifier(adv_x)
            re = decoder(encoder(adv_x))
            adv_pred = F.softmax(adv_pred, dim=1)
            adv_pred = torch.max(adv_pred, dim=1)[1].detach().cpu().numpy()
            if show:
              img = x[0].detach().cpu().numpy()
              img = np.transpose(img, axes=[1, 2, 0])
              plt.figure(dpi=224)
              plt.subplot(2, 2, 1)
              plt.title("{}".format(label[0].detach().cpu().numpy()))
              plt.imshow(img)
              plt.xticks([])
              plt.yticks([])
              img = adv_x[0].detach().cpu().numpy()
              img = np.transpose(img, axes=[1, 2, 0])
              plt.subplot(2, 2, 2)
              plt.title("{}".format(adv_pred[0]))
              plt.imshow(img)
              plt.xticks([])
              plt.yticks([])
              #plt.show()
              img = re[0].detach().cpu().numpy()
              img = np.transpose(img, axes=[1, 2, 0])
              plt.subplot(2, 2, 3)
              plt.title("{}".format("re"))
              plt.imshow(img)
              plt.xticks([])
              plt.yticks([])
              plt.show()
              show = False
            pred = classifier(x)
            pred = torch.max(F.softmax(pred, dim=1), dim=1)[1].detach().cpu().numpy()
            for i in adv_pred:
                p_a.append(i)
            for i in pred:
                p_p.append(i)
    acc_u = accuracy_score(ls, p_p)
    acc_a = accuracy_score(ls, p_a)
    f1_u = f1_score(ls, p_p, average="macro")
    f1_a = f1_score(ls, p_a, average="macro")
    if low_acc > acc_a or low_f1 > f1_a:
      torch.save(encoder.state_dict(), "best_encoder.pth")
      torch.save(decoder.state_dict(), "best_decoder.pth")
      torch.save(classifier.state_dict(), "f_classifier.pth")
      low_acc = acc_a
    if best_acc < acc_a or best_f1 < f1_a:
      best_acc = acc_a
      torch.save(encoder.state_dict(), "f_encoder.pth")
      torch.save(decoder.state_dict(),"f_decoder.pth")
      torch.save(classifier.state_dict(), "best_classifier.pth")
    print("acc_u:{} f1_u:{} acc_a:{} f1_a:{}".format(acc_u, f1_u, acc_a, f1_a))