import torch
from torch.nn import functional as F
from torchvision.models import resnet18
import timm
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
import torchvision.transforms as tsf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

from datasets import PALMClassifyDataset
from modules.resnet import ResNet

img_size = 224
batch_size = 16
num_workers = 4
num_classes = 3
channels = 3
epochs = 200
lr = 0.01*batch_size/256
test_size = 0.2
image_dir = "/data/adversarial_tianchi/images/"
csv_path = "/data/adversarial_tianchi/{}.csv"
train_data = pd.read_csv(csv_path.format("train"))
train_paths = train_data["ImageId"].values
train_paths = [os.path.join(image_dir, p) for p in train_paths]
train_labels = train_data["TrueLabel"].values
val_data = pd.read_csv(csv_path.format("test"))
val_paths = val_data["ImageId"].values
val_paths = [os.path.join(image_dir, p) for p in val_paths]
val_labels = val_data["TrueLabel"].values
#train_paths, val_paths, train_labels, val_labels = train_test_split(paths, labels, random_state=0, stratify=labels, test_size=test_size)
train_dataset = PALMClassifyDataset(train_paths, train_labels, augmentation=True, img_size=img_size)
test_dataset = PALMClassifyDataset(val_paths, val_labels, augmentation=False, img_size=img_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = ResNet(in_ch=channels, n_layers=50, num_classes=1000, light_head=True, avd=True, avd_first=False, avg_layer=True, avg_down=True)
optmizer = opt.SGD([{'params':model.parameters(), 'initia_lr':lr}], lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
lr_she = torch.optim.lr_scheduler.CosineAnnealingLR(optmizer, T_max=epochs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

loss_func = nn.CrossEntropyLoss()

best_acc = 0
best_f1 = 0
for epoch in range(epochs):
  model.train()
  for x, label in train_loader:
    x = x.to(device, torch.float32)
    label = label.to(device, torch.long)
    optmizer.zero_grad()
    pred = model(x)
    loss = loss_func(pred, label)
    loss.backward()
    optmizer.step()
  lr_she.step()
  model.eval()
  ls = []
  preds = []
  with torch.no_grad():
    for x, label in test_loader:
      for l in label:
        ls.append(l.cpu().numpy())
      x = x.to(device, dtype=torch.float)
      pred = model(x)
      pred = F.softmax(pred, dim=1)
      pred = torch.max(pred, dim=1)[1].detach().cpu().numpy()
      for i in pred:
        preds.append(i)
    
    acc = accuracy_score(ls, preds)
    f1 = f1_score(ls, preds, average="macro")
    torch.save(model.state_dict(), "resnest50.pth")
    if best_acc < acc:
      torch.save(model.state_dict(), "resnest50_acc.pth")
      best_acc = acc
    if best_f1 < f1:
      torch.save(model.state_dict(), "resnest50_f1.pth")
      best_f1 = f1
    print("epoch:{} acc:{} f1:{}".format(epoch, acc, f1))