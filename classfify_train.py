import torch
from torch.nn import functional as F
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
import torchvision.transforms as tsf
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from datasets import PALMClassifyDataset

img_size = 224
batch_size = 16
num_workers = 4
num_classes = 3
channels = 3
epochs = 200
lr = 0.01*batch_size/256


train_paths, val_paths, train_labels, val_labels = train_test_split(paths, labels, random_state=0, stratify=labels)
train_dataset = PALMClassifyDataset(train_paths, train_labels, augmentation=True, img_size=img_size)
test_dataset = PALMClassifyDataset(val_paths, val_labels, augmentation=False, img_size=img_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = resnet18(pretrained=False, num_classes=num_classes, zero_init_residual=False)
optmizer = opt.SGD([{'params':model.parameters(), 'initia_lr':lr}], lr=lr, momentum=0.995, nesterov=True, weight_decay=5e-4)
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
    torch.save(model.state_dict(), "resnet18.pth")
    if best_acc < acc:
      torch.save(model.state_dict(), "resnet18_acc.pth")
      best_acc = acc
    if best_f1 < f1:
      torch.save(model.state_dict(), "resnet18_f1.pth")
      best_f1 = f1
    print("epoch:{} acc:{} f1:{}".format(epoch, acc, f1))