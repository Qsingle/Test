#-*- coding:utf-8 -*-
#!/etc/env python
'''
   @Author:Zhongxi Qiu
   @File: run_attack.py
   @Time: 2021-01-07 16:54:47
   @Version:1.0
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as tsf
import numpy as np
import json
import cv2

from display_utils import denormalize
from modules.encoder import Encoder
from modules.decoder import Decoder

f = open("config.json")
config = json.load(f)
f.close()
img_size = config["img_size"]
batch_size = config["batch_size"]
num_workers = config["num_workers"]
means = config["means"]
std = config["stds"]
channels = config["channel"]
alpha = config["alpha"]

if isinstance(config["out_size"], int):
    out_size = (config["out_size"]) * 2
elif isinstance(config["out_size"], list):
    out_size = (config["out_size"][0], config["out_size"][1])
else:
    raise ValueError("Unknown size")

val_transform =  tsf.Compose([
    tsf.Resize((img_size, img_size)),
    tsf.ToTensor(),
    tsf.Normalize(mean=means, std=std)
])
val_dataset = CIFAR10(root="./data", train=False, transform=val_transform, download=True)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

encoder = Encoder(channels, out_ch=2048)
decoder = Decoder(2048, channels)

encoder.load_state_dict(torch.load(config["encoder"], map_location="cpu"))
decoder.load_state_dict(torch.load(config["decoder"], map_location="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

encoder.eval()
decoder.eval()
x_adv = []
with torch.no_grad():
    for x, label in test_loader:
        x = x.to(device, dtype=torch.float32)
        en = encoder(x)
        de = decoder(en)
        de = x + alpha*de
        de = denormalize(de[0], means=means, stds=std)
        de = de.detach().cpu().numpy()
        de = np.transpose(de, axes=[1, 2, 0])
        de = cv2.resize(de, (32, 32))
        de = de * 255
        de = np.clip(de, 0, 255)
        de = np.expand_dims(de, axis=0)
        x_adv.append(de)

x_out = np.concatenate(x_adv, axis=0)
save_path = config["save_path"]
np.save(save_path, x_out)
print("result are saved to {}".format(save_path))