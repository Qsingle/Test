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
import pandas as pd
from albumentations import Compose, Resize, Normalize
import os
import tqdm

from display_utils import denormalize
from modules.encoder import Encoder
from modules.decoder import Decoder


f = open("config.json")
config = json.load(f)
f.close()
img_size = config["img_size"]
means = config["means"]
std = config["stds"]
channels = config["channel"]
alpha = config["alpha"]
csv_path = config["csv_path"]
img_dir = config["image_dir"]
output_dir = config["output_dir"]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
data = pd.read_csv(csv_path)
paths = data["ImageId"].values
paths = [os.path.join(img_dir, p) for p in paths]
labels = data["TrueLabel"].values

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
    bar = tqdm.tqdm(paths)
    for path in bar:
        filename = os.path.basename(path)
        bar.set_description(f"processing:{filename}")
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        norm = Compose([
            Resize(img_size, img_size, always_apply=True),
            Normalize(mean=means, std=std, always_apply=True)
        ])
        norm_data = norm(image=image)
        image = norm_data["image"]
        if image.ndim > 2:
            image = np.transpose(image, axes=[2, 0, 1])
        else:
            image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.to(device, dtype=torch.float32)
        en = encoder(image)
        de = decoder(en)
        adv = image + alpha*de
        #print(adv.shape)
        adv = adv.squeeze(0)
        adv = denormalize(adv, means, std)
        adv = adv * 255
        adv = torch.clamp(adv, 0, 255)
        adv = adv.detach().cpu().numpy()
        adv = np.transpose(adv, axes=[1, 2, 0])
        adv = cv2.cvtColor(adv, cv2.COLOR_RGB2BGR)
        adv = cv2.resize(adv, (w, h))
        adv = cv2.imwrite("./output/{}".format(filename), adv)