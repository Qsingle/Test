#-*- coding:utf-8 -*-
#!/etc/env python
'''
   @Author:Zhongxi Qiu
   @File: attack_check.py
   @Time: 2020-12-11 15:17:33
   @Version:1.0
'''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from torchvision.models import resnet18
from foolbox import  PyTorchModel,accuracy
from foolbox.criteria import TargetedMisclassification
import foolbox.attacks as fa
import numpy as np
from foolbox.models import Model
import eagerpy as ep
import warnings
import os
import glob
import cv2
from datasets import PALMClassifyDataset
from sklearn.metrics import accuracy_score

class_map = {
    "cat":0,
    "dog":1
}
def get_datas(data_dir, img_suffix=".jpg"):
    image_paths = glob.glob(os.path.join(data_dir,"*{}".format(img_suffix)))
    labels = []
    for path in image_paths:
        filename = os.path.basename(path)
        label = filename.split("_")[0]
        label = class_map[label]
        labels.append(label)
    return image_paths, labels
        
num_classes = 2
batchsize = 16
data_dir = "../cat_dog/val"
classifier_path = "../ckpts/best_classifier.pth"
image_suffix = ".jpg"

classfier = resnet18(pretrained=False, num_classes=num_classes)
classfier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
classfier.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracys = []
fmodel = PyTorchModel(classfier, bounds=(0, 1), device=device)
image_paths, labels = get_datas(data_dir, image_suffix)

eval_dataset = PALMClassifyDataset(image_paths, labels)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batchsize, num_workers=0, shuffle=False)
preds = []
for x, _ in eval_loader:
    x = x.to(device)
    pred = fmodel(x).argmax(axis=-1).tolist()
    preds += pred

acc_clean = accuracy_score(labels, preds)
print(f"clean accuracy: {acc_clean * 100:.1f} %")
attacks = [
    fa.FGSM(),
    fa.LinfPGD(),
    fa.LinfBasicIterativeAttack(),
    fa.LinfAdditiveUniformNoiseAttack(),
    fa.LinfDeepFoolAttack(),
]
epsilons = [
    0.0,
    0.0005,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.005,
    0.01,
    0.02,
    0.03,
    0.1,
    0.3,
    0.5,
    1.0
]
print("epsilons")
print(epsilons)
print("")

attack_success = np.zeros((len(attacks), len(epsilons), len(labels)), dtype=np.bool)
for i, attack in enumerate(attacks):
    successes = None
    for images, las in eval_loader:
        images = images.to(device)
        las = las.to(device, dtype=torch.long)
        _, _, success = attack(fmodel, images, las, epsilons=epsilons)
        assert success.shape == (len(epsilons), len(images))
        success_ = success.detach().cpu().numpy()
        if successes is None:
            successes = success_.copy()
        else:
            successes = np.concatenate((successes, success_), axis=-1)
        assert success_.dtype == np.bool
    
    attack_success[i] = successes
    print(attack)
    print("  ", 1.0 - successes.mean(axis=-1).round(4))

robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
print("")
print("-"*79)
print("")
print("worst case (best attack per-sample)")
print("  ", robust_accuracy.round(4))
print("")

print("robust accuracy for perturbations with")
for eps, acc in zip(epsilons, robust_accuracy):
    print(f"   Linf norm <= {eps:<6}:{acc.item()*100:5.2f} %")