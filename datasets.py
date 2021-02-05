#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
   @Author: Zhongxi Qiu
   @Filename:datasets.py
   @time:2020/11/04 09:03:03
'''


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
from albumentations import Compose,RandomBrightnessContrast,RandomGamma, HorizontalFlip, VerticalFlip, ChannelShuffle, Resize, PadIfNeeded,Normalize
from torch.utils.data import Dataset
from skimage.filters import sobel
import numpy as np
import os
import glob
import cv2

def get_paths(img_dir, mask_dir, img_suffix, mask_suffix):
    '''
        Get the paths of images and masks.
        @img_dir: str,the path of directory that contain image
        @mask_dir: str, the path of direcotry that contain mask file
        @img_suffix: str, the suffix for the image
        @mask_suffix: str, the suffix for mask file
        return: (list, list), the tuple of list 
    '''
    assert os.path.exists(img_dir), "Cannot find the directory {}".format(img_dir)
    assert os.path.exists(mask_dir), "Cannot find the directory {}".format(mask_dir)
    img_paths = glob.glob(os.path.join(img_dir, "*{}".format(img_suffix)))
    mask_paths = []
    for path in img_paths:
        filename = os.path.basename(path)[:-len(img_suffix)]
        mask_path = os.path.join(mask_dir, filename + mask_suffix)
        mask_paths.append(mask_path)
    return img_paths, mask_paths

class PALMDataset(Dataset):
    '''
        The dataset object to read the data in PALM dataset.
    '''
    def __init__(self, img_paths, mask_paths, img_size=512, augumentation=False):
        '''
            Initilize the ocject, set the paths and the size of image.
            img_paths: list, the paths of the images
            mask_paths: list, the paths of the masks
            img_size: int or tuple, the size of  output image
            augumentation: boolean, whether use data augumentation
        '''
        super(PALMDataset, self).__init__()
        assert len(img_paths) == len(mask_paths), "The length of list of image paths must equal to the length of the list of masks, but got {}/{}" .format(len(img_paths), len(mask_paths))
        assert len(img_paths) > 0, "The number of samples in dataset are except to greater than 0, but got {}".format(len(img_paths))
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augumentation = augumentation
        self.length = len(img_paths)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        '''
            The method to get one data from dataset.
            index: int, the index of the sample in dataset.
        '''
        #get data from dataset
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #modify the mask to satisfied our requirments for the mask
        if not os.path.exists(mask_path):
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        else:
            mask = cv2.imread(mask_path, 0)
            mask[mask > 1] = 2
            mask[mask < 1] = 1
            mask[mask > 1] = 0


        if isinstance(self.img_size, tuple):
            height = self.img_size[1]
            width  = self.img_size[0]
        elif isinstance(self.img_size, int):
            width = height = self.img_size
        
        if self.augumentation:
            #augmentation methods
            task = Compose([
                RandomBrightnessContrast(),
                RandomGamma(),
                HorizontalFlip(),
                VerticalFlip(),
                ChannelShuffle(),
                PadIfNeeded(height, width),
            ])
            #augmentation
            augument_data = task(image=img, mask=mask)
            img = augument_data["image"]
            mask = augument_data["mask"]
        resize = Compose([
             Resize(height=height, width=width, always_apply=True),
             Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True)
        ])
        #resize data
        resize_data = resize(image=img, mask=mask)
        img, mask = resize_data["image"], resize_data["mask"]
        if img.ndim > 2:
            img = np.transpose(img, axes=[2, 0, 1])
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(mask.astype(np.float32))

class PALMClassifyDataset(Dataset):
    def __init__(self, image_paths, labels, augmentation=False, img_size=224,
                 mean=(0.5, 0.5, 0,5), std=(0.5, 0.5, 0.5)):
        '''
            Implementation of the dataset for palm dataset classifier task.
            args:
                image_paths: list, the paths of the images
                labels: list, the labels for the image
                augmentation: bool, whether use data augmentation
                img_size: int or tuple, the size of output image
                mean (Union[list, tuple]), the mean of the normalization
                std (Union[list, tuple]), the std of the normalization
        '''
        super(PALMClassifyDataset, self).__init__()
        assert len(image_paths) == len(labels), "The lengh of image_paths and labels must be equal, but got {}/{}".format(len(image_paths), len(labels))
        self.image_paths = image_paths
        self.labels = labels
        self.length = len(image_paths)
        self.augmentation = augmentation
        self.img_size = img_size
        self.mean = mean
        self.std = std
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        '''
            Magic method for get one sample from the dataset.
            args:
                index: int, the index of the sample
            return:
                the spcify sample in the dataset
        '''
        #get sample from the dataset
        img_path = self.image_paths[index]
        label = np.array(self.labels[index], dtype=np.float32)
        assert os.path.exists(img_path), "Cannot found the image file {}".format(img_path)
        #read image 
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isinstance(self.img_size, tuple):
            height = self.img_size[1]
            width  = self.img_size[0]
        elif isinstance(self.img_size, int):
            width = height = self.img_size
        #data augmentation
        if self.augmentation:
            #data augmentation metiod
            task = Compose([
                RandomBrightnessContrast(),
                RandomGamma(),
                HorizontalFlip(),
                VerticalFlip(),
                ChannelShuffle(),
                PadIfNeeded(height, width),
            ])
            aug_data = task(image=img)
            img = aug_data["image"]
        #resize the image
        resize = Compose([
            Resize(height=height, width=width, always_apply=True),
            Normalize(mean=self.mean, std=self.std,always_apply=True)
        ])
        resize_data = resize(image=img)
        img = resize_data["image"]
        if img.ndim > 2:
            img = np.transpose(img, axes=[2, 0, 1])
        else:
            img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img), torch.from_numpy(label)