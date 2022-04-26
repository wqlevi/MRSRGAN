#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:07:35 2021

- [x] 3d images
- [x] 3d resize and padding
- [x] paired hr,lr images with padding, interpolation
- [x] feature wise normalized
@author: qiwang
"""

import glob
import numpy as np
import nibabel as nb
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
#mean = np.array([0.5,])
#std = np.array([0.5,])
mean = np.array([355.2167])
std = np.array([359.8541])
def denormalize(tensors,channels=1):
    """ Denormalizes image tensors using mean and std """
    for c in range(channels):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_length, hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [   
                # transforms.Grayscale(num_output_channels=1),
                
                transforms.ToTensor(),
                # transforms.Normalize(mean, std)
            ]
        )
        self.hr_transform = transforms.Compose(
            [   
                transforms.ToTensor(),
                # transforms.intepolate(hr_height)
                # write interpolation for 3D images in Lambda
                # transforms.Normalize(mean, std),
            ]
        )
        
        # lists all files in root folder
        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = np.array(nb.load(self.files[index % len(self.files)]).dataobj)
        
        # padding to resize 128*128*128
        img = np.expand_dims(np.array(img), axis = 0)
        img = torch.Tensor(img).unsqueeze(0)
        img_pad = torch.nn.functional.pad(img,(7,8,0,0,7,8))
        
        img_hr = torch.nn.functional.interpolate(img_pad,128).squeeze(1)
        img_lr = torch.nn.functional.interpolate(img_pad,64).squeeze(1)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)
    

class CropDataset(Dataset):
    def __init__(self,root):
        # hr_length, hr_height, hr_width = hr_shape
        self.norm = transforms.Lambda(lambda x : (x - mean)/std)
        self.files = sorted(glob.glob(root + "/*.nii*"))
    def __getitem__(self, index):
        img = np.array(nb.load(self.files[index % len(self.files)]).dataobj)
        
        img = np.expand_dims(np.array(img), axis = 0)
        img = torch.Tensor(img).unsqueeze(0)
        #img = self.norm(img) # for per item normalize
        img_hr = img.squeeze(0)
        img_lr = torch.nn.functional.interpolate(img,32).squeeze(0)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)

class Val_Dataset(Dataset):
    def __init__(self,root):
        # hr_length, hr_height, hr_width = hr_shape
        self.norm = transforms.Lambda(lambda x : (x - mean)/std)
        self.files = sorted(glob.glob(root + "/*.nii*"))
    def __getitem__(self, index):
        img = nb.load(self.files[index % len(self.files)])
        return torch.Tensor(img.get_fdata())

    def __len__(self):
        return len(self.files)
