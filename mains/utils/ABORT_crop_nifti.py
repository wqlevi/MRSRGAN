#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:05:52 2021

@author: qiwang
"""
import nibabel as nb
import numpy as np
import os
import time
import glob
import sys
# img = nb.load('/home/qiwang/Downloads/HCP_data/3T_v1_3D/LS2001.nii')
crop_size = 64
imgs = []
data_path = sys.argv[1]
path_save = sys.argv[2]
def crop_img3d(img,path_save,crop_size):   
    img = nb.load(img)
    for k in range(0,img.shape[2],16):
           for j in range(0,img.shape[1],15):
               for i in range(0,img.shape[0],16):
                   img_c = img.slicer[i:i+crop_size,j:j+crop_size,k:k+crop_size]
                   img_c = img_c.to_filename(f'{path_save}/{img.get_filename().split("/")[-1].split(".")[0]}_{i}_{j}_{k}.nii')
                   imgs.append(img_c)


files = sorted(glob.glob(data_path+'/*.nii'))
os.makedirs(path_save,exist_ok = True)
[crop_img3d(fi,path_save,crop_size) for fi in files] 
