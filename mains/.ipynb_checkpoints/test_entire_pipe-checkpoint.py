#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:38:41 2022

@author: qiwang
"""
import sys
sys.path.append('./utlis')

import nibabel as nb
import numpy as np
import torch

import os
from tqdm import tqdm

import glob
import re

from ESRGAN_3d_dataset import *
from model_base import img2lr, load_model
from utils import *


#%% 
# GLOBAL VARs
#CROP_SIZE = 64
#OVERLAP=16

# inline arguments
upscale_factor = 2
data_path = '/ptmp/wangqi/transfer_folder/100610.nii.gz'
nii = nb.load(data_path)
hdr = nii.header

# function handle to return number of crops along each axis
del nii
#%% cropping images
def crop_img3d_2list(img_path,crop_size,*step_size,path_save=''):
    '''
    Args: 
        0: image path for loading
        1: image saving folder
        2: crop_size
        *3: step_size for xyz
    Output: 
        list | coordinate order |
             |    x > y > z     |
        new image shape : tuple
    '''
    print(f'****image\t {img_path.split("/")[-1]} is being cropped****')
    step_size_x,step_size_y,step_size_z = step_size if (len(step_size) == 3) else quit("specify x,y,z step size plz!")
    img = nb.load(img_path)
    
    # pad image
    pad_num = [int((np.ceil(x/64)*64-x)/2) for x in img.get_fdata().shape] # pad all dim to be divided by 64
    pad_img = np.pad(img.get_fdata(),((pad_num[0],pad_num[0]),(pad_num[1],pad_num[1]),(pad_num[2],pad_num[2])),'constant',constant_values=(0,))
    pad_nii = nb.Nifti1Image(pad_img,np.eye(4))
    
    imgs = []
    print(pad_img.shape,'\t',img.get_filename())
    for k in range(0,(pad_img.shape[2]-crop_size)+step_size_x,step_size_x):
        for j in range(0,(pad_img.shape[1]-crop_size)+step_size_y,step_size_y):
            for i in range(0,(pad_img.shape[0]-crop_size)+step_size_z,step_size_z):                
                img_c = pad_nii.slicer[i:i+crop_size,j:j+crop_size,k:k+crop_size]
                #img_c = img_c.to_filename(f'{path_save}/{img.get_filename().split("/")[-1].split(".")[0]}_{i}_{j}_{k}.nii') # not using .gz for acceleration
                imgs.append(img_c.get_fdata())
    return imgs,pad_img.shape


def assemble_img_X64(crop_list:list,new_shape,scale:int=1):
    '''
    input : 
        0 : crop list of numpy arrays
        1 : shape tuple of padded images
    output:
        0 : newly assembled image  
    '''
    X_NUM,Y_NUM,Z_NUM = [int(np.ceil(x/64)) for x in new_shape] # num of non-overlapping crops along each dim
    CROP_SIZE = 64*scale
    tmp_arr = np.ones((X_NUM*CROP_SIZE,Y_NUM*CROP_SIZE,Z_NUM*CROP_SIZE)).astype(dtype = np.float32) # allocate output image w/ appending size
    i = 0 # iterator
    for idz in range(0,Z_NUM):
        for idy in range(0,Y_NUM):
            for idx in range(0,X_NUM):                
                tmp_arr[CROP_SIZE*idx:CROP_SIZE*(idx+1), CROP_SIZE*idy:CROP_SIZE*(idy+1), CROP_SIZE*idz:CROP_SIZE*(idz+1)] = crop_list[i]
                i += 1
    return tmp_arr     
def produce(model,crop_img,scale:int,device,hr_shape=64):
    '''
    input:
        0:model
        1:crop_img (ndarray)
        2:upscale factor
        3:device to run inferencing
        4:HR shape
    '''
    model.eval()
    with torch.no_grad():
        _,data_tensor_lr = img2lr(crop_img,device,hr_shape,scale)
        res_tensor = model(data_tensor_lr.to(device))
    return res_tensor.detach().squeeze().cpu().numpy()

#%% testing
#######
# Testing SR
#######

device  = torch.device('cuda:0')
hr_shape = 64 # input orig crop size
model_name = 'c11_generator_102_1359'
# start to load model
model = Generator().to(device)
model.load_state_dict(torch.load(f'../saved_models/{model_name}.pth'),strict = False)
model.eval()

subj_list = [os.path.basename(x).split('.')[0].split('_')[-1] for x in glob.glob("/ptmp/wangqi/transfer_folder/anatomy/*")]
# reverse list
for subj in tqdm(subj_list):
    crop_list,newshape = crop_img3d_2list(f"/ptmp/wangqi/transfer_folder/anatomy/{subj}.nii.gz",64,64,64,64)
    tmp = [produce(model,x,2,device) for x in crop_list]
    new_img = assemble_img_X64(tmp,newshape,2)[60:-60,9:-9,60:-60]
    new_nii = nb.Nifti1Image(new_img,np.eye(4),hdr)
    nb.save(new_nii,f"/ptmp/wangqi/HCP_v3_fake/fake_c11_4fold_102ep_skull_{subj}.nii.gz")
