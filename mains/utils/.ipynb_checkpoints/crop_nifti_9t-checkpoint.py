#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:05:52 2021
 - [x] From this verion on, all xyz will use same step size
 - [x] manually pad axis with 0 to fullfill step_size(constant now = 16)

USAGE:
    Args:
    0: the program itself
    1: data path
    2: subject prefix for wildcard search
    3: [optional] flag for padding
@author: qiwang
"""
import nibabel as nb
import numpy as np
import os
import sys
import glob

crop_size = 64
step_size_x = 16
step_size_y = 16
step_size_z = 16
print(f"*****image will be cropped to size of {crop_size}*****")
imgs = []

data_path = sys.argv[1] # data orig path
subj = sys.argv[2] # subj
pad_flag = False

# see if padding flag will be set
try:
    if (sys.argv[3]):
        pad_flag = True
except IndexError:
    pass



#%% Functions
def crop_img3d(img_path,path_save,crop_size,*step_size):
    '''
    Args: 
        0: image path for loading
        1: image saving folder
        2: crop_size
        *3: step_size for xyz
    '''
    print(f'****image\t {img_path.split("/")[-1]} is being cropped****')
    step_size_x,step_size_y,step_size_z = step_size if (len(step_size) == 3) else quit()
    img = nb.load(img_path)
    print(img.shape,'\t',img.get_filename())
    for k in range(0,(img.shape[2]-crop_size)+step_size_x,step_size_x):
           for j in range(0,(img.shape[1]-crop_size)+step_size_y,step_size_y):
               #if j>236:
               #    continue
               for i in range(0,(img.shape[0]-crop_size)+step_size_z,step_size_z):
                   img_c = img.slicer[i:i+crop_size,j:j+crop_size,k:k+crop_size]
                   img_c = img_c.to_filename(f'{path_save}/{img.get_filename().split("/")[-1].split(".")[0]}_{i}_{j}_{k}.nii') # not using .gz for acceleration
                   imgs.append(img_c)




def pad_nii(data_path):
    '''
    Here padding was applied to all images of same dataset(AHEAD 7T example here), but padding size should be adjusted per dataset
    Note: padding on specific dimension to make residual of being devided by 16 to 0.
    '''
    img_nii = nb.load(data_path)
    img = np.array(img_nii.dataobj)
    img = np.pad(img,((0,0),(2,2),(0,0)),'constant',constant_values=(0,)) # pad zero at both side of array
    img_save = nb.Nifti1Image(img, img_nii.affine, img_nii.header)
    nb.save(img_save,img_nii.get_filename())
#%% Execute
save_path = os.path.join(data_path,'crops')
os.makedirs(save_path,exist_ok = True)
if pad_flag:
    for f in glob.glob(data_path+f'/{subj}*'):
        pad_nii(f)
    
[crop_img3d(data_p,save_path,crop_size,step_size_x,step_size_y,step_size_z) for data_p in glob.glob(data_path+f'/{subj}*')]


