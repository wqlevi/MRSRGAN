"""

Created on Mon Jul  5 16:46:23 2021

THIS SCRIPT IS DEDICATED FOR GENERAL DATASETS
- [x] Equivalent step sizes in all directions 
- [x] order of xyz index changed 
- [ ] All concatenation should be done in function scope
@author: qiwang
"""
import numpy as np
import nibabel as nb
import glob
import re
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path",type = str,default='/path/to/your/data/')
parser.add_argument("--subj",type=str,default = 'sub-0036')
parser.add_argument("--scale",type=int,default = 2,help='times of upscale: 1 -> 2x | 2 -> 4x')
opt = parser.parse_args()
args = parser.parse_args()
print(args)

# global variables
subj = opt.subj
path = opt.path
upscale_factor = opt.scale# 1 -> 2x | 2 -> 4x upscaling
step_size = 16 # for all xyz

# sort files by ascending order of : x>y>z
files = sorted(glob.glob(f'{path}/fake_{subj}*.nii'))
files=sorted(files,key=lambda x:(float(re.findall("(\d+)",x)[-3]),float(re.findall("(\d+)",x)[-2]),float(re.findall("(\d+)",x)[-1])))

    

def zcat(files,idx,idy):
    '''
     i, j = x_index , y_index
    '''
    # index 2 for y axis
    file_list = (img for img in files if int(re.findall("(\d+)",img)[-2]) == idy and int(re.findall("(\d+)",img)[-3]) == idx)
    for idz,x in enumerate(file_list,1):
        # start to manipulate
        print(idz,x)
        img = np.array(nb.load(x).dataobj)
        if idz==1:
            z_tmp = img
            continue
        else:
            z_tmp = np.concatenate((z_tmp,img),axis=2)
            del img
    return z_tmp

x_lim = int(re.findall("(\d+)",files[-1])[-3])
y_lim = int(re.findall("(\d+)",files[-1])[-2])
z_lim = int(re.findall("(\d+)",files[-1])[-1])

def ycat(files,idx):
    for idy in range(0,y_lim+step_size,step_size):
        if idy == 0:
            y_tmp = zcat(files,idx,idy)
        else:
            y_tmp = np.concatenate((y_tmp,zcat(files,idx,idy)),axis = 1)
    return y_tmp

def xcat(files):
    for idx in range(0,x_lim+step_size,step_size):
        if idx == 0:
            x_tmp = ycat(files,idx)
        else:
            x_tmp = np.concatenate((x_tmp,ycat(files,idx)),axis = 0)
    return x_tmp

EntireImage = xcat(files)
           
crop_size = 64*upscale_factor
steps = (step_size*upscale_factor,step_size*upscale_factor,step_size*upscale_factor)
(stp_x,stp_y,stp_z) = steps
(n_x,n_y,n_z) = EntireImage.shape
(rest_x,rest_y,rest_z) = [crop_size - x for x in steps]


mask = np.ones(EntireImage.shape,dtype=bool)
for k in range(0,n_z,crop_size):
    for j in range(0,n_y,crop_size):
        for i in range(0,n_x,crop_size):
            if i == 0 or j == 0 or k == 0:
                continue
            else:
                mask[:,:,k:k+rest_z] = False
                mask[:,j:j+rest_y,:] = False
                mask[i:i+(rest_x),:,:] = False
out_image = np.reshape(EntireImage[mask],(mask.sum(axis=0)[0][0],mask.sum(axis=1)[0][0],mask.sum(axis=2)[0][0]))

#%% IO
img_nb = nb.Nifti1Image(out_image, np.eye(4))
save_path = os.path.join(path,'../fake')
print(save_path)
os.makedirs(save_path,exist_ok=True)          
nb.save(img_nb,f'{save_path}/fake_{subj}.nii')
print("successed")

# let'em free...
del mask
del out_image
del EntireImage



