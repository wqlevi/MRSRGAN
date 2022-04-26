import torch
import numpy as np
import nibabel as nb
from model.toy_model_3d import Generator
import argparse
import glob
import os



def img2tensor(filename,device,hr_shape,scale = 1):
    img = np.array(nb.load(filename).dataobj)
    img = np.expand_dims(img,axis=0)
    img = np.expand_dims(img,axis=0)
    tensor = torch.from_numpy(img).float()
    tensor_lr = torch.nn.functional.interpolate(tensor,int(scale*hr_shape/2)).to(device)
    return tensor,tensor_lr

def load_model(model_name,device):
    model = Generator().to(device)
    model.load_state_dict(torch.load(f'../saved_models/{model_name}.pth'),strict = False)
    return model
'''
def img2lr(filename,device,hr_shape,scale = 1):
    img = nb.load(filename).get_fdata()
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    tensor_lr = torch.nn.functional.interpolate(tensor,int(hr_shape*scale/2)).to(device)
    return tensor,tensor_lr
'''

def img2lr(img,device,hr_shape,scale = 1):
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    tensor_lr = torch.nn.functional.interpolate(tensor,int(hr_shape*scale/2)).to(device)
    return tensor,tensor_lr

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad=False
        
def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad=True