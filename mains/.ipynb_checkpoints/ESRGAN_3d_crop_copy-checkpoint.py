#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:54:00 2021

- [x] windowing patchs of 3D images, and trained patched model 

@author: qiwang
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch import from_numpy
import os
from torch.autograd import Variable
from torchvision.utils import save_image
from scipy.io import savemat
import glob
import re
import sys
sys.path.append('utils/')

from ESRGAN_3d_dataset import *
from ESRGAN_3d_model import *
from utils import *

import glob
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path',type=str,help = 'path of training data')
parser.add_argument('--checkpoint',type=int,default=0)
parser.add_argument('--model',type=str,help = 'name of current model')
opt = parser.parse_args()

# --- arguments ---
datapath = opt.path
checkpoint = opt.checkpoint
model_name = opt.model
device = torch.device('cuda:0')
# ---- model config ---

num_epoch = 25
data = DataLoader(CropDataset(datapath),batch_size = 2,drop_last=True,num_workers=2,shuffle=True)

Tensor = torch.cuda.FloatTensor

criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
#criterion_content = torch.nn.L1Loss().to(device)
#criterion_pixel = torch.nn.L1Loss().to(device)
# changed to L2 loss, 13 Jan 2022
criterion_content = torch.nn.MSELoss().to(device)
criterion_pixel = torch.nn.MSELoss().to(device)
Dnet = Discriminator().to(device)
# Gnet = Generator().to(device)
Gnet = Generator()
FE = FeatureExtractor().to(device)
# --- checkpoint ---
if not checkpoint == 0:
    #checkpoint_models_G = glob.glob('saved_models/*generator*')
    #checkpoint_models_D = glob.glob('saved_models/*discriminator*')
    #
    #checkpoint_model_G = sorted(checkpoint_models_G,key=lambda x: float(re.findall("(\d+)",x)[2]))[0]
    #checkpoint_model_D = sorted(checkpoint_models_D,key=lambda x: float(re.findall("(\d+)",x)[2]))[0]
    checkpoint_model_G = f'saved_models/{model_name}_generator_{checkpoint}_{len(data)-1}.pth'
    checkpoint_model_D = f'saved_models/{model_name}_discriminator_{checkpoint}_{len(data)-1}.pth'
    checkpoint_model_FE = f'saved_models/{model_name}_FE_{checkpoint}_{len(data)-1}.pth'
    Dnet.load_state_dict(torch.load(f'{checkpoint_model_G}'),strict=False)
    Gnet.load_state_dict(torch.load(f'{checkpoint_model_D}'),strict=False)
    FE.load_state_dict(torch.load(f'{checkpoint_model_FE}'),strict=False)

optimizer_G = torch.optim.Adam(Gnet.parameters(), lr=.0002, betas=(0.9, .999))
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=.0002, betas=(0.9, .999))

psnr = psnr()
#%% training function
def train_loop(dataloader,model,loss_function,optimizer,device,epoch,num_epoch,checkpoint,model_name):
    '''
    model = list
    optimizer = list
    '''
    epoch = epoch + checkpoint
    optimizer_D,optimizer_G = optimizer# revised 3rd Aug 2021
    Dnet,Gnet,FE = model
    criterion_pixel,criterion_GAN,criterion_content=loss_function
    D_loss = []
    G_loss = []
    pixel_loss = []
    content_loss = []
    GAN_loss = []
    psnr_value = []
    for i,imgs in enumerate(dataloader):
        
        imgs_hr = imgs['hr'].to(device).to(torch.float32)
        imgs_lr = imgs['lr'].to(device).to(torch.float32)
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *Dnet.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *Dnet.output_shape))), requires_grad=False)
        optimizer_G.zero_grad()
        gen_hr = Gnet(imgs_lr).to(device) #[ ] size error
        loss_pixel = criterion_pixel(gen_hr,imgs_hr)
        
        if (epoch==0):
            loss_pixel.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, num_epoch, i, len(dataloader), loss_pixel.item())
                 )   
            #[print(f"device{j} : {(torch.cuda.memory_reserved(j) - torch.cuda.memory_allocated(j))/(1024**2)} MB") for j in [0,1,2,3]]
            print(f"device{0} : {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0))/(1024**2) }MB \n")
            
            continue
        pred_real = Dnet(imgs_hr).detach()
        pred_fake = Dnet(gen_hr)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
        print(f'''
               device{0} : {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0))/(1024**2) }MB \n
               device{1} : {(torch.cuda.memory_reserved(1) - torch.cuda.memory_allocated(1))/(1024**2) }MB \n
               device{2} : {(torch.cuda.memory_reserved(2) - torch.cuda.memory_allocated(2))/(1024**2) }MB \n
               '''
               )

        gen_features = FE(gen_hr) # [ ] memo overflow
        real_features = FE(imgs_hr).detach()
        
        loss_content = criterion_content(gen_features, real_features)
        loss_G = loss_content + 5e-3 * loss_GAN + 1e-2 * loss_pixel
    
        loss_G.backward()
        optimizer_G.step()
        #
        # Discriminator
        #
        optimizer_D.zero_grad()
    
        pred_real = Dnet(imgs_hr)
        pred_fake = Dnet(gen_hr.detach())
    
        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)
    
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
    
        loss_D.backward()
        optimizer_D.step()# added 2nd Aug 2021
        with torch.no_grad(): 
            D_loss.append(loss_D.item())
            G_loss.append(loss_G.item())
            pixel_loss.append(loss_pixel.item())
            content_loss.append(loss_content.item())
            GAN_loss.append(loss_GAN.item())
            psnr_value.append(psnr.cal_psnr(imgs_hr,gen_hr).mean())
        print(f'''
                [{i}/{len(dataloader)}],[{epoch}/{num_epoch}]\t[D loss: {loss_D.item():.5f}\tG loss: {loss_G.item():.5f}]
                \npixel loss: {loss_pixel.item():.5f}
                \ncontent loss: {loss_content.item():.5f}
                \nGAN loss: {loss_GAN.item():.5f}
                \nPSNR value per batch: {psnr.cal_psnr(imgs_hr,gen_hr).mean()}
                '''
                 )
        if i == (len(dataloader)-1):
            torch.save(Gnet.state_dict(),'saved_models/%s_generator_%d_%d.pth'%(model_name,epoch,i),_use_new_zipfile_serialization=False)

            torch.save(Dnet.state_dict(),'saved_models/%s_discriminator_%d_%d.pth'%(model_name,epoch,i),_use_new_zipfile_serialization=False)
            torch.save(FE.state_dict(),'saved_models/%s_FE_%d_%d.pth'%(model_name,epoch,i),_use_new_zipfile_serialization=False)
            os.makedirs('images/training/3dimg_crop_less_%s/'%model_name,exist_ok=True)
            os.makedirs('saved_models/',exist_ok=True)
            # Save image grid with upsampled inputs and ESRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=2) # scale factor = times of magnification
            BN = nn.BatchNorm3d(1).to(device)
            # Normalize demonstrations
            imgs_lr = BN(imgs_lr)# 24MB per img
            gen_hr = BN(gen_hr) # 24MB per img
            img_grid = torch.cat((imgs_lr[:,:,imgs_lr.size(2)//2], gen_hr[:,:,imgs_lr.size(2)//2]), -1).detach().cpu()
            save_image(img_grid, "images/training/3dimg_crop_less_%s/%d_%d.png" % (model_name,i,epoch), nrow=1, normalize=False)
            torch.cuda.empty_cache()

        # --------------Empty cache here-------------------------------
        torch.cuda.empty_cache()
        del gen_hr
    return D_loss,G_loss,pixel_loss,content_loss,GAN_loss,psnr_value


# it ran 5 iterations then memo overflowed.....[2:23 a.m. 8 Juni 2021 ]
D_loss = []
G_loss = []
pixel_loss = []
content_loss = []
GAN_loss = []
psnr_value = []
for epoch in range(num_epoch):
    (D_l,G_l,pix_l,cont_l,GAN_l,p_value)=train_loop(data,
               model=[Dnet,Gnet,FE],
               loss_function = [criterion_pixel,criterion_GAN,criterion_content],
               optimizer= [optimizer_D,optimizer_G],
               device = device,epoch = epoch,num_epoch=num_epoch,checkpoint=checkpoint,model_name = model_name)
    D_loss.append(D_l)
    G_loss.append(G_l)
    pixel_loss.append(pix_l)
    content_loss.append(cont_l)
    GAN_loss.append(GAN_l)
    psnr_value.append(p_value)
dict_save = {'D_loss':D_loss,
             'G_loss':G_loss,
             'Pixel_loss':pixel_loss,
             'Content_loss':content_loss,
             'GAN_loss':GAN_loss,
             'PSNR_value':psnr_value
             }
os.makedirs('loss',exist_ok=True)
savemat(f'loss/table_3d_crop_less_{model_name}_ESRGAN_{num_epoch+checkpoint}ep.mat',dict_save,oned_as='row')


