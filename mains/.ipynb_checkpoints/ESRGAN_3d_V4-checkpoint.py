#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:54:00 2021

- [x] This script aims to produce whole brain images 

@author: qiwang
"""
import nibabel as nb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch import from_numpy
import os
from torch.autograd import Variable
from torchvision.utils import save_image
from scipy.io import savemat

from ESRGAN_3d_dataset import *
from toy_model_3d import *

datapath = '../utils/3T_v2_3D_GAN'
device = torch.device('cuda:0')
data = DataLoader(ImageDataset(datapath,(128,128,128)),batch_size = 1,num_workers=8,shuffle=True)

os.makedirs('loss/',exist_ok = True)
os.makedirs('images/training/3dimg_whole/',exist_ok = True)

_,img = next(enumerate(data))
Tensor = torch.cuda.FloatTensor

criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

checkpoint = 9

Dnet = Discriminator().to(device)
Gnet = Generator().to(device)
FE = FeatureExtractor().to(device)
if not checkpoint ==0:
    Dnet.load_state_dict(torch.load(f'saved_models/wholebrain_discriminator_{checkpoint}.pth'),strict = False)
    Gnet.load_state_dict(torch.load(f'saved_models/wholebrain_generator_{checkpoint}.pth'),strict = False)


optimizer_G = torch.optim.Adam(Gnet.parameters(), lr=.0002, betas=(0.9, .999))
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=.0002, betas=(0.9, .999))

num_epoch = 10
#%% training function
def train_loop(dataloader,model,loss_function,optimizer,device,epoch,num_epoch,checkpoint):
    '''
    model = list
    optimizer = list
    '''
    optimizer_D,optimizer_D = optimizer
    Dnet,Gnet,FE = model
    criterion_pixel,criterion_GAN,criterion_content=loss_function
    D_loss = []
    G_loss = []
    for i,imgs in enumerate(dataloader):
        
        imgs_hr = imgs['hr'].to(device)
        imgs_lr = imgs['lr'].to(device)
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *Dnet.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *Dnet.output_shape))), requires_grad=False)
        optimizer_G.zero_grad()
        gen_hr = Gnet(imgs_lr) #[ ] size error
        loss_pixel = criterion_pixel(gen_hr,imgs_hr)
        
        if ((epoch)==0):
            loss_pixel.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, num_epoch, i, len(dataloader), loss_pixel.item())
            )
            continue
        
        pred_real = Dnet(imgs_hr).detach()
        pred_fake = Dnet(gen_hr)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
        
        gen_features = FE(gen_hr) # [ ] memo overflow
        print(f"device 0 : {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0))/(1024**3)} GB")
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
        
            
        D_loss.append(loss_D.item())
        G_loss.append(loss_G.item())
        print(f'[{i}/{len(dataloader)}],[{epoch}/{num_epoch}\t[D loss: {loss_D.item():.5f} G loss: {loss_G.item():.5f}]]')
        torch.save(Gnet.state_dict(),'saved_models/wholebrain_generator_%d.pth'%(epoch))
        torch.save(Dnet.state_dict(),'saved_models/wholebrain_dicriminator_%d.pth'%(epoch))
        # --------------Empty cache here-------------------------------
        torch.cuda.empty_cache()
        
        # Save image grid with upsampled inputs and ESRGAN outputs
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=2) # scale factor = times of magnification
        BN = nn.BatchNorm3d(1).to(device)
        # Normalize demonstrations
        imgs_lr = BN(imgs_lr)# 24MB per img
        gen_hr = BN(gen_hr) # 24MB per img
        img_grid = torch.cat((imgs_lr[:,:,imgs_lr.size(2)//2], gen_hr[:,:,imgs_lr.size(2)//2]), -1).detach().cpu()
        save_image(img_grid, "images/training/3dimg_whole/%d_%d.png" % (i,epoch), nrow=1, normalize=False)
        torch.cuda.empty_cache()
    return D_loss,G_loss


# it ran 5 iterations then memo overflowed.....[2:23 a.m. 8 Juni 2021 ]
D_loss = []
G_loss = []
for epoch in range(num_epoch):
    epoch = epoch + checkpoint
    (D_l,G_l)=train_loop(data,
               model=[Dnet,Gnet,FE],
               loss_function = [criterion_pixel,criterion_GAN,criterion_content],
               optimizer= [optimizer_D,optimizer_G],
               device = device,epoch = epoch,num_epoch=num_epoch,checkpoint = checkpoint)
    D_loss.append(D_l)
    G_loss.append(G_l)
dict_save = {'D_loss':D_loss,
             'G_loss':G_loss}
savemat(f'loss/table_3d_whole2_ESRGAN_{num_epoch}ep.mat',dict_save,oned_as='row')

