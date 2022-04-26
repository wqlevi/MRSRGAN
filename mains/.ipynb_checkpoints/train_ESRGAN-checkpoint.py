#!/usr/bin/env python
# coding: utf-8

# - [x] all on same GPU A100, Raven. Remote visualization session
# - ~[x] PSNR value still incorrect(9 Mar 2022)~
# - [x] half precision works on gradient but not on data!(gradient vanish) 
# - [x] tensorboard writting loss, gradient, and intermediate image
# - [x] weight initialized by Xavier
# - [x] WGAN training plan (24 Mar 2022)
# - [ ] Gnet gradient vanished after warm-up epoch
# - [x] lr decay scheduler

import nibabel as nb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch import from_numpy
import os
import argparse

from tqdm import tqdm
from torch.autograd import Variable
from torchvision.utils import save_image
from scipy.io import savemat
from skimage.metrics import structural_similarity

import glob
import re
import sys
sys.path.append('utils/')


from data.ESRGAN_3d_dataset import CropDataset
from model.ESRGAN_3d_model import *
from model.model_wgan_pg import compute_gradient_penalty,get_grads_D_WAN,get_grads_G
from utils import pixel_shuffle3d, utils
from torch.cuda.amp import autocast


print(f"GPU information:\n{torch.cuda.get_device_properties(0)}")

# ***Argparser***
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--path',type=str,default='')
parser.add_argument('--checkpoint',type=int,default=0)
parser.add_argument('--model',type=str,default='')
parser.add_argument('--precision',type=int,default=1,help="if 1 -> Float32Tensor | 0 -> HalfTensor ; dont call 16bit while using autocast()")
opt = parser.parse_args()
print(opt)

scaler_G = torch.cuda.amp.GradScaler()
scaler_D = torch.cuda.amp.GradScaler()

batch_size = opt.batch_size
datapath = opt.path
model_name = opt.model
checkpoint = opt.checkpoint
precision = opt.precision


device = torch.device('cuda:0')
data = DataLoader(CropDataset(datapath),batch_size = batch_size,drop_last=True,num_workers=8,shuffle=True)

if not precision:
    Tensor = torch.cuda.HalfTensor
else:
    Tensor = torch.cuda.FloatTensor


criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

save_every_iter = 500

Dnet = Discriminator().to(device)
Gnet = Generator().to(device)
FE = FeatureExtractor().to(device)
[m.apply(utils.weights_init) for m in [Dnet,Gnet,FE]] # init weight for each network's conv3d

if not checkpoint == 0:
    checkpoint_model_G = f'../saved_models/{model_name}_Crop_generator_{checkpoint}_999.pth'
    checkpoint_model_D = f'../saved_models/{model_name}_Crop_discriminator_{checkpoint}_999.pth'
    checkpoint_model_FE = f'../saved_models/{model_name}_Crop_FE_{checkpoint}_999.pth'
    Dnet.load_state_dict(torch.load(f'{checkpoint_model_G}'),strict=False)
    Gnet.load_state_dict(torch.load(f'{checkpoint_model_D}'),strict=False)
    FE.load_state_dict(torch.load(f'{checkpoint_model_FE}'),strict=False)
    
optimizer_G = torch.optim.Adam(Gnet.parameters(), lr=.0002, betas=(0.9, .999))
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=.0002, betas=(0.9, .999))

# lr scheduler
scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G,gamma=0.9)
scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D,gamma=0.9)

num_epoch = 20

        
#%% training function
def train_loop(dataloader,model,loss_function,optimizer,device,epoch,num_epoch,checkpoint,model_name:str="jupyter"):
    '''
    model = list
    optimizer = list
    '''
    warm_up = False # whether want a warm up session for Gnet
    epoch = epoch + checkpoint
    optimizer_D,optimizer_D = optimizer
    Dnet,Gnet,FE = model
    criterion_pixel,criterion_GAN,criterion_content=loss_function
    for i,imgs in enumerate(dataloader):
        
        imgs_hr = imgs['hr'].to(device)
        imgs_lr = imgs['lr'].to(device)
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *Dnet.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *Dnet.output_shape))), requires_grad=False)
        optimizer_G.zero_grad() # clean all grad at begining of update
        with autocast():
            gen_hr = Gnet(imgs_lr) 
            loss_pixel = criterion_pixel(gen_hr,imgs_hr)
        
        if (epoch==0 and warm_up == True):
            loss_pixel.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, num_epoch, i, len(dataloader), loss_pixel.item())
                 )   
            #print(f"device 0 : {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0))/(1024**2) }MB \n")
            continue
        
        with autocast():
            pred_real = Dnet(imgs_hr).detach()
            pred_fake = Dnet(gen_hr)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
            #print(f"device 0 : {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0))/(1024**2) }MB \n")
        
        with autocast():
            gen_features = FE(gen_hr) # [ ] memo overflow
            real_features = FE(imgs_hr).detach()
        
            loss_content = criterion_content(gen_features, real_features)
            loss_G = loss_content + 5e-3 * loss_GAN + 1e-2 * loss_pixel
        
        scaler_G.scale(loss_G).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()
        #
        # Discriminator
        #
        optimizer_D.zero_grad()
        with autocast():
            pred_real = Dnet(imgs_hr)
            pred_fake = Dnet(gen_hr.detach())
  
        # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)
   
        # Total loss
            loss_D = (loss_real + loss_fake) / 2
    
        scaler_D.scale(loss_D).backward()
        scaler_D.step(optimizer_D)
        
        scaler_D.update()
        ##################
        # scheduler update
        ##################
        scheduler_D.step()
        scheduler_G.step()
        # Write Gnet gradient at end of epoch
        if i == (len(dataloader)-1):    
            for tag, parm in Gnet.named_parameters():
                writer.add_histogram("Gnet/"+tag, parm.grad.data.cpu().numpy(), epoch)
            for tag, parm in Dnet.named_parameters():
                writer.add_histogram("Dnet/"+tag, parm.grad.data.cpu().numpy(), epoch)
            [print(sched.get_last_lr()) for sched in [scheduler_D,scheduler_G]] # add print lr per iteration
      
        with torch.no_grad(): 
            psnr_v = utils.psnr(gen_hr[0].squeeze().cpu().numpy(),imgs_hr[0].squeeze().cpu().numpy())
            ssim_v = utils.ssim(gen_hr[0].squeeze().cpu().numpy(),imgs_hr[0].squeeze().cpu().numpy())
            if np.isnan(loss_D.item()) or np.isnan(loss_G.item()):
                sys.exit("Gradient vanished! Exiting...")
            writer.add_scalar('Loss/train_D_loss',loss_D.item(), i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_G_loss',loss_G.item(), i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_GAN_loss',loss_GAN.item(), i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_content_loss',loss_content.item(), i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_pixel_loss',loss_pixel.item(), i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_psnr',psnr_v, i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_ssim',ssim_v, i+len(dataloader)*epoch)
            
            print(f'[{i}/{len(dataloader)}],[{epoch}/{num_epoch}\t[D loss: {loss_D.item():.5f} G loss: {loss_G.item():.5f}]]')
            print(f"device 0 : {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0))/(1024**2) }MB \n")
            if (i+1)%save_every_iter == 0:  
                os.makedirs('saved_models',exist_ok=True) 
                
                torch.save(Gnet.state_dict(),'saved_models/%s_Crop_generator_%d_%d.pth'%(model_name,epoch,i),_use_new_zipfile_serialization=False)
                torch.save(Dnet.state_dict(),'saved_models/%s_Crop_discriminator_%d_%d.pth'%(model_name,epoch,i),_use_new_zipfile_serialization=False)
                torch.save(FE.state_dict(),'saved_models/%s_Crop_FE_%d_%d.pth'%(model_name,epoch,i),_use_new_zipfile_serialization=False)

               
                               
        # validate
        Gnet.eval()
        sr = Gnet(imgs_lr)
        print(f"[{i}/{len(dataloader)}]|[epoch:{epoch}]")
        img_grid = torch.cat((imgs_hr[:,:,sr.size(2)//2], sr[:,:,sr.size(2)//2]), -1).detach().cpu()

        writer.add_images(f"img/",img_grid/img_grid.max(),i+len(dataloader)*epoch,dataformats='NCHW')
        # --------------Empty cache here-------------------------------
        writer.flush()
        torch.cuda.empty_cache()


writer = SummaryWriter(comment=f"{model_name}_BATCH_{batch_size}")

for epoch in tqdm(range(num_epoch)):
    train_loop(data,
               model=[Dnet,Gnet,FE],
               loss_function = [criterion_pixel,criterion_GAN,criterion_content],
               optimizer= [optimizer_D,optimizer_G],
               device = device,epoch = epoch,num_epoch=num_epoch,checkpoint=checkpoint,model_name = model_name)
writer.close()

