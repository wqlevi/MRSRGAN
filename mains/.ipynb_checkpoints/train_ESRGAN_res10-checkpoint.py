#!/usr/bin/env python
# coding: utf-8

# - [x] all on same GPU A100, Raven. Remote visualization session
# - ~[x] PSNR value still incorrect(9 Mar 2022)~
# - [x] tensorboard writting loss, gradient, and intermediate image
# - [x] weight initialized by Xavier


# - [x] load pre-trained MedicalNet weight to FE, and freeze it during training

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
from skimage.metrics import structural_similarity

import glob
import re
import sys
sys.path.append('utils/')


from data.ESRGAN_3d_dataset import CropDataset
from model.model_VGG16_IN import *
from model.model_pretrain_resnet import *
from model.model_base import *
from utils.utils import *




print(f"GPU information:\n{torch.cuda.get_device_properties(0)}")

# ***Argparser***
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--path',type=str,default='')
parser.add_argument('--checkpoint',type=int,default=0)
parser.add_argument('--model',type=str,default='')
parser.add_argument('--precision',type=int,default=1,help="if 1 -> Float32Tensor | 0 -> HalfTensor ; dont call 16bit while using autocast()")
parser.add_argument('--lr',type=float,default=.0002)
opt = parser.parse_args()
print(opt)

# Initial parameters
batch_size = opt.batch_size
datapath = opt.path
model_name = opt.model
checkpoint = opt.checkpoint
precision = opt.precision
lr = opt.lr


device = torch.device('cuda:0')
data = DataLoader(CropDataset(datapath),batch_size = batch_size,drop_last=True,num_workers=8,shuffle=True)

if not precision:
    Tensor = torch.cuda.HalfTensor
else:
    Tensor = torch.cuda.FloatTensor


criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

num_epoch = 20
save_every_iter = 500

Dnet = Discriminator().to(device)
Gnet = Generator().to(device)
FE = resnet10( sample_input_W=32,
                sample_input_H=32,
                sample_input_D=32,num_seg_classes=32).to(device) # pre-trained FE
freeze_model(FE)

optimizer_G = torch.optim.Adam(Gnet.parameters(), lr=lr, betas=(0.9, .999))
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=lr, betas=(0.9, .999))

if not checkpoint == 0:
    ckp = torch.load("saved_models/%s_Crop_%d_%d.pth"%(model_name,checkpoint,2*save_every_iter-1))
    Dnet.load_state_dict(ckp['Dnet_state_dict'])
    Gnet.load_state_dict(ckp['Gnet_state_dict'])
    FE.load_state_dict(ckp['FE_state_dict'])
    optimizer_D.load_state_dict(ckp['optimizer_D_state_dict'])
    optimizer_G.load_state_dict(ckp['optimizer_G_state_dict'])

    



        
#%% training function
def train_loop(dataloader,model,loss_function,optimizer,device,epoch,num_epoch,checkpoint,model_name,writer):
    '''
    model = list
    optimizer = list
    '''
    epoch += checkpoint
    num_epoch += checkpoint
    optimizer_D,optimizer_G = optimizer# revised 3rd Aug 2021
    Dnet,Gnet,FE = model
    criterion_pixel,criterion_GAN,criterion_content=loss_function
    
    Dnet.train()
    Gnet.train()
    
    warm_up = True # whether want a warm up session for Gnet
    GD_update_ratio = 5
    valid = Variable(Tensor(np.ones((batch_size, *Dnet.output_shape))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((batch_size, *Dnet.output_shape))), requires_grad=False)
    for i,imgs in enumerate(dataloader):
        imgs_hr = imgs['hr'].to(device)
        imgs_lr = imgs['lr'].to(device)

        #################
        # Generator
        #################
        
        optimizer_G.zero_grad() # clean all grad at begining of update
        gen_hr = Gnet(imgs_lr) 
        loss_pixel = criterion_pixel(gen_hr,imgs_hr)

        if (epoch==0 and warm_up == True):
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

        gen_features = FE(gen_hr) 
        real_features = FE(imgs_hr).detach()

        loss_content = criterion_content(gen_features, real_features)
        loss_G = loss_content + 5e-3 * loss_GAN + 1e-2 * loss_pixel

        loss_G.backward()
        optimizer_G.step()


        #################
        # Discriminator
        #################
        optimizer_D.zero_grad()

        pred_real = Dnet(imgs_hr)
        pred_fake = Dnet(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        if i%GD_update_ratio == 0: # update D net every 5 Gnet step
            loss_D.backward()
            optimizer_D.step()

        # Write Gnet gradient at each Dnet update
        if not i%5:     
            for tag, parm in Gnet.named_parameters():
                writer.add_histogram("Gnet/"+tag, parm.grad.data.cpu().numpy(), epoch)
            for tag, parm in Dnet.named_parameters():
                writer.add_histogram("Dnet/"+tag, parm.grad.data.cpu().numpy(), epoch)


        with torch.no_grad(): 

            

            writer.add_scalar('Loss/train_D_loss',loss_D.item(), i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_G_loss',loss_G.item(), i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_GAN_loss',loss_GAN.item(), i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_content_loss',loss_content.item(), i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_pixel_loss',loss_pixel.item(), i+len(dataloader)*epoch)
            
            print(f'[{i}/{len(dataloader)}],[{epoch}/{num_epoch}\t[D loss: {loss_D.item():.5f} G loss: {loss_G.item():.5f}]]')
            

            if (i+1)%save_every_iter == 0:       
                os.makedirs('saved_models/',exist_ok=True) 
                
                torch.save({
                'epoch': epoch,
                'Gnet_state_dict': Gnet.state_dict(),
                'Dnet_state_dict': Dnet.state_dict(),
                'FE_state_dict':FE.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict()
                }, 'saved_models/%s_Crop_%d_%d.pth'%(model_name,epoch,i))

        Gnet.eval()
        sr = Gnet(imgs_lr)
        print(f"[{i}/{len(dataloader)}]|[epoch:{epoch}]")
        img_grid = torch.cat((imgs_hr[:,:,sr.size(2)//2], gen_hr[:,:,sr.size(2)//2]), -1).detach().cpu()
        with torch.no_grad():
            psnr_v = psnr(sr.squeeze().cpu().numpy(),imgs_hr.squeeze().cpu().numpy())
            ssim_v = ssim(sr.squeeze().cpu().numpy(),imgs_hr.squeeze().cpu().numpy())
        writer.add_scalar('Loss/train_psnr',psnr_v, i+len(dataloader)*epoch)
        writer.add_scalar('Loss/train_ssim',ssim_v, i+len(dataloader)*epoch)
        writer.add_images(f"img/",img_grid/img_grid.max(),i+len(dataloader)*epoch,dataformats='NCHW')
        writer.flush()

        # --------------Empty cache here-------------------------------
        torch.cuda.empty_cache()

if __name__ == '__main__':

    writer = SummaryWriter(log_dir=f'{model_name}/run1',comment=f"{model_name}_5G1D")
    #hparams={'lr': lr, 'bsize': batch_size}
    metrics={'Loss/train_D_loss':None,
             'Loss/train_G_loss':None,
             'Loss/train_GAN_loss':None,
             'Loss/train_pixel_loss':None,
             'Loss/train_content_loss':None,
             'Loss/train_psnr':None,
             'Loss/train_ssim':None
            }
    if not checkpoint:
        [m.apply(weights_init) for m in [Dnet,Gnet]]
    for epoch in tqdm(range(num_epoch)):
        train_loop(data,
               model=[Dnet,Gnet,FE],
               loss_function = [criterion_pixel,criterion_GAN,criterion_content],
               optimizer= [optimizer_D,optimizer_G],
               device = device,epoch = epoch,num_epoch=num_epoch,checkpoint=checkpoint,model_name = model_name,writer=writer)
    
    #writer.add_hparams(hparams,metrics)
    writer.close()






