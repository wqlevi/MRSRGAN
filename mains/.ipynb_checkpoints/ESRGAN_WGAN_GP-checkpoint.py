'''
- [x] save optmizer.stat_dict as well, refer to pytorch doc!!!  
- [ ] The WGAN-GP-SR still face gradient vanishing problem, and diverge after 30th epoch 
- [ ] implement Differentiable augmentation GAN method to improve training
'''
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import from_numpy
import os
from torch.autograd import Variable
from torchvision.utils import save_image,make_grid
import glob
import re
import argparse

from data.ESRGAN_3d_dataset import CropDataset
from model.ESRGAN_3d_model_Copy import *
from model.model_wgan_pg import compute_gradient_penalty,get_grads_D_WAN,get_grads_G
from utils import pixel_shuffle3d, utils
from torch.cuda.amp import autocast

# set up argparser
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--path',type=str,default='/ptmp/wangqi/transfer_folder/LS200X_Norm/train_crops')
parser.add_argument('--val_path',type=str,default='/ptmp/wangqi/transfer_folder/LS200X_Norm/val_crops/crops')
parser.add_argument('--checkpoint',type=int,default=0)
parser.add_argument('--model_name',type=str,default='Jupyter_C14')
parser.add_argument('--precision',type=int,default=0,help="if 1 -> Float32Tensor | 0 -> HalfTensor ; dont call 16bit while using autocast()")
opt = parser.parse_args()
print(opt)
# argument parser
datapath = opt.path
val_path = opt.val_path
device = torch.device('cuda:0')
batch_size = opt.batch_size
data = DataLoader(CropDataset(datapath),batch_size = batch_size,drop_last=True,num_workers=8,shuffle=True)
val_data = DataLoader(CropDataset(val_path),batch_size = batch_size,drop_last=True,num_workers=2,shuffle=True)
model_name = opt.model_name
Tensor = torch.cuda.FloatTensor if opt.precision else torch.cuda.HalfTensor  # half precision is used when default args
num_epoch = 20
checkpoint = opt.checkpoint

# model config
mse = nn.MSELoss()
save_every_iter = 500
Dnet = Discriminator().to(device) # used as critic net here
Gnet = Generator().to(device)
optimizer_G = torch.optim.Adam(Gnet.parameters(), lr=.0002, betas=(0.9, .999))
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=.0002, betas=(0.9, .999))

# ckp
if not checkpoint == 0:
    ckp = torch.load("saved_models/%s_Crop_%d_%d.pth"%(model_name,checkpoint,2*save_every_iter-1))
    #checkpoint_model_G = f'../saved_models/{model_name}_generator_{checkpoint}_{len(data)-1}.pth'
    #checkpoint_model_D = f'../saved_models/{model_name}_discriminator_{checkpoint}_{len(data)-1}.pth'
    Dnet.load_state_dict(ckp['Dnet_state_dict'])
    Gnet.load_state_dict(ckp['Gnet_state_dict'])
    optimizer_D.load_state_dict(ckp['optimizer_D_state_dict'])
    optimizer_G.load_state_dict(ckp['optimizer_G_state_dict'])



# number of parameters per network/model
print('# generator parameters:', sum(param.numel() for param in Gnet.parameters()))
print('# discriminator parameters:', sum(param.numel() for param in Dnet.parameters()))

valid = Variable(Tensor(np.ones((batch_size, *Dnet.output_shape))), requires_grad=False)
fake = Variable(Tensor(np.zeros((batch_size, *Dnet.output_shape))), requires_grad=False)


def train_loop(epoch,dataloader,valid,fake,device,writer):
    Gnet.train()
    Dnet.train()
    
    for i,imgs in enumerate(dataloader):
        imgs_lr = imgs['lr'].to(device)
        imgs_hr = imgs['hr'].to(device)
        # training D
        Dnet.zero_grad()

        gen_imgs = Gnet(imgs_lr)
        logits_fake = Dnet(gen_imgs).mean()
        logits_real = Dnet(imgs_hr).mean()
        g_p = compute_gradient_penalty(Dnet,imgs_hr,gen_imgs)
        d_loss = logits_fake - logits_real + 10*g_p
        
        d_loss.backward(retain_graph=True)
        optimizer_D.step()
        dtg,dbg = get_grads_D_WAN(Dnet)

        # training G
        Gnet.zero_grad()
        img_loss = mse(gen_imgs,imgs_hr)
        adv_loss = -1*Dnet(gen_imgs).mean()
        g_loss = img_loss + 1e-3*adv_loss

        g_loss.backward(retain_graph=True)
        optimizer_G.step()
        gtg,gbg = get_grads_G(Gnet)
        
        # Write Gnet gradient at end of epoch
        if i == (len(dataloader)-1):    
            for tag, parm in Gnet.named_parameters():
                writer.add_histogram("Gnet/"+tag, parm.grad.data.cpu().numpy(), epoch)
            for tag, parm in Dnet.named_parameters():
                writer.add_histogram("Dnet/"+tag, parm.grad.data.cpu().numpy(), epoch)
                
        with torch.no_grad(): 
            psnr_v = utils.psnr(gen_imgs[0].squeeze().cpu().numpy(),imgs_hr[0].squeeze().cpu().numpy())
            ssim_v = utils.ssim(gen_imgs[0].squeeze().cpu().numpy(),imgs_hr[0].squeeze().cpu().numpy())
            writer.add_scalar('Loss/train_D_loss',d_loss.item(), i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_G_loss',g_loss.item(), i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_psnr',psnr_v, i+len(dataloader)*epoch)
            writer.add_scalar('Loss/train_ssim',ssim_v, i+len(dataloader)*epoch)
            
            if (i+1)%save_every_iter == 0:      
                os.makedirs("saved_models",exist_ok=True)
                torch.save({
                'epoch': epoch,
                'Gnet_state_dict': Gnet.state_dict(),
                'Dnet_state_dict': Dnet.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict()
                }, 'saved_models/%s_Crop_%d_%d.pth'%(model_name,epoch,i))
        
        # validation
        Gnet.eval()
        _,img = next(enumerate(val_data))
        sr = Gnet(img['lr'].to(device))
        with torch.no_grad():
            psnr_val = utils.psnr(sr.squeeze().cpu().numpy(),img['hr'].to(device).squeeze().cpu().numpy())
            ssim_val = utils.ssim(sr.squeeze().cpu().numpy(),img['hr'].to(device).squeeze().cpu().numpy())
            writer.add_scalar('Loss/val_psnr',psnr_val, i+len(dataloader)*epoch)
            writer.add_scalar('Loss/val_ssim',ssim_val, i+len(dataloader)*epoch)
        print(f"[{i}/{len(dataloader)}]|[epoch:{epoch}]")
        img_grid = torch.cat((imgs_hr[:,:,sr.size(2)//2], gen_imgs[:,:,sr.size(2)//2]), -1).detach().cpu()
        writer.add_images(f"img/",img_grid/img_grid.max(),i+len(dataloader)*epoch,dataformats='NCHW')
        #vis_plot(sr)
        writer.flush()
        
if __name__ == '__main__':
    writer = SummaryWriter(comment=f"{model_name}/WGAN_GP")
    if not checkpoint:
        [m.apply(utils.weights_init) for m in [Dnet,Gnet]]
    for epoch in range(num_epoch): 
        train_loop(epoch+checkpoint,data,valid,fake,device,writer)
    writer.close()
