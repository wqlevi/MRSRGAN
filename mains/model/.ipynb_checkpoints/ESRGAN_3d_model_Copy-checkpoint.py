#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 22:44:39 2021

- [x] all on same GPU A100, Raven. Remote visualization session

@author: qiwang
"""
import torch.nn as nn
import torch
from utils.pixel_shuffle3d import PixelShuffle3d


# last MaxPooling removed for C13, also due to what ESRGAN paper stated.
VGG16 = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M']
class FeatureExtractor(nn.Module):
    """
    Extract feature into shape of 7,8,7
    """
    def __init__(self,in_channels=1):
        super(FeatureExtractor,self) .__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16)
        
    def forward(self,x):
        x = self.conv_layers(x)
        return x
    
    def create_conv_layers(self,ar):
        layers = []
        in_channels = self.in_channels
        
        for x in ar:
            if type(x) == int:
                out_channels = x
                
                layers += [nn.Conv3d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=3,stride=1,padding=1),
   #                        nn.BatchNorm3d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool3d(kernel_size=2,stride=2)]
                
        return nn.Sequential(*layers)

class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv3d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        ''' original 
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )
        '''
        # personalized BN layer for Gnet, added to WGAN-GP for stablizing gradient
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters), nn.BatchNorm3d(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x
    
class Generator(nn.Module): # interpolation scheme happens
    def __init__(self, channels=1, filters=64, num_res_blocks=1, num_upsample=1):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Conv3d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks1 = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        self.res_blocks3 = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv3d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv3d(filters, filters * 8, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                PixelShuffle3d(scale=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv3d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks1(out1)
        out = self.res_blocks3(out)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.input_shape = (1,64,64,64) # hr image shape
        in_channels, in_height, in_width, in_depth = self.input_shape
        patch_h, patch_w, patch_d = int(in_height / 2 ** 4), int(in_width / 2 ** 4), int(in_depth / 2 ** 4) # meaning of 4: layers of conv
        self.output_shape = (1, patch_h, patch_w, patch_d) # Dnet output shape

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv3d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv3d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))#half size
            layers.append(nn.BatchNorm3d(out_filters)) 
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv3d(out_filters, 1, kernel_size=3, stride=1, padding=1)) # 4 Conv layers in block

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
