# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyun Jo <shjo.april@gmail.com>

import math
import torch

from torch import nn
from torch.nn import functional as F

from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.ReLU(inplace=True)

        self.upsample = upsample

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.block = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1)

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res_x = x

        x = self.block(x)
        x = self.conv(x)
        x = self.norm(x)

        x = self.act(x + res_x)
        return x
        
class Transformer(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, out_channels=32, kernel_size=9, stride=1),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2),
        )
        
        self.res_block = nn.Sequential(
            ResBlock(in_channels=128),
            ResBlock(in_channels=128),
            ResBlock(in_channels=128),
            ResBlock(in_channels=128),
            ResBlock(in_channels=128),
        )

        self.decoder = nn.Sequential(
            ConvBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, upsample=True),
            ConvBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, upsample=True),
            nn.Conv2d(in_channels=32, out_channels=in_channels, kernel_size=9, stride=1, padding=9//2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_block(x)
        x = self.decoder(x)

        return x

class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def preprocess(self, x):
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        x = x.clamp(0, 255).div_(255.0)
        return (x - mean) / std

    def forward(self, x):
        x = self.preprocess(x)

        x = self.slice1(x); relu1_2 = x
        x = self.slice2(x); relu2_2 = x
        x = self.slice3(x); relu3_3 = x
        x = self.slice4(x); relu4_3 = x
        
        return {
            'relu1_2': relu1_2,
            'relu2_2': relu2_2,
            'relu3_3': relu3_3,
            'relu4_3': relu4_3,
        }
