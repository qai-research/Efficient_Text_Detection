#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________

This file contain heatmap models for text detection
_____________________________________________________________________________
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path

from utils.torchutils import init_weights
from models.modules.backbones.Resnet_Extractor import Resnet_Extractor

class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class RESNET_FPN_HEAT(nn.Module):
    def __init__(self, freeze=False):
        super(RESNET_FPN_HEAT, self).__init__()

        # Base network
        self.resnet = Resnet_Extractor()

        # U network
        # output chanel of resnet = [2048, 1024, 512, 256, 64]
        self.upconv1 = DoubleConv(2048, 1024, 512)
        self.upconv2 = DoubleConv(512, 512, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(64, 64, 16)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, num_class, kernel_size=1),
        )
        
        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.resnet(x)
        """ U network """
        y = sources[0]
        y = F.interpolate(y, size=sources[1].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[1]], dim=1)
        y = self.upconv1(y)
        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)
        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature