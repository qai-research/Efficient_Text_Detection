#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Mon November 03 10:00:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain backbones and feature extractor for every models
_____________________________________________________________________________
"""

from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models

from utils.torchutils import init_weights

class FpnFeature(torch.nn.Module):
    def __init__(self, freeze=False):
        super(FpnFeature, self).__init__()
        vgg_features = models.vgg16_bn().features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):  # conv2_2
            self.slice1.add_module(str(x), vgg_features[x])
        for x in range(12, 19):  # conv3_3
            self.slice2.add_module(str(x), vgg_features[x])
        for x in range(19, 29):  # conv4_3
            self.slice3.add_module(str(x), vgg_features[x])
        for x in range(29, 39):  # conv5_3
            self.slice4.add_module(str(x), vgg_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )

        init_weights(self.slice1.modules())
        init_weights(self.slice2.modules())
        init_weights(self.slice3.modules())
        init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())  # no pre-trained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():  # only first conv
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
