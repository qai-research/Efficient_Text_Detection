
"""
_____________________________________________________________________________

This file contain backbones and feature extractor for text detection
_____________________________________________________________________________
"""

from torch import nn
from torchvision.models import resnet152
import torch
from utils.torchutils import init_weights

class Resnet_Extractor(torch.nn.Module):
    def __init__(self):
        super(Resnet_Extractor, self).__init__()
        self.resnet = resnet152()
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]) # get layers of resnet except Dense (last layer)
        self.slice0 = nn.Sequential()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for i in range(0,3):    # before layer 1    (output size: 64 x H/2 x W/2)   (original size: 3 x H x W)
            self.slice0.add_module(str(i), self.resnet[i])
        for i in range(3,5):    # layer 1   (output size: 256 x H/4 x W/4)
            self.slice1.add_module(str(i), self.resnet[i])
        self.slice2.add_module('5', self.resnet[5])     # layer 2 (output size: 512 x H/8 x W/8)
        self.slice3.add_module('6', self.resnet[6])     # layer 3 (output size: 1024 x H/16 x W/16)
        self.slice4.add_module('7', self.resnet[7])     # layer 4 (output size: 2048 x H/32 x W/32)
        
        init_weights(self.slice0.modules())
        init_weights(self.slice1.modules())
        init_weights(self.slice2.modules())
        init_weights(self.slice3.modules())
        init_weights(self.slice4.modules())

    def forward(self, x):
        x = self.slice0(x)
        y_level1 = x
        x = self.slice1(x)
        y_level2 = x
        x = self.slice2(x)
        y_level3 = x
        x = self.slice3(x)
        y_level4 = x
        x = self.slice4(x)
        
        return (x,y_level4,y_level3,y_level2,y_level1)