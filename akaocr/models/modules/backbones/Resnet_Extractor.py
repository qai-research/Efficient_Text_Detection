
"""
_____________________________________________________________________________

This file contain backbones and feature extractor for text detection
_____________________________________________________________________________
"""

from torch import nn
from torchvision.models import resnet152
import torch
class Resnet(torch.nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.resnet = resnet152()
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        print(self.resnet)
        self.slice0 = nn.Sequential()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for i in range(0,3):    # before layer 1
            self.slice0.add_module(str(i), self.resnet[i])
        for i in range(3,5):    # layer 1   (3 blocks)
            self.slice1.add_module(str(i), self.resnet[i])
        self.slice2.add_module('5', self.resnet[5])     # layer 2 (8 blocks)
        self.slice3.add_module('6', self.resnet[6])     # layer 3 (36 blocks)
        self.slice4.add_module('7', self.resnet[7])     # layer 4 (3 blocks)
    
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