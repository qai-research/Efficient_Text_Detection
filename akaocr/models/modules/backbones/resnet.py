
"""
_____________________________________________________________________________

This file contain backbones and feature extractor for text detection
_____________________________________________________________________________
"""

from torch import nn
from torchvision.models import resnet152
import torch
class resnet(torch.nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet = resnet152()
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        # print(self.resnet)
        self.slice0 = nn.Sequential()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for i in range(0,3):
            self.slice0.add_module(str(i), self.resnet[i])
        for i in range(3,5):
            self.slice1.add_module(str(i), self.resnet[i])
        self.slice2.add_module('5', self.resnet[5])
        self.slice3.add_module('6', self.resnet[6])
        self.slice4.add_module('7', self.resnet[7]) 
    
    def forward(self, x):
        x = self.slice0(x)
        x0 = x
        x = self.slice1(x)
        x1 = x
        x = self.slice2(x)
        x2 = x
        x = self.slice3(x)
        x3 = x
        x = self.slice4(x)
        
        return (x,x3,x2,x1,x0)