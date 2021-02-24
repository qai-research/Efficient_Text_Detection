# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Thu January 28 15:14:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contain method to handle images before feeding to model
_____________________________________________________________________________
"""
import torch
import cv2
import numpy as np

def images2tensor(images, norm="imagenet"):
    tensors = list()
    check = 0
    for img in images:
        size = len(img.shape)
        if check == 0:
            check = size
        elif check != size:
            raise Exception(f"List of image have different dimension some is {check}, and there are {size}")

    for img in images:
        img = torch.from_numpy(img).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        tensors.append(img)
    # x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]

if __name__ == '__main__':
    image = torch.randn(20, 20, 3)
    print(image.shape)