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


def normalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)
    if len(img.shape) == 2:
        img -= np.array([mean[0] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0], dtype=np.float32)
    else:
        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def denormalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(img, max_size, interpolation, mag_ratio=1):
    try:
        height, width, channel = img.shape
    except:
        height, width = img.shape
        channel = 1
    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > max_size:
        target_size = max_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap

if __name__ == '__main__':
    image = torch.randn(20, 20, 3)
    print(image.shape)