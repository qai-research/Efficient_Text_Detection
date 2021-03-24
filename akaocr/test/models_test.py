#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Mon November 03 10:00:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain unit test for models
_____________________________________________________________________________
"""

import sys

sys.path.append("../")
import os
import torch
import numpy as np
import cv2

from models.detec.heatmap import HEAT
from models.recog.atten import Atten
from models.modules.converters import AttnLabelConverter
from engine.config import setup, dict2namespace, load_yaml_config
from engine.build import build_dataloader
#################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model_detec():
    model = HEAT()

    model = model.to(device)
    x = torch.randn(1, 3, 768, 768).to(device)
    print(x.shape)
    y = model(x)
    print(y[0].shape)
    print(y[1].shape)


def test_model_recog():
    config_recog_yaml = '../data/attention_resnet_base_v1.yaml'
    config = load_yaml_config(config_recog_yaml)
    config = dict2namespace(config)
    config.MODEL.NUM_CLASS = 3210
    config.SOLVER.DEVICE = device
    model = Atten(config)
    model.to(device=device)

    x = torch.randn(1, 1, 32, 128)
    x = x.cuda()
    x = x.to(device=device)

    text = ["xxx"]
    converter = AttnLabelConverter(["x", "X", "o"], device=device)
    text, length = converter.encode(text, max_label_length=config.MODEL.MAX_LABEL_LENGTH)
    y = model(x, text)
    print(y.shape)
    # print(y)


if __name__ == '__main__':
    test_model_detec()
    test_model_recog()
