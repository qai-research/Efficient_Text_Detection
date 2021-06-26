#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Ngoc Nghia - Nghiann3
Created Date: Fri March 12 13:00:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file compute accuracy while training
_____________________________________________________________________________
"""

import torch
from engine.metric import recog_eval
class RecogAccuracy():
    def __init__(self, cfg, criterion, converter):
        self.cfg = cfg
        self.criterion = criterion
        self.converter = converter
               
    def run(self, model, inputs):
        with torch.no_grad():
            inputs = zip([inputs[0]], [inputs[1]])
            _,accuracy,_,_,_,_,_,_ = recog_eval.validation(model, self.criterion, inputs, self.converter, self.cfg)
        return accuracy
    
class DetecAccuracy():
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, model, inputs):
        return None