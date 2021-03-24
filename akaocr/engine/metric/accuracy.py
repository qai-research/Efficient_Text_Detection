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
from models.modules.converters import CTCLabelConverter, AttnLabelConverter
from engine.metric import recog_eval
class RecogAccuracy():
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.MODEL.PREDICTION == 'CTC':
            self.converter = CTCLabelConverter(cfg.MODEL.VOCAB)
            self.criterion = torch.nn.CTCLoss(zero_infinity=True).to(self.cfg.SOLVER.DEVICE)
        elif cfg.MODEL.PREDICTION == 'Attn':
            self.converter = AttnLabelConverter(cfg.MODEL.VOCAB, device=cfg.SOLVER.DEVICE)
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(self.cfg.SOLVER.DEVICE)
        else:
            raise ValueError(f"invalid model prediction type")
       
    def run(self, model, inputs):
        with torch.no_grad():
            _,accuracy,_,_,_,_,_,_ = recog_eval.validation(model, self.criterion, inputs, self.converter, self.cfg)
        return accuracy
    
class DetecAccuracy():
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, model, inputs):
        return None