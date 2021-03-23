#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Ngoc Nghia - Nghiann3
Created Date: Fri March 12 13:00:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contains unit test for model evaluation
_____________________________________________________________________________
"""

import sys
sys.path.append("../")
from models.detec.heatmap import HEAT
from models.recog.atten import Atten
import torch
from utils.utility import initial_logger
logger = initial_logger()

from engine.metric.evaluation import DetecEvaluation, RecogEvaluation
from engine.config import setup
from utils.data.dataloader import load_test_dataset_detec

from engine.build import build_dataloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detec_test_evaluation(model_path, data_path):
    cfg = setup("detec")
    model = HEAT()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    test_loader = load_test_dataset_detec(data_path)
    evaluation = DetecEvaluation(cfg)
    evaluation.run(model, test_loader)
    
def recog_test_evaluation(model_path, data_path):
    cfg = setup("recog")
    model = Atten(cfg)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    test_loader = build_dataloader(cfg, data_path, selected_data=["CR_sample1"])
    evaluation = RecogEvaluation(cfg)
    evaluation.run(model, test_loader)
        
if __name__=='__main__':
    model_detec_path = '/home/nghianguyen/smz_detec/best_accuracy.pth'
    data_detec_path = '/home/nghianguyen/train_data/lake_detec/ST_Demo_1'
    detec_test_evaluation(model_detec_path, data_detec_path)

    model_recog_path = '/home/nghianguyen/ocr_compare/ocr_new/data/saved_models_recog/smz_recog/best_accuracy.pth'
    data_recog_path = '/home/nghianguyen/train_data/lake_recog'
    recog_test_evaluation(model_recog_path, data_recog_path)