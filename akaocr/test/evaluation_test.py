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
import argparse
import sys
sys.path.append("../")
from models.detec.heatmap import HEAT
from models.recog.atten import Atten
import torch
from utils.utility import initial_logger
logger = initial_logger()

from engine.metric.evaluation import DetecEvaluation, RecogEvaluation
from engine.config import setup
from utils.data.dataloader import LoadTestDetecDataset

from engine.build import build_dataloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detec_test_evaluation(model_path, data_path):
    cfg = setup("detec")
    model = HEAT()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    test_loader = LoadTestDetecDataset(data_path, cfg)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_detec_path', type=str, help='path to detect model',
                        default='/home/tanhv1/kleentext/akaocr/data/saved_models_detec/smz_detec/best_accuracy.pth')
    parser.add_argument('--data_detec_path', type=str, help='path to detect data',
                        default='/home/tanhv1/kleentext/akaocr/data/data_detec/train/')
    parser.add_argument('--model_recog_path', type=str, help='path to test detect data',
                        default='/home/bacnv6/nghiann3/ocr-old/data/saved_models_recog/test1/best_accuracy.pth')
    parser.add_argument('--data_recog_path', type=str, help='path to test detect data',
                        default='/home/bacnv6/nghiann3/data/RECOG/')
    opt = parser.parse_args()

    # model_detec_path = '/home/tanhv1/kleentext/akaocr/data/saved_models_detec/smz_detec/best_accuracy.pth'
    # data_detec_path = '/home/tanhv1/kleentext/akaocr/data/data_detec/train/'
    detec_test_evaluation(opt.model_detec_path, opt.data_detec_path)

    # model_recog_path = '/home/bacnv6/nghiann3/ocr-old/data/saved_models_recog/test1/best_accuracy.pth'
    # data_recog_path = '/home/bacnv6/nghiann3/data/RECOG/'
    recog_test_evaluation(opt.model_recog_path, opt.data_recog_path)
