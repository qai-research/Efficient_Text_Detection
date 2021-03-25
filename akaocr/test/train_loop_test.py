# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Wed January 06 14:30:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contain unit test for dataloader
_____________________________________________________________________________
"""
import argparse
import sys
import torch

sys.path.append("../")
from models.detec.heatmap import HEAT
from models.recog.atten import Atten
from engine import Trainer
from engine.config import setup, dict2namespace, load_yaml_config
from engine.trainer.loop import CustomLoopHeat, CustomLoopAtten
from engine.build import build_dataloader
from utils.data.dataloader import LoadTestDetecDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from engine.metric.accuracy import RecogAccuracy, DetecAccuracy
from engine.metric.evaluation import DetecEvaluation, RecogEvaluation

def test_recog(root_data_recog):
    cfg = setup("recog")
    cfg.SOLVER.DATA_SOURCE = root_data_recog
    model = Atten(cfg)
    model.to(device=cfg.SOLVER.DEVICE)

    evaluate = RecogEvaluation(cfg)
    acc = RecogAccuracy(cfg)
    lossc = CustomLoopAtten(cfg)
    train_loader = build_dataloader(cfg, root_data_recog)
    test_loader = build_dataloader(cfg, root_data_recog, selected_data=["CR_HW_JP_v1_1"])
    trainer = Trainer(cfg, model, train_loader=train_loader, test_loader=test_loader, custom_loop=lossc, resume=True)
    trainer.do_train()


def test_detec(root_data_detec, data_test_path):
    cfg = setup("detec")
    cfg.MODEL.NUM_CLASS = 3210
    cfg.SOLVER.DEVICE = str(device)
    cfg.SOLVER.DATA_SOURCE = root_data_detec
    print(cfg)
    
    model = HEAT(cfg)
    model.to(device=device)

    evaluate = DetecEvaluation(cfg)
    acc = DetecAccuracy(cfg)
    lossc = CustomLoopHeat(cfg)
    train_loader = build_dataloader(cfg, root_data_detec)
    test_loader = LoadTestDetecDataset(data_test_path, cfg)
    trainer = Trainer(cfg, model, train_loader=train_loader, test_loader=test_loader, custom_loop=lossc, accuracy=acc, evaluation=evaluate, resume=True)
    trainer.do_train()

def main():
    # root_data_recog = "/home/bacnv6/nghiann3/data/RECOG/"
    # root_data_detec = "/home/tanhv1/kleentext/akaocr/data/data_detec/train/"
    # data_test_path = '/home/tanhv1/kleentext/akaocr/data/data_detec/test/ST_Doc_WORD_v2_2/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_recog', type=str, help='path to recog data', default='/home/bacnv6/nghiann3/data/RECOG/')
    parser.add_argument('--root_data_detec', type=str, help='path to detect data', default='/home/tanhv1/kleentext/akaocr/data/data_detec/train/')
    parser.add_argument('--data_test_path', type=str, help='path to test detect data', default='/home/tanhv1/kleentext/akaocr/data/data_detec/test/ST_Doc_WORD_v2_2/')
    opt = parser.parse_args()
    test_recog(opt.root_data_recog)
    test_detec(opt.root_data_detec, opt.data_test_path)

if __name__ == '__main__':
    main()