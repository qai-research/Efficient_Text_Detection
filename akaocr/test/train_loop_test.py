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

import sys
import torch

sys.path.append("../")
from models.detec.heatmap import HEAT
from models.recog.atten import Atten
from engine import Trainer
from engine.config import setup, dict2namespace, load_yaml_config
from engine.trainer.loop import CustomLoopHeat, CustomLoopAtten
from engine.build import build_dataloader
from utils.data.dataloader import load_test_dataset_detec
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# root_data_recog = "/home/bacnv6/data/train_data/lake_recog"
# root_data_detec = "/home/bacnv6/data/train_data/lake_detec"

root_data_recog = "/home/nghianguyen/train_data/lake_recog"
root_data_detec = "/home/nghianguyen/train_data/lake_detec"
data_test_path = '/home/nghianguyen/train_data/lake_detec/ST_Demo_1'
from engine.metric.accuracy import RecogAccuracy, DetecAccuracy
from engine.metric.evaluation import DetecEvaluation, RecogEvaluation
def test_recog():
    cfg = setup("recog")
    cfg.SOLVER.DATA_SOURCE = root_data_recog
    model = Atten(cfg)
    model.to(device=cfg.SOLVER.DEVICE)

    evaluate = RecogEvaluation(cfg)
    acc = RecogAccuracy(cfg)
    lossc = CustomLoopAtten(cfg)
    train_loader = build_dataloader(cfg, root_data_recog)
    test_loader = build_dataloader(cfg, root_data_recog, selected_data=["CR_sample3"])
    trainer = Trainer(cfg, model, train_loader=train_loader, test_loader=test_loader, custom_loop=lossc, accuracy=acc, evaluation=evaluate, resume=True)
    trainer.do_train()


def test_detec():
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
    test_loader = build_dataloader(cfg, root_data_detec, selected_data=["ST_Demo_1"])
    test_loader = load_test_dataset_detec(data_test_path)
    trainer = Trainer(cfg, model, train_loader=train_loader, test_loader=test_loader, custom_loop=lossc, accuracy=acc, evaluation=evaluate, resume=True)
    trainer.do_train()

if __name__ == '__main__':
    # test_recog()
    test_detec()
