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
from utils.file_utils import read_vocab
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_data_recog = "/home/bacnv6/data/train_data/lake_recog"
root_data_detec = "/home/bacnv6/data/train_data/lake_detec"


def test_recog():
    cfg = setup("recog")
    cfg.SOLVER.DEVICE = str(device)
    cfg.SOLVER.DATA_SOURCE = root_data_recog
    print(cfg)
    cfg.MODEL.VOCAB = read_vocab(cfg.MODEL.VOCAB)
    cfg.MODEL.NUM_CLASS = len(cfg.MODEL.VOCAB)

    lossc = CustomLoopAtten(cfg)
    # print(cfg)
    model = Atten(cfg)
    model.to(device=device)
    do_train(cfg, model, custom_loop=lossc, resume=True)


def test_detec():
    cfg = setup("detec")
    cfg.MODEL.NUM_CLASS = 3210
    cfg.SOLVER.DEVICE = str(device)
    cfg.SOLVER.DATA_SOURCE = root_data_detec
    print(cfg)
    model = HEAT(cfg)
    model.to(device=device)

    lossc = CustomLoopHeat(cfg)
    trainer = Trainer(cfg, model, custom_loop=lossc, resume=True)
    trainer.do_train()


if __name__ == '__main__':
    # test_recog()
    test_detec()