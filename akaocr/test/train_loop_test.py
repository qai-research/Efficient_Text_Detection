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
from engine import do_train
from engine.config import setup
from engine.config import setup, dict2namespace, load_yaml_config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_data_recog = "/home/bacnv6/data/train_data/lake_recog"
root_data_detec = "/home/bacnv6/data/train_data/lake_detec"


def test_recog():
    config = setup("recog")
    config.MODEL.NUM_CLASS = 3210
    config.SOLVER.DEVICE = str(device)
    config.SOLVER.DATA_SOURCE = root_data_recog
    print(config)
    model = Atten(config)
    model.to(device=device)
    do_train(config, model, resume=False)


def test_detec():
    config = setup("detec")
    config.MODEL.NUM_CLASS = 3210
    config.SOLVER.DEVICE = str(device)
    config.SOLVER.DATA_SOURCE = root_data_detec
    print(config)
    model = HEAT(config)
    model.to(device=device)
    do_train(config, model, resume=False)


if __name__ == '__main__':
    # test_recog()
    test_detec()