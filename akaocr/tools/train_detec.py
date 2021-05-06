# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Huu Kim - Kimnh3
Created Date: Mar 31, 2021 6:31pm GMT+0700
Project : AkaOCR core
_____________________________________________________________________________

This file contain code for train detec
_____________________________________________________________________________
"""
import sys
import torch

sys.path.append("../")
from models.detec.heatmap import HEAT
from models.detec.resnet_fpn_heatmap import HEAT_RESNET
from models.detec.efficient_heatmap import HEAT_EFFICIENT
from engine import Trainer
from engine.config import setup, parse_base
from engine.trainer.loop import CustomLoopHeat, CustomLoopAtten
from engine.build import build_dataloader
from engine.build import build_dataloader, build_test_data_detec
from engine.metric.accuracy import RecogAccuracy, DetecAccuracy
from engine.metric.evaluation import DetecEvaluation, RecogEvaluation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_detec(args):
    cfg = setup("detec", args)
    cfg.MODEL.NUM_CLASS = 3210
    cfg.SOLVER.DEVICE = str(device)
    cfg.SOLVER.DATA_SOURCE = args.data_detec

    if cfg.MODEL.NAME == "CRAFT":
        model = HEAT()
    elif cfg.MODEL.NAME == "RESNET":
        model = HEAT_RESNET()
    elif cfg.MODEL.NAME == "EFFICIENT":
        model = HEAT_EFFICIENT()    
    model.to(device=device)

    evaluate = DetecEvaluation(cfg)
    acc = DetecAccuracy(cfg)
    lossc = CustomLoopHeat(cfg)
    train_loader = build_dataloader(cfg, args.data_detec)
    test_loader = build_test_data_detec(cfg, args.data_test_detec, selected_data=None)
    trainer = Trainer(cfg, model, train_loader=train_loader, test_loader=test_loader, custom_loop=lossc, accuracy=acc,
                      evaluation=evaluate, resume=True)
    trainer.do_train()

def main():
    parser = parse_base()
    parser.add_argument('--data_detec', type=str, default="../data/data_detec/train", help='path to detect data')
    parser.add_argument('--data_test_detec', type=str, default="../data/data_detec/val",
                        help='path to test detect data')
    args = parser.parse_args()
    test_detec(args)

if __name__ == '__main__':
    main()