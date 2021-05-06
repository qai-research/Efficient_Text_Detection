# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Huu Kim - Kimnh3
Created Date: Mar 31, 2021 6:31pm GMT+0700
Project : AkaOCR core
_____________________________________________________________________________

This file contain code for train recog
_____________________________________________________________________________
"""
import sys
import torch

sys.path.append("../")
from models.recog.atten import Atten
from engine import Trainer
from engine.config import setup, parse_base
from engine.trainer.loop import CustomLoopHeat, CustomLoopAtten
from engine.build import build_dataloader
from engine.metric.accuracy import RecogAccuracy, DetecAccuracy
from engine.metric.evaluation import DetecEvaluation, RecogEvaluation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_recog(args):
    cfg = setup("recog", args)
    cfg.SOLVER.DATA_SOURCE = args.data_recog
    model = Atten(cfg)
    model.to(device=cfg.SOLVER.DEVICE)

    evaluate = RecogEvaluation(cfg)
    acc = RecogAccuracy(cfg)
    lossc = CustomLoopAtten(cfg)
    train_loader = build_dataloader(cfg, args.data_recog)
    test_loader = build_dataloader(cfg, args.data_test_recog)
    trainer = Trainer(cfg, model, train_loader=train_loader, test_loader=test_loader, custom_loop=lossc, accuracy=acc,
                      evaluation=evaluate, resume=True)
    trainer.do_train()

def main():
    parser = parse_base()
    parser.add_argument('--data_recog', type=str, default="../data/data_recog/train", help='path to recog data')
    parser.add_argument('--data_test_recog', type=str, default="../data/data_recog/val", help='path to test recog data')
    args = parser.parse_args()
    test_recog(args)

if __name__ == '__main__':
    main()