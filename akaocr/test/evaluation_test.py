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
from models.detec.resnet_fpn_heatmap import HEAT_RESNET
from models.detec.efficient_heatmap import HEAT_EFFICIENT
from models.recog.atten import Atten
import torch
from utils.utility import initial_logger

logger = initial_logger()

from engine.metric.evaluation import DetecEvaluation, RecogEvaluation
from engine.config import setup, parse_base
from engine.build import build_dataloader, build_test_data_detec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def detec_test_evaluation(args):
    cfg = setup("detec", args)
    if cfg.MODEL.NAME == "CRAFT":
        model = HEAT()
    elif cfg.MODEL.NAME == "RESNET":
        model = HEAT_RESNET()
    elif cfg.MODEL.NAME == "EFFICIENT":
        model = HEAT_EFFICIENT()
    model.load_state_dict(torch.load(args.w_detec, map_location=torch.device(device)))
    model = model.to(device)
    test_loader = build_test_data_detec(cfg, args.data_test_detec, selected_data=None)
    evaluation = DetecEvaluation(cfg)
    evaluation.run(model, test_loader)


def recog_test_evaluation(args):
    cfg = setup("recog", args)
    model = Atten(cfg)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(args.w_recog, map_location=torch.device(device)), strict=False)
    model = model.to(device)
    test_loader = build_dataloader(cfg, args.data_recog)
    evaluation = RecogEvaluation(cfg)
    evaluation.run(model, test_loader)


def main():
    parser = parse_base()
    parser.add_argument('--w_detec', type=str, help='path to detect model')
    parser.add_argument('--data_test_detec', type=str, help='path to detect data')
    parser.add_argument('--w_recog', type=str, help='path to test detect data')
    parser.add_argument('--data_recog', type=str, help='path to test detect data')
    args = parser.parse_args()

    if args.w_detec is not None and args.data_test_detec is not None:
        detec_test_evaluation(args)

    if args.w_recog is not None and args.data_recog is not None:
        recog_test_evaluation(args)


if __name__ == '__main__':
    main()
