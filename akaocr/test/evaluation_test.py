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
from engine.config import setup, parse_base
from utils.data.dataloader import LoadTestDetecDataset

from engine.build import build_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def detec_test_evaluation(model_path, data_path):
    cfg = setup("detec")
    model = HEAT()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    test_loader = LoadTestDetecDataset(data_path, cfg)
    evaluation = DetecEvaluation(cfg)
    evaluation.run(model, test_loader)


def recog_test_evaluation(model_path, data_path):
    cfg = setup("recog")
    model = Atten(cfg)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=False)
    test_loader = build_dataloader(cfg, data_path, selected_data=["CR_sample1"])
    evaluation = RecogEvaluation(cfg)
    evaluation.run(model, test_loader)


def main():
    parser = parse_base()
    parser.add_argument('--w_detec', type=str, help='path to detect model')
    parser.add_argument('--data_detec', type=str, help='path to detect data')
    parser.add_argument('--w_recog_', type=str, help='path to test detect data')
    parser.add_argument('--data_recog', type=str, help='path to test detect data')
    args = parser.parse_args()

    if args.w_detec is not None and args.data_detec is not None:
        detec_test_evaluation(args.model_detec_path, args.data_detec_path)

    if args.w_recog is not None and args.data_recog is not None:
        recog_test_evaluation(args.model_recog_path, args.data_recog_path)


if __name__ == '__main__':
    main()
