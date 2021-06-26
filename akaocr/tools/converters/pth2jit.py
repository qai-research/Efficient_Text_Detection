#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Huu Kim - Kimnh3
Created Date: Wed Jun 9 13:00:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contains converter from torch .pth model to torchscript (jit) .pt model for Triton
_____________________________________________________________________________
"""
import sys
sys.path.append("../../")
from icecream import ic
import torch
import cv2
import numpy as np
import shutil
import os
import re
from PIL import Image
import argparse
from pathlib import Path
import torch.nn.functional as F

from utils.file_utils import read_vocab
from models.detec.heatmap import HEAT
from models.detec.resnet_fpn_heatmap import HEAT_RESNET
from models.detec.efficient_heatmap import HEAT_EFFICIENT
from models.recog.atten import Atten
from engine.solver import ModelCheckpointer
from models.modules.converters import AttnLabelConverter
from engine.infer.heat2boxes import Heat2boxes
from engine.infer.intercept_vocab import InterceptVocab
from engine.config import get_cfg_defaults
from utils.utility import initial_logger
from pipeline.util import AlignCollate, experiment_loader
from pre.image import ImageProc


logger = initial_logger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_detec(exp_name='test', data_path='../../data', newname='detec_pt.pt'):
    model_path, model_config = experiment_loader(name=exp_name, type='detec',
                                                                data_path=data_path)
    config = model_config
    model_path = model_path
    # base_name=os.path.basename(model_path)
    # base_name=base_name[:base_name.index('.')]

    logger.info(f"load model from : {model_path}")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)

    if cfg.MODEL.NAME == "CRAFT":
        model = HEAT()
    elif cfg.MODEL.NAME == "RESNET":
        model = HEAT_RESNET()
    elif cfg.MODEL.NAME == "EFFICIENT":
        model = HEAT_EFFICIENT()

    checkpointer = ModelCheckpointer(model)
    #strict_mode=False (default) to ignore different at layers/size between 2 models, otherwise, must be identical and raise error.
    checkpointer.resume_or_load(model_path, strict_mode=False)
    model = model.to(device)
    # Switch the model to eval model
    model.eval()

    # Prepare name
    # Set output path for .pt file
    output_path = Path(data_path).joinpath("exp_detec", exp_name)

    # Set name for .pt files, default='<model_name>_pt.pt' as default of Triton config
    # output_name = os.path.join(output_path, base_name+'_pt.pt') 
    output_name = os.path.join(output_path, newname) 

    # An example input you would normally provide to your model's forward() method.
    x = torch.randn(1, 3, 768, 768).to(device)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, x)

    # Save the TorchScript model
    traced_script_module.save(output_name)
    print("Convert detec completed!")

def convert_recog(exp_name='test', data_path='../../data', newname='recog_pt.pt'):
    model_path, model_config = experiment_loader(name=exp_name, type='recog',
                                                                data_path=data_path)
    config = model_config
    model_path = model_path
    # base_name=os.path.basename(model_path)
    # base_name=base_name[:base_name.index('.')]
    logger.info(f"load model from : {model_path}")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(config)
    cfg.MODEL.TRANSFORMATION = str(cfg.MODEL.TRANSFORMATION)
    if cfg.MODEL.VOCAB is not None:
        cfg.MODEL.VOCAB = os.path.join(data_path, "vocabs", cfg.MODEL.VOCAB)
    cfg.SOLVER.DEVICE = str(device)
    if cfg.MODEL.VOCAB:  # vocabulary is given
        cfg.MODEL.VOCAB = read_vocab(cfg.MODEL.VOCAB)
        cfg["character"] = cfg.MODEL.VOCAB
    else:  # use character list instead
        cfg["character"] = list(cfg["character"])
    if cfg.SOLVER.UNKNOWN:
        cfg["character"].append(cfg.SOLVER.UNKNOWN)
    cfg["character"].sort()
    if 'CTC' in cfg.MODEL.PREDICTION:
        converter = CTCLabelConverter(cfg["character"])
    else:
        converter = AttnLabelConverter(cfg["character"], device=cfg.SOLVER.DEVICE)
    cfg.MODEL.NUM_CLASS = len(converter.character)
    # cfg.MODEL.NUM_CLASS = 3210
    # cfg.SOLVER.DEVICE = device
    model = Atten(cfg)

    checkpointer = ModelCheckpointer(model)
    #strict_mode=False (default) to ignore different at layers/size between 2 models, otherwise, must be identical and raise error.
    checkpointer.resume_or_load(model_path, strict_mode=False)
    model = model.to(device)
    # Switch the model to eval model
    model.eval()

    # Prepare name
    # Set output path for .pt file
    output_path = Path(data_path).joinpath("exp_recog", exp_name)

    # Set name for .pt files, default='<model_name>_pt.pt' as default of Triton config
    # output_name = os.path.join(output_path, base_name+'_pt.pt') 
    output_name = os.path.join(output_path, newname) 

    # An example input you would normally provide to your model's forward() method.
    x = torch.randn(1, 1, 32, 128).cuda().to(device)
    text = ["xxx"]
    converter = AttnLabelConverter(["x", "X", "o"], device=device)
    text, length = converter.encode(text, max_label_length=cfg.MODEL.MAX_LABEL_LENGTH)
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, (x, text))

    # Save the TorchScript model
    traced_script_module.save(output_name)
    print("Convert recog completed!")

def main():
    parser = argparse.ArgumentParser(description="Convert torch model (.pth) into torchscript model (.pt)")
    parser.add_argument("--input", required=False, help="Path to input torch model .pth", default='../../data')
    parser.add_argument("--exp", required=False, help="Experiment name", default='test')
    args=parser.parse_args()

    convert_detec(exp_name=args.exp, data_path=args.input)
    convert_recog(exp_name=args.exp, data_path=args.input)
    # Print result
    print("Converted successfully, check at: ", args.input)

if __name__ == '__main__':
    main()
