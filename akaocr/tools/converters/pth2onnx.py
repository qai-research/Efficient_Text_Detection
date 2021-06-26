# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Huu Kim - Kimnh3
Created Date: May 24 14:31:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contain code for converting trained model into ONNX format, for both detec and recog
Refer from TensorRT example: tensorrt/bin/python/onnx_packnet
_____________________________________________________________________________
"""
import sys
import os
from pathlib import Path

from icecream import ic
import torch
import onnx
import numpy as np
import argparse
import onnx_graphsurgeon as gs

sys.path.append("../../")
from pipeline.util import experiment_loader
from engine.config import get_cfg_defaults


from models.modules.converters import AttnLabelConverter
from utils.file_utils import read_vocab

from models.detec.heatmap import HEAT
from models.detec.resnet_fpn_heatmap import HEAT_RESNET
from models.detec.efficient_heatmap import HEAT_EFFICIENT
from models.recog.atten import Atten
from utils.utility import initial_logger
logger = initial_logger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_akaocr(input, exp, output, args):
    """Load the akaocr network and export it to ONNX
    """
    logger.info("Converting detec pth to onnx...")

    detec_model_path, detec_model_config = experiment_loader(name=exp, type='detec',
                                                             data_path=input)
    # recog_model_path, recog_model_config = experiment_loader(name=exp, type='recog',
    #                                                          data_path=input)
    
    # Load config come with trained model
    cfg_detec = get_cfg_defaults()
    cfg_detec.merge_from_file(detec_model_config)

    # Processing recog cfg:

    # cfg_recog = get_cfg_defaults()
    # cfg_recog.merge_from_file(recog_model_config)
    # if cfg_recog.MODEL.VOCAB is not None:
    #     cfg_recog.MODEL.VOCAB = os.path.join(input, "vocabs", cfg_recog.MODEL.VOCAB)
    # cfg_recog.SOLVER.DEVICE = 'cpu'#str(device)
    # if cfg_recog.MODEL.VOCAB:  # vocabulary is given
    #     cfg_recog.MODEL.VOCAB = read_vocab(cfg_recog.MODEL.VOCAB)
    #     cfg_recog["character"] = cfg_recog.MODEL.VOCAB
    # else:  # use character list instead
    #     cfg_recog["character"] = list(cfg_recog["character"])
    # if cfg_recog.SOLVER.UNKNOWN:
    #     cfg_recog["character"].append(cfg_recog.SOLVER.UNKNOWN)
    # cfg_recog["character"].sort()
    # if 'CTC' in cfg_recog.MODEL.PREDICTION:
    #     converter = CTCLabelConverter(cfg_recog["character"])
    # else:
    #     converter = AttnLabelConverter(cfg_recog["character"], device='cpu')#cfg_recog.SOLVER.DEVICE)
    # cfg_recog.MODEL.NUM_CLASS = len(converter.character)

    # Set output path for onnx files
    output_detec_path = Path(output).joinpath("exp_detec", exp)
    # output_recog_path = Path(output).joinpath("exp_recog", exp)

    # Set name for onnx files
    # detec_name = Path(detec_model_path).stem
    # recog_name = Path(recog_model_path).stem
    output_detec = os.path.join(output_detec_path, "detec_onnx.onnx")
    # output_recog = os.path.join(output_recog_path, "recog_onnx.onnx")

    # Dummy input data for 2 models
    input_tensor = torch.randn((1, 3, 768, 768), requires_grad=False)
    input_tensor_recog = torch.randn(1, 1, 32, 128)
    txt = ["xxx"]
    converter = AttnLabelConverter(["x", "X", "o"], device='cpu')#device)
    # text, length = converter.encode(txt, max_label_length=cfg_recog.MODEL.MAX_LABEL_LENGTH)

    # Build the model
    model_detec = HEAT()
    # model_recog = Atten(cfg_recog)

    model_name = str(cfg_detec.MODEL.NAME)
    if model_name == "HEAT":
        model_detec = HEAT()
    elif model_name == "RESNET":
        model_detec = HEAT_RESNET()
    elif model_name == "EFFICIENT":
        model_detec = HEAT_EFFICIENT()

    # model_name = str(cfg_recog.MODEL.NAME) # Only ATTEN
    # if model_name == "ATTEN":
    #     model_recog = Atten(cfg_recog)

    pretrain = torch.load(detec_model_path)
    model_detec.load_state_dict(pretrain, strict=False)
    # Into eval mode
    model_detec.eval()
    
    # pretrain = torch.load(recog_model_path)
    # model_recog.load_state_dict(pretrain, strict=False)
    # Into eval mode
    # model_recog.eval()
    

    # Convert the model into ONNX
    # For detec model
    torch.onnx.export(model_detec, input_tensor, output_detec,
                      verbose=cfg_detec.INFERENCE.OX_VERBOSE, opset_version=cfg_detec.INFERENCE.OX_OPSET,
                      do_constant_folding=cfg_detec.INFERENCE.OX_DO_CONSTANT_FOLDING,
                      export_params=cfg_detec.INFERENCE.OX_EXPORT_PARAMS,
                      input_names=["input"],
                      output_names=["output", "281"],
                      dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}})#,
                                    #"output": {0: "Transposeoutput_dim_0", 1: "Transposeoutput_dim_1", 2: "Transposeoutput_dim_2"}})
                                    # "218": {0: "Transposeoutput_dim_0", 1: 32, 2: "Relu281_dim_2", 3: "Relu281_dim_3"}})
    logger.info("Convert detec pth to ONNX sucess; converting recog pth to ONNX.")
                                    
    # For recog model: not run well yet, cause of onnx not support adaptive pooling
    # torch.onnx.export(model_recog, (input_tensor_recog, text), output_recog,
    #                   verbose=args.verbose, opset_version=args.opset,
    #                   do_constant_folding=True,
    #                     export_params=True,
    #                   input_names=["input", "text"],
    #                   output_names=["output"],
    #                   dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"}})
    # logger.info("Convert recog pth to ONNX sucess.")
    

def main():
    parser = argparse.ArgumentParser(description="Exports akaOCR model to ONNX, and post-processes it to insert TensorRT plugins")
    parser.add_argument("--input", required=True, help="Path to parent data folder, include 2 exp (parent of exp_<detec/recog>/<exp_name>)", default="../../data")
    parser.add_argument("--exp", required=True, help="Experiment name (default=\"test\")", default="test")
    parser.add_argument("--output", required=True, help="Path to save the generated ONNX model", default="../../data")
    args=parser.parse_args()

    build_akaocr(args.input, args.exp, args.output, args)
    # Print result
    # print("Export models from .pth to .onnx, check at: ", args.output)

if __name__ == '__main__':
    main()

# example cmd:
# run this onnx convert
# python convert_to_onnx.py --input ../../data --exp test --output ../../data

# trtexec to create engine
# /usr/src/tensorrt/bin/trtexec --onnx=<name>.onnx --explicitBatch --fp16 --workspace=5000 --minShapes=input:1x3x256x256 --optShapes=input:1x3x700x700 --maxShapes=input:1x3x1200x1200 --buildOnly --saveEngine=akaocr_detec.engine

# /usr/src/tensorrt/bin/trtexec --onnx=../../data/exp_detec/test/best_accuracy_detec.onnx --explicitBatch --fp16 --workspace=5000 --minShapes=input:1x3x256x256 --optShapes=input:1x3x700x700 --maxShapes=input:1x3x1200x1200 --buildOnly --saveEngine=../../data/exp_detec/test/akaocr_detec.engine
# CUDA_VISIBLE_DEVICES=0 /usr/src/tensorrt/bin/trtexec --onnx=../../data/exp_detec/test/best_accuracy_detec.onnx --explicitBatch --workspace=5000 --minShapes=input:1x3x256x256 --optShapes=input:1x3x700x700 --maxShapes=input:1x3x1200x1200 --buildOnly --saveEngine=../../data/exp_detec/test/akaocr_detec.engine