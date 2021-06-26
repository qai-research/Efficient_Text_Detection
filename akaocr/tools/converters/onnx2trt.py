#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Huu Kim - Kimnh3
Created Date: Jun 12 16:22:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contains function to build tensorRT engine from onnx
_____________________________________________________________________________
"""

import sys
import tensorrt as trt
sys.path.append("../../")
from pipeline.util import experiment_loader
from engine.config import get_cfg_defaults
import pycuda.driver as cuda
import pycuda.autoinit
import time
import copy
import numpy as np
import os
import torch
import cv2
from pathlib import Path

import onnx
import argparse

from utils.utility import initial_logger
logger = initial_logger()

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
a=(int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


# def exp_onnx_loader(name='test', type='detec', data_path='../../data'):
#     data_path = Path(data_path)
#     if type == 'detec':
#         saved_models_path = 'exp_detec'
#     else:
#         saved_models_path = 'exp_recog'
#     data_path = data_path.joinpath(saved_models_path, name)
#     if not data_path.exists():
#         raise Exception("No experiment folder for", name)
#     saved_model = sorted(data_path.glob('*.onnx'))
#     saved_config = sorted(data_path.glob('*.yaml'))

#     if len(saved_model) < 1:
#         raise Exception("No onnx model for experiment ", name, type, "in", data_path)
#     if len(saved_config) < 1:
#         raise Exception("No onnx config for experiment ", name, type, "in", data_path)

#     return str(saved_model[0]), str(saved_config[0])

def GiB(val):
    return val * 1 << 30

def build_detec_engine(onnx_path, using_half, engine_file, dynamic_input=True, workspace_size=5, 
                min_shape=(1,3,256,256), opt_shape=(1,3,700,700), max_shape=(1,3,1200,1200)):
    trt.init_libnvinfer_plugins(None, '')
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1 # always 1 for explicit batch
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(int(workspace_size))
        if using_half:
            config.set_flag(trt.BuilderFlag.FP16)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        
        if dynamic_input:
            profile = builder.create_optimization_profile();
            profile.set_shape("input", min_shape, opt_shape, max_shape) 
            # profile.set_shape("input", (1,3,256,256), (1,3,700,700), (1,3,1200,1200)) 
            # profile.set_shape("input_1", (1,3,512,512), (1,3,1600,1600), (1,3,1024,1024)) 
            config.add_optimization_profile(profile)
        
        previous_output = network.get_output(0)
        network.unmark_output(previous_output)
        sigmoid_layer=network.add_activation(previous_output,trt.ActivationType.SIGMOID)
        network.mark_output(sigmoid_layer.get_output(0))
        return builder.build_engine(network, config) 


if __name__ == '__main__':
    # set dynamic input size in associate function, find: profile.set_shape
    parser = argparse.ArgumentParser(description="Exports akaOCR model from ONNX to TensorRT engine for inference")
    parser.add_argument("--input", required=False, help="Path to parent data folder, include 2 exp (parent of exp_<detec/recog>/<exp_name>)", default="../../data")
    parser.add_argument("--exp", required=False, help="Experiment name (default=\"test\")", default="test")
    parser.add_argument("--output", required=False, help="Path to save the generated RT engine", default="../../data")
    # parser.add_argument("--amp", required=False, help="Use FP16 (half precision), default=False", default=False)
    # parser.add_argument("--dynamic", required=False, help="Use dynamic input shape", default=True)
    # parser.add_argument("--workspace", required=False, help="Workspace size for export engine process (in GB), default=1 GB", default=1)
    args=parser.parse_args()
    logger.info("Converting detec ONNX to TensorRT engine...")

    onnx_detec_path, detec_model_config = experiment_loader(name=args.exp, type='detec', model_format='onnx',
                                                             data_path=args.input)
    # onnx_detec_path,_ = exp_onnx_loader(name=args.exp, type='detec',
    #                                                                  data_path=args.input)
    # onnx_recog_path,_ = exp_onnx_loader(name=args.exp, type='recog',
    #                                                                  data_path=args.input)
    
    # Process names
    # Set output path for onnx files
    output_detec_path = Path(args.output).joinpath("exp_detec", args.exp)
    # output_recog_path = Path(args.output).joinpath("exp_recog", args.exp)
    cfg_detec = get_cfg_defaults()
    cfg_detec.merge_from_file(detec_model_config)

    # Set name for onnx files
    # detec_name = Path(onnx_detec_path).stem
    # recog_name = Path(onnx_recog_path).stem
    output_detec = os.path.join(output_detec_path, "detec_trt.engine")
    # output_recog = os.path.join(output_recog_path, recog_name+"_recog.engine")
    # start build
    # for detec
    detec_engine=build_detec_engine(onnx_detec_path, cfg_detec.INFERENCE.TRT_AMP, output_detec, dynamic_input=cfg_detec.INFERENCE.TRT_DYNAMIC, 
                workspace_size=cfg_detec.INFERENCE.TRT_WORKSPACE, min_shape=cfg_detec.INFERENCE.TRT_MIN_SHAPE, 
                opt_shape=cfg_detec.INFERENCE.TRT_OPT_SHAPE, max_shape=cfg_detec.INFERENCE.TRT_MAX_SHAPE)
    with open(output_detec, "wb") as f:
        f.write(detec_engine.serialize())
    logger.info('Build RT engine of detec successfully!')

    # recog_engine=build_engine(onnx_detec_path, args.amp, output_recog, dynamic=args.dynamic, workspace_size=workspace)
    # with open(output_recog, "wb") as f:
    #     f.write(recog_engine.serialize())
    # print('Build RT engine of recog successfully!')

# Sample run cmd:
# CUDA_VISIBLE_DEVICES=0 python onnx2trt.py --amp=False --dynamic=True --workspace=5

# use trtexec cmd
# /usr/src/tensorrt/bin/trtexec --onnx=../../data/exp_detec/test/best_accuracy_detec.onnx --explicitBatch --fp16 --workspace=5000 --minShapes=input:1x3x256x256 --optShapes=input:1x3x700x700 --maxShapes=input:1x3x1200x1200 --buildOnly --saveEngine=../../data/exp_detec/test/akaocr_detec.engine
# CUDA_VISIBLE_DEVICES=0 /usr/src/tensorrt/bin/trtexec --onnx=../../data/exp_detec/test/best_accuracy_detec.onnx --explicitBatch --workspace=5000 --minShapes=input:1x3x256x256 --optShapes=input:1x3x700x700 --maxShapes=input:1x3x1200x1200 --buildOnly --saveEngine=../../data/exp_detec/test/akaocr_detec.engine