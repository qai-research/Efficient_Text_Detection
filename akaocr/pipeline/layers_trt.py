#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Huu Kim - Kimnh3
Created Date: Jun 12 16:22:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contains pipeline of whole model by tensorRT self.engine
_____________________________________________________________________________
"""
import ctypes

import torch
import cv2
import numpy as np
import shutil
import os
import re
from PIL import Image
import torch.nn.functional as F
import pycuda.autoinit

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
from pipeline.util import AlignCollate
from pre.image import ImageProc
from pathlib import Path

import pycuda.driver as cuda
import tensorrt as trt
from utils.runtime import start_timer, end_timer_and_print


logger = initial_logger()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRT_LOGGER = trt.Logger()

# Utils

def exp_engine_loader(name='test', type='detec', data_path='../../data'):
    data_path = Path(data_path)
    if type == 'detec':
        saved_models_path = 'exp_detec'
    else:
        saved_models_path = 'exp_recog'
    data_path = data_path.joinpath(saved_models_path, name)
    if not data_path.exists():
        raise Exception("No experiment folder for", name)
    saved_model = sorted(data_path.glob('*.engine'))
    saved_config = sorted(data_path.glob('*.yaml'))

    if len(saved_model) < 1:
        raise Exception("No model for experiment ", name, type, "in", data_path)
    if len(saved_config) < 1:
        raise Exception("No config for experiment ", name, type, "in", data_path)

    return str(saved_model[0]), str(saved_config[0])

class SlideWindow():
    """
    A basic slide window.
    Attributes
    ----------
    preprocess : callable
        the preprocess function whose input is a image,

    Methods
    -------
    __call__
        return list of windows from left to right, top to bottom
    """

    def __init__(self, window=(1280, 800), bufferx=50, buffery=20, preprocess=None):
        super().__init__()
        self.preprocess = preprocess
        self.window = window
        self.bufferx = bufferx
        self.buffery = buffery

    def __call__(self, img):
        if self.preprocess is not None:
            img = self.preprocess(img)

        original_shape = img.shape
        repeat_x = int(original_shape[1] // (self.window[0] - self.bufferx / 2) + 1)
        repeat_y = int(original_shape[0] // (self.window[1] - self.buffery / 2) + 1)
        # print(repeatx, repeaty)
        all_windows = []
        for i in range(repeat_y):
            perpen_window = []
            for j in range(repeat_x):
                crop_img = img[(self.window[1] - self.buffery) * i:(self.window[1] - self.buffery) * i + self.window[1],
                           (self.window[0] - self.bufferx) * j:(self.window[0] - self.bufferx) * j + self.window[0]]
                perpen_window.append(crop_img)
            all_windows.append(perpen_window)

        return all_windows


class Detectlayer():
    """
    A basic pipeline for performing detection.
    Attributes
    ----------
    preprocess : callable
        the preprocess function whose input is a image.
    detector : type of detector will be use (heatmap)

    config : path to config file for the detector, this will provide model with information and pretrained model
    Methods
    -------
    __call__
        execute the pipeline
    """

    def __init__(self, config=None, model_path=None, model_name='test', data_path='../../data',
                 window=(1280, 800), bufferx=50, buffery=20, preprocess=None, postprocess=None, engine_path=None, cuda_ctx=None, input_shape=None):
        super().__init__()
        if not config:
            engine_path, detec_model_config = exp_engine_loader(name=model_name, type='detec',
                                                                     data_path=data_path)
            config = detec_model_config
            model_path = engine_path
        logger.info(f"load model from : {model_path}")

        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file(config)

        self.engine_path=engine_path
        self.window_shape = window
        self.bufferx = bufferx
        self.buffery = buffery
        
        ##new trt
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()
        self.input_shape = input_shape

    def detect(self, img):
        img_resized, target_ratio = ImageProc.resize_aspect_ratio(
            img, self.cfg.MODEL.MAX_SIZE, interpolation=cv2.INTER_LINEAR
        )
        ratio_h = ratio_w = 1 / target_ratio
        img_resized = ImageProc.normalize_mean_variance(img_resized)
        # img_resized=np.asarray(img_resized).astype('float32')
        img_resized = torch.from_numpy(img_resized).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        img_resized = (img_resized.unsqueeze(0))  # [c, h, w] to [b, c, h, w]

        height, width = img_resized.shape[2:4]
        self.input_shape = (height,width)
        img_resized = img_resized.cpu().detach().numpy()
        
        # img_resized = img_resized.numpy()
        print(6,img_resized.shape)
        
        #from cn ex
        segment_inputs, segment_outputs, segment_bindings = self._allocate_buffers()
        
        stream = cuda.Stream()    

        with self.engine.create_execution_context() as context:
            context.active_optimization_profile = 0
            origin_inputshape=context.get_binding_shape(0)
            
            if (origin_inputshape[-1]==-1):
                origin_inputshape[-2],origin_inputshape[-1]=(self.input_shape)
                context.set_binding_shape(0,(origin_inputshape))
            
            input_img_array = np.array([img_resized] * self.engine.max_batch_size)
            img = torch.from_numpy(input_img_array).float().numpy()
            segment_inputs[0].host = img
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in segment_inputs]#Copy from the Python buffer src to the device pointer dest (an int or a DeviceAllocation) asynchronously,
            stream.synchronize()#Wait for all activity on this stream to cease, then return.
        
            context.execute_async(bindings=segment_bindings, stream_handle=stream.handle)#Asynchronously execute inference on a batch. 
            stream.synchronize()
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in segment_outputs]#Copy from the device pointer src (an int or a DeviceAllocation) to the Python buffer dest asynchronously
            stream.synchronize()
            # print(1,segment_outputs[0].host.shape)
            y = segment_outputs[0].host
        
        y1 =  y[0:np.array(context.get_binding_shape(2)).prod()].reshape(context.get_binding_shape(2))
        print('head: ',y[0:np.array(context.get_binding_shape(2)).prod()])
        print('tail: ',y[np.array(context.get_binding_shape(2)).prod()-2:])
        # y1 =  y[0:np.array(context.get_binding_shape(2)).prod()].reshape(context.get_binding_shape(2))
        
        y2 = torch.from_numpy(y1)
        print(5,'y2: ',y2.shape)
        print('value: ',y2)
        #post process for detec

        box_list = Heat2boxes(self.cfg, y2, ratio_w, ratio_h)
        box_list, heatmap = box_list.convert()
        for i in range(len(box_list)):
            box_list[i] = [[box_list[i][0], box_list[i][4]],
                           [box_list[i][1], box_list[i][5]],
                           [box_list[i][2], box_list[i][6]],
                           [box_list[i][3], box_list[i][7]]]
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        return np.array(box_list), heatmap

    def __call__(self, imgs):
        start_timer()
        if isinstance(imgs, list):
            all_boxes = []
            all_heat = []
            for i, row in enumerate(imgs):
                for j, img in enumerate(row):
                    boxes, heatmap = self.detect(img)
                    list_heat = list()
                    for bo in boxes:
                        y0, x0 = np.min(bo, axis=0)
                        y1, x1 = np.max(bo, axis=0)
                        roi = heatmap[int(x0):int(x1), int(y0):int(y1)]
                        list_heat.append(roi)
                    center = [(sum(box[:, :1]) / 4, sum(box[:, 1:2]) / 4) for box in boxes]
                    for ce, bo, he in zip(center, boxes, list_heat):
                        correct = 1
                        buffx = self.bufferx / 2
                        buffy = self.buffery / 2
                        ebuffx = self.window_shape[0] - buffx
                        ebuffy = self.window_shape[1] - buffy
                        if ce[0] < buffx or ce[0] > ebuffx or ce[1] < buffy or ce[1] > ebuffy:
                            correct = 0

                        if i == 0 and ce[1] < buffy:
                            correct = 1
                        elif i == len(imgs) - 1 and ce[1] > ebuffy:
                            correct = 1
                        elif j == 0 and ce[0] < buffx:
                            correct = 1
                        elif j == len(row) and ce[0] > ebuffx:
                            correct = 1

                        if i == 0 and j == 0 and ce[0] < buffx and ce[1] < buffy:
                            correct = 1
                        elif i == 0 and j == len(row) and ce[0] > ebuffx and ce[1] < buffy:
                            correct = 1
                        elif i == len(imgs) and j == 0 and ce[0] < buffx and ce[1] > ebuffy:
                            correct = 1
                        elif i == len(imgs) and j == len(row) and ce[0] > ebuffx and ce[1] > ebuffy:
                            correct = 1
                        if correct == 1:
                            x_plus = (self.window_shape[0] - self.bufferx) * j
                            y_plus = (self.window_shape[1] - self.buffery) * i
                            bo = [[b[0] + x_plus, b[1] + y_plus] for b in bo]
                            all_boxes.append(np.array(bo))
                            all_heat.append(he)
            result = np.array(all_boxes)
        else:
            result, all_heat = self.detect(imgs)
        end_timer_and_print('Time of Inference by TensorRT: ')
        return result, all_heat

    ## new trt
    def _load_plugins(self):
        if trt.__version__[0] < '7':
            ctypes.CDLL("./libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        assert os.path.exists(self.engine_path)
        print("Reading engine from file {}".format(self.engine_path))
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        class HostDeviceMem(object):
            def __init__(self, host_mem, device_mem):
                self.host = host_mem
                self.device = device_mem

            def __str__(self):
                return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

            def __repr__(self):
                return self.__str__()
        for binding in self.engine:
            
            dims = self.engine.get_binding_shape(binding)
            # print(dims)
            if dims[-1] == -1:
                assert(self.input_shape is not None)
                dims[-2],dims[-1] = self.input_shape
            size = trt.volume(dims) * self.engine.max_batch_size#The maximum batch size which can be used for inference.
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):#Determine whether a binding is an input binding.
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings

    # def __del__(self):
    #     """Free CUDA memories and context."""
    #     del self.cuda_outputs
    #     del self.cuda_inputs
    #     del self.stream
    