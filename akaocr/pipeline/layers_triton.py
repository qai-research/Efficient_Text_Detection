#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Huu Kim - Kimnh3
Created Date: Jun 12 16:22:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contains layers for Triton server inferencing
_____________________________________________________________________________
"""
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import cv2
import numpy as np
import shutil
import os
import re
from PIL import Image
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
from utils.runtime import start_timer, end_timer_and_print


logger = initial_logger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# triton

import argparse
from functools import partial
import sys
from attrdict import AttrDict

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 2:
        raise Exception("expecting 2 output, got {}".format(
            len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    input_batch_dim = (model_config.max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata.name,
                   len(input_metadata.shape)))

    if type(input_config.format) == str:
        FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
        input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

    # if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
    #     (input_config.format != mc.ModelInput.FORMAT_NHWC)):
    #     raise Exception("unexpected input format " +
    #                     mc.ModelInput.Format.Name(input_config.format) +
    #                     ", expecting " +
    #                     mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
    #                     " or " +
    #                     mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (model_config.max_batch_size, input_metadata.name,
            output_metadata.name, c, h, w, input_config.format,
            input_metadata.datatype)


def preprocess(img, format, dtype, c, h, w, scaling, protocol):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    # np.set_printoptions(threshold='nan')

    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')

    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    npdtype = triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)

    if scaling == 'INCEPTION':
        scaled = (typed / 127.5) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed

    # Swap to CHW if necessary
    if format == mc.ModelInput.FORMAT_NCHW:
        ordered = np.transpose(scaled, (2, 0, 1))
    else:
        ordered = scaled

    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def postprocess(results, output_name, batch_size, batching):
    """
    Post-process results to show classifications.
    """
    lst_out = list()
    output_array = results.as_numpy(output_name)
    # print(3,output_array.shape)
    output_array = torch.from_numpy(output_array)#.permute(0, 2, 1, 3)
    # print(4,output_array.shape)
    # print(4, output_array)
    return output_array
    # if len(output_array) != batch_size:
    #     raise Exception("expected {} results, got {}".format(
    #         batch_size, len(output_array)))
    # # Include special handling for non-batching models
    # for results in output_array:
    #     if not batching:
    #         results = [results]
    #     for result in results:
    #         # print(4,type(result))
    #         lst_out.append(result)
    # return lst_out
            
            # if output_array.dtype.type == np.object_:
            #     cls = "".join(chr(x) for x in result).split(':')
            # else:
            #     cls = result.split(':')
            # print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))


def requestGenerator(batched_image_data, input_name, output_name, dtype, FLAGS):
    protocol = FLAGS.protocol.lower()

    if protocol == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [
        client.InferRequestedOutput(output_name)#, class_count=FLAGS.classes)
    ]

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version


def convert_http_metadata_config(_metadata, _config):
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config


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
                 window=(1280, 800), bufferx=50, buffery=20, preprocess=None, postprocess=None):
        super().__init__()
        if not config:
            _, detec_model_config = experiment_loader(name=model_name, type='detec',
                                                                       data_path=data_path)
            config = detec_model_config
            # model_path = detec_model_path
        # logger.info(f"load model from : {model_path}")

        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file(config)

        if self.cfg.MODEL.NAME == "CRAFT":
            model = HEAT()
        elif self.cfg.MODEL.NAME == "RESNET":
            model = HEAT_RESNET()
        elif self.cfg.MODEL.NAME == "EFFICIENT":
            model = HEAT_EFFICIENT()
            
        self.window_shape = window
        self.bufferx = bufferx
        self.buffery = buffery
        # self.detector = model

    def detect(self, img, FLAGS):
        img_resized, target_ratio = ImageProc.resize_aspect_ratio(
            img, self.cfg.MODEL.MAX_SIZE, interpolation=cv2.INTER_LINEAR
        )
        ratio_h = ratio_w = 1 / target_ratio
        img_resized = ImageProc.normalize_mean_variance(img_resized)
        img_resized = torch.from_numpy(img_resized).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        # img_resized = (img_resized.unsqueeze(0))  # [c, h, w] to [b, c, h, w] # unecessary, cause triton add max_batch_size (from config) ahead of chw->bchw
        # img_resized = img_resized.to(device)

        # y,_ = self.detector(img_resized)
        # replace above with infer request to triton
        start_timer() 

        # triton
        if FLAGS.streaming and FLAGS.protocol.lower() != "grpc":
            raise Exception("Streaming is only allowed with gRPC protocol")

        try:
            if FLAGS.protocol.lower() == "grpc":
                # Create gRPC client for communicating with the server
                triton_client = grpcclient.InferenceServerClient(
                    url=FLAGS.url, verbose=FLAGS.verbose)
            else:
                # Specify large enough concurrency to handle the
                # the number of requests.
                concurrency = 20 if FLAGS.async_set else 1
                triton_client = httpclient.InferenceServerClient(
                    url=FLAGS.url, verbose=FLAGS.verbose, concurrency=concurrency)
        except Exception as e:
            print("client creation failed: " + str(e))
            sys.exit(1)

        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        try:
            model_metadata = triton_client.get_model_metadata(
                model_name=FLAGS.model_name, model_version=FLAGS.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the metadata: " + str(e))
            sys.exit(1)
        # print(1,model_metadata)
        try:
            model_config = triton_client.get_model_config(
                model_name=FLAGS.model_name, model_version=FLAGS.model_version)
        except InferenceServerException as e:
            print("failed to retrieve the config: " + str(e))
            sys.exit(1)

        if FLAGS.protocol.lower() == "grpc":
            model_config = model_config.config
        else:
            model_metadata, model_config = convert_http_metadata_config(
                model_metadata, model_config)

        max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(
            model_metadata, model_config)
        # print(0,'\n max_batch_size: ',max_batch_size, '\n input_name: ',input_name, '\n output_name: ',output_name, 
            # '\n c-h-w: ',c, h, w, 
            # '\n format: ',format, 
            # '\n dtype: ', dtype)
        # filenames = []
        # if os.path.isdir(FLAGS.image_filename):
        #     filenames = [
        #         os.path.join(FLAGS.image_filename, f)
        #         for f in os.listdir(FLAGS.image_filename)
        #         if os.path.isfile(os.path.join(FLAGS.image_filename, f))
        #     ]
        # else:
        #     filenames = [
        #         FLAGS.image_filename,
        #     ]

        # filenames.sort()

        # Preprocess the images into input data according to model
        # requirements
        image_data = []
        img_resized = np.array(img_resized)

        image_data.append(img_resized)
        # for filename in filenames:
        #     img = Image.open(filename)
        #     image_data.append(
        #         preprocess(img, format, dtype, c, h, w, FLAGS.scaling,
        #                 FLAGS.protocol.lower()))

        # Send requests of FLAGS.batch_size images. If the number of
        # images isn't an exact multiple of FLAGS.batch_size then just
        # start over with the first images until the batch is filled.
        requests = []
        responses = []
        result_filenames = []
        request_ids = []
        image_idx = 0
        last_request = False
        user_data = UserData()

        # Holds the handles to the ongoing HTTP async requests.
        async_requests = []

        sent_count = 0

        if FLAGS.streaming:
            triton_client.start_stream(partial(completion_callback, user_data))

        while not last_request:
            # input_filenames = []
            repeated_image_data = []
            for idx in range(FLAGS.batch_size):
                # input_filenames.append(filenames[image_idx])
                repeated_image_data.append(image_data[image_idx])
                image_idx = (image_idx + 1) % len(image_data)
                if image_idx == 0:
                    last_request = True
            if max_batch_size > 0:
                batched_image_data = np.stack(repeated_image_data, axis=0)
            else:
                batched_image_data = repeated_image_data[0]
            # batched_image_data = repeated_image_data[0]
            # print(batched_image_data.shape)

            # Send request
            try:
                for inputs, outputs, model_name, model_version in requestGenerator(
                        batched_image_data, input_name, output_name, dtype, FLAGS):
                    sent_count += 1
                    if FLAGS.streaming:
                        triton_client.async_stream_infer(
                            FLAGS.model_name,
                            inputs,
                            request_id=str(sent_count),
                            model_version=FLAGS.model_version,
                            outputs=outputs)
                    elif FLAGS.async_set:
                        if FLAGS.protocol.lower() == "grpc":
                            triton_client.async_infer(
                                FLAGS.model_name,
                                inputs,
                                partial(completion_callback, user_data),
                                request_id=str(sent_count),
                                model_version=FLAGS.model_version,
                                outputs=outputs)
                        else:
                            async_requests.append(
                                triton_client.async_infer(
                                    FLAGS.model_name,
                                    inputs,
                                    request_id=str(sent_count),
                                    model_version=FLAGS.model_version,
                                    outputs=outputs))
                    else:
                        responses.append(
                            triton_client.infer(FLAGS.model_name,
                                                inputs,
                                                request_id=str(sent_count),
                                                model_version=FLAGS.model_version,
                                                outputs=outputs))
                    

            except InferenceServerException as e:
                print("inference failed: " + str(e))
                if FLAGS.streaming:
                    triton_client.stop_stream()
                sys.exit(1)

        if FLAGS.streaming:
            triton_client.stop_stream()

        if FLAGS.protocol.lower() == "grpc":
            if FLAGS.streaming or FLAGS.async_set:
                processed_count = 0
                while processed_count < sent_count:
                    (results, error) = user_data._completed_requests.get()
                    processed_count += 1
                    if error is not None:
                        print("inference failed: " + str(error))
                        sys.exit(1)
                    responses.append(results)
        else:
            if FLAGS.async_set:
                # Collect results from the ongoing async requests
                # for HTTP Async requests.
                for async_request in async_requests:
                    responses.append(async_request.get_result())
        # print(responses)
        for response in responses:
            if FLAGS.protocol.lower() == "grpc":
                this_id = response.get_response().id
            else:
                this_id = response.get_response()["id"]
            print("Request {}, batch size {}".format(this_id, FLAGS.batch_size))
            # print(response)
            y=postprocess(response, output_name, FLAGS.batch_size, max_batch_size > 0)
        
            # print(5,(y))
        end_timer_and_print("Inference using Triton server: ")
        print("PASS")
        # return response

        box_list = Heat2boxes(self.cfg, y, ratio_w, ratio_h)
        box_list, heatmap = box_list.convert()
        for i in range(len(box_list)):
            box_list[i] = [[box_list[i][0], box_list[i][4]],
                            [box_list[i][1], box_list[i][5]],
                            [box_list[i][2], box_list[i][6]],
                            [box_list[i][3], box_list[i][7]]]
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        return np.array(box_list), heatmap

    def __call__(self, imgs, FLAGS):
        if isinstance(imgs, list):
            all_boxes = []
            all_heat = []
            for i, row in enumerate(imgs):
                for j, img in enumerate(row):
                    boxes, heatmap = self.detect(img, FLAGS)
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
            result, all_heat = self.detect(imgs, FLAGS)


            
        return result, all_heat


class Recoglayer():
    """
    A basic pipeline for performing recognition.

    Attributes
    ----------
    preprocess : callable
        the preprocess function whose input is a image
    recognizer : type of recognizer
        Recognizer subclass

    Methods
    -------
    __call__(image, boxes)
        image : crop image to recognize or big image with bounding boxes to crop and return list of recognized results
    """
    def __init__(self, config=None, model_path=None, model_name='test', data_path='../../data',
                 preprocess=None, postprocess=None, lang='eng+jpn'):
        super().__init__()

        self.preprocess = preprocess
        self.postprocess = postprocess
      
        if not config:
            recog_model_path, recog_model_config = experiment_loader(name=model_name, type='recog',
                                                                       data_path=data_path)
            config = recog_model_config
            model_path = recog_model_path
        logger.info(f"load model from : {model_path}")
        
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file(config)
        if self.cfg.MODEL.VOCAB is not None:
            self.cfg.MODEL.VOCAB = os.path.join(data_path, "vocabs", self.cfg.MODEL.VOCAB)
        self.cfg.SOLVER.DEVICE = str(device)
        if self.cfg.MODEL.VOCAB:  # vocabulary is given
            self.cfg.MODEL.VOCAB = read_vocab(self.cfg.MODEL.VOCAB)
            self.cfg["character"] = self.cfg.MODEL.VOCAB
        else:  # use character list instead
            self.cfg["character"] = list(self.cfg["character"])
        if self.cfg.SOLVER.UNKNOWN:
            self.cfg["character"].append(self.cfg.SOLVER.UNKNOWN)
        self.cfg["character"].sort()
        if 'CTC' in self.cfg.MODEL.PREDICTION:
            self.converter = CTCLabelConverter(self.cfg["character"])
        else:
            self.converter = AttnLabelConverter(self.cfg["character"], device=self.cfg.SOLVER.DEVICE)
        self.cfg.MODEL.NUM_CLASS = len(self.converter.character)
       
        model = Atten(self.cfg)

        checkpointer = ModelCheckpointer(model)
        #strict_mode=False (default) to ignore different at layers/size between 2 models, otherwise, must be identical and raise error.
        checkpointer.resume_or_load(model_path, strict_mode=True)
        model = model.to(device)

        self.recognizer = model
        self.max_h = self.cfg.MODEL.IMG_H
        self.max_w = self.cfg.MODEL.IMG_W
        self.pad = self.cfg.MODEL.PAD
        self.max_label_length = self.cfg.MODEL.MAX_LABEL_LENGTH
        
    def _remove_unknown(self, text):
        text = re.sub(f'[{self.cfg.SOLVER.UNKNOWN}]+', "", text)
        return text

    def recog(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if len(img.shape) == 2:  # (h, w)
            img = np.expand_dims(img, axis=-1)
        else:
            img = img
        if self.cfg.MODEL.INPUT_CHANNEL == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:  # grayscale
            img = Image.fromarray(img[:, :, 0])
        align_collate = AlignCollate(img_h=self.max_h, img_w=self.max_w, keep_ratio_with_pad=self.pad)
        img_tensor = align_collate(img)
        with torch.no_grad():
            image = img_tensor.to(device)
            length_for_pred = torch.IntTensor([self.max_label_length]).to(device)
            text_for_pred = torch.LongTensor(1, self.max_label_length + 1).fill_(0).to(device)
            if 'Attn' in self.cfg.MODEL.PREDICTION:
                preds = self.recognizer(image, text_for_pred, is_train=False)
                # select max probability (greedy decoding) then decode index to character
                if self.intercept:
                    preds = self.intercept.intercept_vocab(preds)
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                pred_eos = preds_str[0].find('[s]')
                pred = preds_str[0][:pred_eos]  # prune after "end of sentence" token ([s])
                preds_max_prob = preds_max_prob[0][:pred_eos]
                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = preds_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0
            else: 
                preds = self.recognizer(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)])
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = self.converter.decode(preds_index.data, preds_size.data)

                pred = preds_str[0]
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)

                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = preds_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])

            text = self._remove_unknown(pred)
            return text, confidence_score

    def __call__(self, img, boxes=None, output=None, seperator=None, subvocab=None):
        if subvocab:
            self.intercept = InterceptVocab(subvocab, self.converter)
        else:
            self.intercept = None
        if output:
            if not os.path.exists(output):
                os.makedirs(output)
        if boxes is None:
            text, score = self.recog(img)
            if output:
                cv2.imwrite(os.path.join(output, text + '.jpg'), img)
            return text
        else:
            recog_result_list = list()
            confidence_score_list = list()
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32)
                if len(poly.shape) == 1:
                    x0, y0 = poly[0], poly[1]
                    x1, y1 = poly[2], poly[3]
                else:
                    x0, y0 = np.min(poly, axis=0)
                    x1, y1 = np.max(poly, axis=0)
                roi = img[y0:y1, x0:x1]
                try:
                    if not seperator:
                        text, score = self.recog(roi)
                    elif seperator == 'logic':
                        _, l_roi = logic_seperator(roi)
                        text = ''
                        for ro in l_roi:
                            te, score = self.recog(ro)
                            text = text + te
                except:
                    text = ''
                    score = -1
                    print('cant recog box ', roi.shape)

                if output:
                    cv2.imwrite(os.path.join(output, 'result' + str(i) + '_' + text + '.jpg'), roi)
                recog_result_list.append(text)
                confidence_score_list.append(score)
            return recog_result_list, confidence_score_list