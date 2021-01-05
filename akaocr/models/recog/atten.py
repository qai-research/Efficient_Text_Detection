#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Minh Trang - Trangnm5
Created Date: Mon November 03 10:00:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain attention type of recognition model
_____________________________________________________________________________
"""

import torch.nn as nn
from pathlib import Path

from models.modules.backbones.ResNet50 import ResNet50
from models.modules.transformation import TPSSpatialTransformerNetwork
from models.modules.sequence_modeling import BidirectionalLSTM
from models.modules.prediction import Attention


class Atten(nn.Module):
    def __init__(self, config):
        super(Atten, self).__init__()
        self.config = config
        self.stages = {'Trans': config.MODEL.TRANSFORMATION, 'Feat': config.MODEL.FEATURE_EXTRACTION,
                       'Seq': config.MODEL.SEQUENCE_MODELING, 'Pred': config.MODEL.PREDICTION}

        # Transformation
        if config.MODEL.TRANSFORMATION == 'TPS':
            self.transformation = TPSSpatialTransformerNetwork(
                F=config.MODEL.NUM_FIDUCIAL, I_size=(config.MODEL.IMG_H, config.MODEL.IMG_W),
                I_r_size=(config.MODEL.IMG_H, config.MODEL.IMG_W), I_channel_num=config.MODEL.INPUT_CHANNEL,
                device=self.config.SOLVER.DEVICE)
        else:
            print('No Transformation module specified')

        # FeatureExtraction
        if config.MODEL.FEATURE_EXTRACTION == 'VGG':
            self.feature_extraction = VGGFeatureExtractor(config.MODEL.INPUT_CHANNEL, config.MODEL.OUTPUT_CHANNEL)
        elif config.MODEL.FEATURE_EXTRACTION == 'RCNN':
            self.feature_extraction = RCNNFeatureExtractor(config.MODEL.INPUT_CHANNEL, config.MODEL.OUTPUT_CHANNEL)
        elif config.MODEL.FEATURE_EXTRACTION == 'ResNet':
            self.feature_extraction = ResNet50(config.MODEL.INPUT_CHANNEL, config.MODEL.OUTPUT_CHANNEL)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.feature_extraction_output = config.MODEL.OUTPUT_CHANNEL  # int(imgH/16-1) * 512
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        # Sequence modeling
        if config.MODEL.SEQUENCE_MODELING == 'BiLSTM':
            self.sequence_modeling = nn.Sequential(
                BidirectionalLSTM(self.feature_extraction_output, config.MODEL.HIDDEN_SIZE, config.MODEL.HIDDEN_SIZE),
                BidirectionalLSTM(config.MODEL.HIDDEN_SIZE, config.MODEL.HIDDEN_SIZE, config.MODEL.HIDDEN_SIZE))
            self.sequence_modeling_output = config.MODEL.HIDDEN_SIZE
        else:
            print('No SequenceModeling module specified')
            self.sequence_modeling_output = self.feature_extraction_output

        # Prediction
        if config.MODEL.PREDICTION == 'CTC':
            self.prediction = nn.Linear(self.sequence_modeling_output, config.MODEL.NUM_CLASS)
        elif config.MODEL.PREDICTION == 'Attn':
            self.prediction = Attention(self.sequence_modeling_output, config.MODEL.HIDDEN_SIZE, config.MODEL.NUM_CLASS,
                                        device=config.SOLVER.DEVICE, beam_size=config.SOLVER.BEAM_SIZE)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, inputs, text, is_train=True):
        # Transformation stage
        if not self.stages['Trans'] == "None":
            inputs = self.transformation(inputs)

        # Feature extraction stage
        visual_feature = self.feature_extraction(inputs)
        visual_feature = self.adaptive_avg_pool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        # Sequence modeling stage
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.sequence_modeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        # Prediction stage
        if self.stages['Pred'] == 'CTC':
            prediction = self.prediction(contextual_feature.contiguous())
        else:
            prediction = self.prediction(contextual_feature.contiguous(), text, is_train,
                                         max_label_length=self.config.MODEL.MAX_LABEL_LENGTH)

        return prediction
