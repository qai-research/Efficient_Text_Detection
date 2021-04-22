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
    def __init__(self, cfg):
        super(Atten, self).__init__()
        self.cfg = cfg
        self.stages = {'Trans': cfg.MODEL.TRANSFORMATION, 'Feat': cfg.MODEL.FEATURE_EXTRACTION,
                       'Seq': cfg.MODEL.SEQUENCE_MODELING, 'Pred': cfg.MODEL.PREDICTION}

        # Transformation
        if cfg.MODEL.TRANSFORMATION == 'TPS':
            self.transformation = TPSSpatialTransformerNetwork(
                F=cfg.MODEL.NUM_FIDUCIAL, I_size=(cfg.MODEL.IMG_H, cfg.MODEL.IMG_W),
                I_r_size=(cfg.MODEL.IMG_H, cfg.MODEL.IMG_W), I_channel_num=cfg.MODEL.INPUT_CHANNEL,
                device=self.cfg.SOLVER.DEVICE)
        else:
            print('No Transformation module specified')

        # FeatureExtraction
        if cfg.MODEL.FEATURE_EXTRACTION == 'VGG':
            self.feature_extraction = VGGFeatureExtractor(cfg.MODEL.INPUT_CHANNEL, cfg.MODEL.OUTPUT_CHANNEL)
        elif cfg.MODEL.FEATURE_EXTRACTION == 'RCNN':
            self.feature_extraction = RCNNFeatureExtractor(cfg.MODEL.INPUT_CHANNEL, cfg.MODEL.OUTPUT_CHANNEL)
        elif cfg.MODEL.FEATURE_EXTRACTION == 'ResNet':
            self.feature_extraction = ResNet50(cfg.MODEL.INPUT_CHANNEL, cfg.MODEL.OUTPUT_CHANNEL)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.feature_extraction_output = cfg.MODEL.OUTPUT_CHANNEL  # int(imgH/16-1) * 512
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        # Sequence modeling
        if cfg.MODEL.SEQUENCE_MODELING == 'BiLSTM':
            self.sequence_modeling = nn.Sequential(
                BidirectionalLSTM(self.feature_extraction_output, cfg.MODEL.HIDDEN_SIZE, cfg.MODEL.HIDDEN_SIZE),
                BidirectionalLSTM(cfg.MODEL.HIDDEN_SIZE, cfg.MODEL.HIDDEN_SIZE, cfg.MODEL.HIDDEN_SIZE))
            self.sequence_modeling_output = cfg.MODEL.HIDDEN_SIZE
        else:
            print('No SequenceModeling module specified')
            self.sequence_modeling_output = self.feature_extraction_output

        # Prediction
        if cfg.MODEL.PREDICTION == 'CTC':
            self.prediction = nn.Linear(self.sequence_modeling_output, cfg.MODEL.NUM_CLASS)
        elif cfg.MODEL.PREDICTION == 'Attn':
            self.prediction = Attention(self.sequence_modeling_output, cfg.MODEL.HIDDEN_SIZE, cfg.MODEL.NUM_CLASS,
                                        device=cfg.SOLVER.DEVICE, beam_size=cfg.SOLVER.BEAM_SIZE)
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
                                         max_label_length=self.cfg.MODEL.MAX_LABEL_LENGTH)

        return prediction
