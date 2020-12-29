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
        self.stages = {'Trans': config["transformation"], 'Feat': config["feature_extraction"],
                       'Seq': config["sequence_modeling"], 'Pred': config["prediction"]}

        # Transformation
        if config["transformation"] == 'TPS':
            self.transformation = TPSSpatialTransformerNetwork(
                F=config["num_fiducial"], I_size=(config["img_h"], config["img_w"]),
                I_r_size=(config["img_h"], config["img_w"]), I_channel_num=config["input_channel"],
                device=self.config["device"])
        else:
            print('No Transformation module specified')

        # FeatureExtraction
        if config["feature_extraction"] == 'VGG':
            self.feature_extraction = VGGFeatureExtractor(config["input_channel"], config["output_channel"])
        elif config["feature_extraction"] == 'RCNN':
            self.feature_extraction = RCNNFeatureExtractor(config["input_channel"], config["output_channel"])
        elif config["feature_extraction"] == 'ResNet':
            self.feature_extraction = ResNet50(config["input_channel"], config["output_channel"])
        else:
            raise Exception('No FeatureExtraction module specified')
        self.feature_extraction_output = config["output_channel"]  # int(imgH/16-1) * 512
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        # Sequence modeling
        if config["sequence_modeling"] == 'BiLSTM':
            self.sequence_modeling = nn.Sequential(
                BidirectionalLSTM(self.feature_extraction_output, config["hidden_size"], config["hidden_size"]),
                BidirectionalLSTM(config["hidden_size"], config["hidden_size"], config["hidden_size"]))
            self.sequence_modeling_output = config["hidden_size"]
        else:
            print('No SequenceModeling module specified')
            self.sequence_modeling_output = self.feature_extraction_output

        # Prediction
        if config["prediction"] == 'CTC':
            self.prediction = nn.Linear(self.sequence_modeling_output, config["num_class"])
        elif config["prediction"] == 'Attn':
            self.prediction = Attention(self.sequence_modeling_output, config["hidden_size"], config["num_class"],
                                        device=config["device"], beam_size=config["beam_size"])
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
                                         max_label_length=self.config["max_label_length"])

        return prediction
