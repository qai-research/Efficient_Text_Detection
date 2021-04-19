#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Ngoc Nghia - Nghiann3
Created Date: Fri March 12 13:00:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contains heatmap model with efficientnet backbone for text detection
_____________________________________________________________________________
"""
import torch
from torch import nn
from models.modules.efficientdet.model import BiFPN, EfficientNet, Classifier

class HEAT_EFFICIENT(nn.Module):
    def __init__(self, num_classes=2, compound_coef=0, **kwargs):
        super(HEAT_EFFICIENT, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        # self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [7, 7, 7, 7, 7, 7, 7, 7, 8]
       
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [16, 24, 40, 112, 320],
            1: [16, 24, 40, 112, 320],
            2: [16, 24, 48, 120, 352],
            3: [24, 32, 48, 136, 384],
            4: [24, 32, 56, 160, 448],
            5: [24, 40, 64, 176, 512],
            6: [32, 40, 72, 200, 576],
            7: [32, 40, 72, 200, 576],
            8: [32, 48, 80, 224, 640],
        }

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef],
                                     num_classes=self.num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef])

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]
        p1, p2, p3, p4, p5 = self.backbone_net(inputs)
        features = (p1, p2, p3, p4, p5)

        features = self.bifpn(features)[0]
        feat = self.classifier(features)
        return feat.permute(0, 2, 3, 1), features