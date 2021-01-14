# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Thu January 07 10:57:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

Custom loop for models
_____________________________________________________________________________
"""

from engine.trainer.loss import MapLoss


class CustomLoopHeat:
    def __init__(self):
        self.loss = MapLoss()

    def loop(self, model, inputs):
        images, char_label, affinity_label, mask = inputs
        out = model(images)
        out1 = out[:, :, :, 0]
        out2 = out[:, :, :, 1]
        loss = self.loss(char_label, affinity_label, out1, out2, mask)
        return loss





