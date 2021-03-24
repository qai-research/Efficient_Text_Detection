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
import torch
from utils.file_utils import read_vocab
from engine.trainer.loss import MapLoss
from models.modules.converters import CTCLabelConverter, AttnLabelConverter, Averager

class CustomLoopHeat:
    def __init__(self, cfg):
        self.loss_func = MapLoss()
        self.cfg = cfg
    def loop(self, model, inputs, accuracy = None):
        images, char_label, affinity_label, mask = inputs
        images = images.to(device=self.cfg.SOLVER.DEVICE)
        char_label = char_label.to(device=self.cfg.SOLVER.DEVICE)
        affinity_label = affinity_label.to(device=self.cfg.SOLVER.DEVICE)
        mask = mask.to(device=self.cfg.SOLVER.DEVICE)
        out, _ = model(images)
        out1 = out[:, :, :, 0]
        out2 = out[:, :, :, 1]
        loss = self.loss_func(char_label, affinity_label, out1, out2, mask)
        if accuracy is not None:
            model.eval()
            acc = accuracy.run(model, inputs)
            model.train()
        else:
            acc = None
        return loss, acc

class CustomLoopAtten:
    def __init__(self, cfg):
        # self.loss = torch.nn.CrossEntropyLoss(ignore_index=0).to(cfg.SOLVER.DEVICE)
        if cfg.MODEL.PREDICTION == 'CTC':
            self.loss_func = torch.nn.CTCLoss(zero_infinity=True).to(cfg.SOLVER.DEVICE)
            self.converter = CTCLabelConverter(cfg.MODEL.VOCAB)
        elif cfg.MODEL.PREDICTION == 'Attn':
            self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=0).to(
                cfg.SOLVER.DEVICE)  # ignore [GO] token = ignore index 0
            self.converter = AttnLabelConverter(cfg.MODEL.VOCAB, device=cfg.SOLVER.DEVICE)
        else:
            raise ValueError(f"invalid model prediction type")
        self.cfg = cfg

    def loop(self, model, inputs, accuracy = None):
        images, labels = inputs
        images = images.to(self.cfg.SOLVER.DEVICE)
        text, length = self.converter.encode(labels, max_label_length=self.cfg.MODEL.MAX_LABEL_LENGTH)
        batch_size = images.size(0)

        if self.cfg.MODEL.PREDICTION == 'CTC':
            preds = model(images, text).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)

            torch.backends.cudnn.enabled = False
            loss = self.loss_func(preds, text.to(self.cfg.SOLVER.DEVICE), preds_size.to(self.cfg.SOLVER.DEVICE),
                             length.to(self.cfg.SOLVER.DEVICE))
            torch.backends.cudnn.enabled = True

        elif self.cfg.MODEL.PREDICTION == 'Attn':
            preds = model(images, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            loss = self.loss_func(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        else:
            raise ValueError(f"invalid model prediction type")
        if accuracy is not None:
            model.eval()
            acc = accuracy.run(model, inputs)
            model.train()
        else:
            acc = None
        return loss, acc