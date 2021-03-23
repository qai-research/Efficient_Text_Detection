#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Ngoc Nghia - Nghiann3
Created Date: Fri March 12 13:00:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file compute accuracy of while training
_____________________________________________________________________________
"""

import torch
from models.modules.converters import CTCLabelConverter, AttnLabelConverter

class RecogAccuracy():
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.MODEL.PREDICTION == 'CTC':
            self.converter = CTCLabelConverter(cfg.MODEL.VOCAB)
        elif cfg.MODEL.PREDICTION == 'Attn':
            self.converter = AttnLabelConverter(cfg.MODEL.VOCAB, device=cfg.SOLVER.DEVICE)
        else:
            raise ValueError(f"invalid model prediction type")
       
    def run(self, model, inputs):
        best_accuracy = -1
        n_correct = 0
        model.eval()
        with torch.no_grad():
            image_tensors, labels = inputs
            batch_size = image_tensors.size(0)
            image = image_tensors.to(self.cfg.SOLVER.DEVICE)
            length_for_pred = torch.IntTensor([int(self.cfg.MODEL.MAX_LABEL_LENGTH)] * batch_size).to(self.cfg.SOLVER.DEVICE)
            text_for_pred = torch.LongTensor(batch_size, int(self.cfg.MODEL.MAX_LABEL_LENGTH) + 1).fill_(0).to(self.cfg.SOLVER.DEVICE)

            text_for_loss, length_for_loss = self.converter.encode(labels, max_label_length=int(self.cfg.MODEL.MAX_LABEL_LENGTH))

            if 'CTC' in self.cfg.MODEL.PREDICTION:
                preds = model(image, text_for_pred)
                # Calculate evaluation loss for CTC deocder.
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                # permute 'preds' to use CTCloss format

                # Select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = self.converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)
                preds = preds[:, :text_for_loss.shape[1] - 1, :]
                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)
                labels = self.converter.decode(text_for_loss[:, 1:], length_for_loss)

            for gt, pred in zip(labels, preds_str):
                if 'Attn' in self.cfg.MODEL.PREDICTION:
                    gt = gt[:gt.find('[s]')]
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

                if not self.cfg.MODEL.SENSITIVE:  # case insensitive
                    pred = pred.lower()
                    gt = gt.lower()
                # print(cfg["character"])
                if self.cfg.SOLVER.DATA_FILTERING:
                    pred = ''.join([c for c in list(pred) if c in self.cfg["character"]])
                    gt = ''.join([c for c in list(gt) if c in self.cfg["character"]])

                if pred == gt:
                    n_correct += 1

        accuracy = n_correct / batch_size * 100
        model.train()
        return accuracy
    
class DetecAccuracy():
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, model, inputs):
        return None