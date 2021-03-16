#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Minh Trang - Trangnm5
Project : AkaOCR core
_____________________________________________________________________________

This file contains evaluation method for recog model
_____________________________________________________________________________
"""

from models.modules.converters import Averager
import time
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance
import torch 

def validation(model, criterion, evaluation_loader, converter, config):
    """ validation or evaluation """
    n_correct = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    norm_ED = 0

    num_batches = 0
    for i in range(len(evaluation_loader.list_iterator)):
        num_batches += len(evaluation_loader.list_iterator[i])
    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(config.SOLVER.DEVICE)
        length_for_pred = torch.IntTensor([int(config.MODEL.MAX_LABEL_LENGTH)] * batch_size).to(config.SOLVER.DEVICE)
        text_for_pred = torch.LongTensor(batch_size, int(config.MODEL.MAX_LABEL_LENGTH) + 1).fill_(0).to(config.SOLVER.DEVICE)

        text_for_loss, length_for_loss = converter.encode(labels, max_label_length=int(config.MODEL.MAX_LABEL_LENGTH))

        start_time = time.time()
        if 'CTC' in config.MODEL.PREDICTION:
            preds = model(image, text_for_pred)
            forward_time = time.time() - start_time

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)

            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in config.MODEL.PREDICTION:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            if not config.MODEL.SENSITIVE:  # case insensitive
                pred = pred.lower()
                gt = gt.lower()
            # print(config["character"])
            if config.SOLVER.DATA_FILTERING:
                pred = ''.join([c for c in list(pred) if c in config["character"]])
                gt = ''.join([c for c in list(gt) if c in config["character"]])

            if pred == gt:
                n_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)

        if i == num_batches-1:  # stop after a number of batches is reached
            break
    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data
