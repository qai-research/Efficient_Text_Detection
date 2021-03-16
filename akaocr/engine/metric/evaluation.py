#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Ngoc Nghia - Nghiann3
Created Date: Fri March 12 13:00:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contains evaluation methods for both detec and recog model
_____________________________________________________________________________
"""

import sys
sys.path.append("../")

from engine.metric import tedeval
from engine.infer.heat2boxes import Heat2boxes
from pre.image import ImageProc
import torch
import cv2
from models.modules.converters import AttnLabelConverter, Averager
import time
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance

class Evaluation:
    """This module contains evaluation methods for detec and recog models"""

    def __init__(self, cfg, model, test_loader, num_samples = None):
        """
        Args:
            model: model for evaluation
            test_loader: data for evaluation
            num_samples: number of sample will be evaluated
        """

        self.cfg = cfg
        self.model = model
        self.test_loader = test_loader
        if self.cfg._BASE_.MODEL_TYPE == "HEAT_BASE":
            self.max_size = self.cfg.MODEL.MAX_SIZE
            if num_samples is None:
                self.num_samples = self.test_loader.get_length()
            else:
                self.num_samples = num_samples
        elif self.cfg._BASE_.MODEL_TYPE == "ATTEN_BASE":
            if 'CTC' in self.cfg.MODEL.PREDICTION:
                self.criterion = torch.nn.CTCLoss(zero_infinity=True).to(self.cfg.SOLVER.DEVICE)
                self.converter = CTCLabelConverter(self.cfg["character"])
            else:
                self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(self.cfg.SOLVER.DEVICE)  # ignore [GO] token = ignore index 0
                self.converter = AttnLabelConverter(self.cfg["character"], device=self.cfg.SOLVER.DEVICE)
        else:
            raise Exception("MODEL_TYPE is not supported, MODEL_TYPE is HEAT_BASE or ATTEN_BASE")

    """Evaluate detec model"""
    def detec_evaluation(self):
        recall = 0
        precision = 0
        hmean = 0
        AP = 0
        for i in range(1, self.num_samples+1):
            img, label = self.test_loader.get_item(i)
            gt_box = list()
            words = list()
            for j in range(len(label['words'])):
                words.append(label['words'][j]['text'])
                x1 = label['words'][j]['x1']
                x2 = label['words'][j]['x2']
                x3 = label['words'][j]['x3']
                x4 = label['words'][j]['x4']
                y1 = label['words'][j]['y1']
                y2 = label['words'][j]['y2']
                y3 = label['words'][j]['y3']
                y4 = label['words'][j]['y4']
                box = [x1, y1, x2, y2, x3, y3, x4, y4]
                gt_box.append(box)
    
            img, target_ratio = ImageProc.resize_aspect_ratio(
                img, self.max_size, interpolation=cv2.INTER_LINEAR
            )
            ratio_h = ratio_w = 1 / target_ratio
            img = ImageProc.normalize_mean_variance(img)
            img = torch.from_numpy(img).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
            img = (img.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
            y,_ = self.model(img)
            del img     # delete variable img to save memory
            box_list = Heat2boxes(self.cfg, y, ratio_w, ratio_h)
            del y       # delete variable y to save memory
            box_list,_ = box_list.convert(evaluation=True)

            confidence_point = list()
            for j in range(len(box_list)):
                confidence_point.append(0.0)
            detec_eval = tedeval.Evaluate(box_list, gt_box, words, confidence_point)
            resdict = detec_eval.do_eval()
            recall += resdict['method']['recall']
            precision += resdict['method']['precision']
            hmean += resdict['method']['hmean']
            AP += resdict['method']['AP']
        print('recall:', recall/self.num_samples, 'precision:', precision/self.num_samples)
        print('hmean:', hmean/self.num_samples, 'AP:', AP/self.num_samples)
    
    """Evaluate recog model"""
    def recog_evaluation(self):
        self.model.eval()
        with torch.no_grad():
            _, accuracy_by_best_model, _, _, _, _, _, _ = self.validation(self.model, self.criterion, self.test_loader, self.converter, self.cfg)
            print(f'Accuracy: {accuracy_by_best_model:0.3f}')

    def validation(self, model, criterion, evaluation_loader, converter, config):
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
            image = image_tensors.to(self.cfg.SOLVER.DEVICE)
            length_for_pred = torch.IntTensor([int(config.MODEL.MAX_LABEL_LENGTH)] * batch_size).to(self.cfg.SOLVER.DEVICE)
            text_for_pred = torch.LongTensor(batch_size, int(config.MODEL.MAX_LABEL_LENGTH) + 1).fill_(0).to(self.cfg.SOLVER.DEVICE)

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

    def do_eval(self):
        if self.cfg._BASE_.MODEL_TYPE == "HEAT_BASE":
            self.detec_evaluation()
        elif self.cfg._BASE_.MODEL_TYPE == "ATTEN_BASE":
            self.recog_evaluation()