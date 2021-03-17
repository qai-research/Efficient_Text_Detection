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

from engine.metric import tedeval, recog_eval
from engine.infer.heat2boxes import Heat2boxes
from pre.image import ImageProc
import torch
import cv2
from models.modules.converters import AttnLabelConverter


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
        best_accuracy = -1
        best_norm_ED = -1
        self.model.eval()
        with torch.no_grad():
            valid_loss, current_accuracy, current_norm_ED, preds, \
            confidence_score, labels, infer_time, length_of_data = recog_eval.validation(self.model, self.criterion, self.test_loader, self.converter, self.cfg)
            current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
            if current_norm_ED > best_norm_ED:
                best_norm_ED = current_norm_ED
            best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'
            loss_model_log = f'\n{current_model_log}\n{best_model_log}'
            print(loss_model_log)

            # show some predicted results
            dashed_line = '-' * 80
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                if 'Attn' in self.cfg.MODEL.PREDICTION:
                    gt = gt[:gt.find('[s]')]
                    pred = pred[:pred.find('[s]')]
                predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)
        self.model.train()
    def do_eval(self):
        if self.cfg._BASE_.MODEL_TYPE == "HEAT_BASE":
            self.detec_evaluation()
        elif self.cfg._BASE_.MODEL_TYPE == "ATTEN_BASE":
            self.recog_evaluation()