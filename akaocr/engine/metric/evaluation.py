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

from engine.metric import tedeval, recog_eval
from engine.infer.heat2boxes import Heat2boxes
from pre.image import ImageProc
import torch
import cv2
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""Evaluate detec model"""
class DetecEvaluation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_size = self.cfg.MODEL.MAX_SIZE
        self.num_samples = self.cfg.SOLVER.NUM_SAMPLES
        self.finish = False
   
    def run(self, model, test_loader, metric=None):
        model.eval()
        recall = 0
        precision = 0
        hmean = 0
        AP = 0
        data_length = 0
        num = 0
        for data in test_loader:
            data_length += data.get_length()
        if self.num_samples > data_length:
            self.num_samples = data_length
        for data in test_loader:
            for i in range(1, data.get_length()+1):
                img, label = data.get_item(i)
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
                img = img.to(device)
                y,_ = model(img)
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
                num += 1
                if num ==self.num_samples:
                    self.finish = True
                    break
            if self.finish:
                break

        mess =  'recall:', recall/self.num_samples, 'precision:', precision/self.num_samples, 'hmean:', hmean/self.num_samples
        recall = recall/self.num_samples
        precision = precision/self.num_samples
        hmean = hmean/self.num_samples
        if metric is None: #log metric
            metric = [recall, precision, hmean, 0]
        elif metric[0] < recall:
            metric[0] = recall
            metric[3] = 0   #reset count if update best F1
        else:
            metric[3] += 1  #increase count if F1 not improve
            if metric[3] > self.cfg.SOLVER.EARLY_STOP_AFTER:
                sys.exit()
        model.train()
        return metric, mess
    
"""Evaluate recog model"""
class RecogEvaluation():
    def __init__(self, cfg, criterion, converter):
        """
        Args:
            model: model for evaluation
            test_loader: data for evaluation
            num_samples: number of sample will be evaluated
        """

        self.cfg = cfg
        self.num_samples = self.cfg.SOLVER.NUM_SAMPLES
        self.criterion = criterion
        self.converter = converter

    def run(self, model, test_loader, metric=None):
        best_accuracy = -1
        best_norm_ED = -1
        data_length = 0
        for i in range(len(test_loader.list_iterator)):
            data_length += len(test_loader.list_iterator[i])
        if self.num_samples > data_length:
            self.num_samples = data_length 
        model.eval()
        with torch.no_grad():
            valid_loss, current_accuracy, current_norm_ED, preds, \
            confidence_score, labels, infer_time, length_of_data = recog_eval.validation(model, self.criterion, test_loader, self.converter, self.cfg)
            
            #log in metric
            if metric is None:
                metric = [current_accuracy, current_norm_ED, 0]
            elif metric[0] < current_accuracy: #check accuracy only, not norm
                metric[0] = current_accuracy
                metric[2] = 0
            else:
                metric[2] += 1
                if metric[2] > self.cfg.SOLVER.EARLY_STOP_AFTER:
                    sys.exit()
            model.train()
            
            if current_norm_ED > metric[1]:
                metric[1] = current_norm_ED

            current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'
            best_model_log = f'{"Best_accuracy":17s}: {metric[0]:0.3f}, {"Best_norm_ED":17s}: {metric[1]:0.2f}'
            loss_model_log = f'\n{current_model_log}\n{best_model_log}'
            mess =  f'{"Best_accuracy":17s}: {current_accuracy:0.3f}, {"Best_norm_ED":17s}: {current_norm_ED:0.2f}, {loss_model_log}'
            return metric, mess

            # if current_accuracy > best_accuracy:
            #     best_accuracy = current_accuracy
            
            
            # print(loss_model_log)

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
        model.train()