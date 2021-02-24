# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Thu January 28 14:39:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contain method convert heatmap to bounding boxes.
_____________________________________________________________________________
"""

from engine.utils import heat_utils
import numpy as np
from functools import cmp_to_key

class Heat2boxes:
    def __init__(self, cfg, y, ratio_w, ratio_h):
        self.cfg = cfg
        self.y = y
        self.ratio_w = ratio_w
        self.ratio_h = ratio_h

    def compare_rect(self, first_rect, second_rect):
        fx, fy, fxi, fyi = first_rect
        sx, sy, sxi, syi = second_rect
        if fxi <= sx:
            return -1  # completely on above
        elif sxi <= fx:
            return 1  # completely on below
        elif fyi <= fy:
            return -1  # completely on left
        elif sxi <= sx:
            return 1  # completely on right
        elif fy != sy:
            return -1 if fy < sy else 1  # starts on more left
        elif fx != sx:
            return -1 if fx < sx else 1  # top most when starts equally
        elif fyi != syi:
            return -1 if fyi < syi else 1  # have least width
        elif fxi != sxi:
            return -1 if fxi < sxi else 1  # have least height
        else:
            return 0  # same

    def convert(self, evaluation = False):
        score_text = self.y[0, :, :, 0].cpu().data.numpy()
        score_link = self.y[0, :, :, 1].cpu().data.numpy()
        
        # Post-processing
        text_threshold = self.cfg.INFERENCE.TEXT_THRESHOLD
        link_threshold = self.cfg.INFERENCE.LINK_THRESHOLD
        low_text_score = self.cfg.INFERENCE.LOW_TEXT_SCORE
        boxes, polys = heat_utils.get_det_boxes(score_text, score_link, text_threshold, link_threshold,
                                                        low_text_score, False)

        # coordinate adjustment
        boxes = heat_utils.adjust_result_coordinates(boxes, self.ratio_w, self.ratio_h)
        polys = heat_utils.adjust_result_coordinates(polys, self.ratio_w, self.ratio_h)
        rects = list()

        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]
            poly = np.array(boxes[k]).astype(np.int32)
            y0, x0 = np.min(poly, axis=0)
            y1, x1 = np.max(poly, axis=0)
            rects.append([x0, y0, x1, y1])

            # extract ROIs
        boxes = []
        for i, rect in enumerate(sorted(rects, key=cmp_to_key(self.compare_rect))):
            idx = rects.index(rect)
            p = polys[idx]
            boxes.append(p)
        box_list = list()
        boxes = np.array(boxes)
        for i in range(boxes.shape[0]):
            x1 = boxes[i, 0, 0]
            x2 = boxes[i, 1, 0]
            x3 = boxes[i, 2, 0]
            x4 = boxes[i, 3, 0]
            y1 = boxes[i, 0, 1]
            y2 = boxes[i, 1, 1]
            y3 = boxes[i, 2, 1]
            y4 = boxes[i, 3, 1]
            if evaluation:
                box_list.append([x4, y4, x1, y1, x2, y2, x3, y3])
            else:  
                box_list.append([x1, x2, x3, x4, y1, y2, y3, y4])
        return box_list, score_text