#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Ngoc Nghia - Nghiann3
Created Date: Fri March 12 13:00:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contains unit test for pipeline of whole model
_____________________________________________________________________________
"""

import argparse
from pipeline.layers import SlideWindow, Detectlayer, Recoglayer
from pipeline.util import Visualizer
from pathlib import Path
import cv2

def test_pipeline(args):
    img = cv2.imread(args.image_path)
    file_path = Path(args.image_path)
    img_name = file_path.stem
    slidewindow = SlideWindow(window=(1280, 800))
    deteclayer = Detectlayer(model_name="test")
    recoglayer = Recoglayer(model_name="test")
    vzer = Visualizer(data_path=args.data_path)

    out = slidewindow(img)
    boxes, heat = deteclayer(out)
    texts_1, conf = recoglayer(img, boxes=boxes)
    img_1 = vzer.visualizer(img, contours=boxes, show=False, texts=texts_1)
    cv2.imwrite(img_name+'_recog.jpg', img_1)

    subvocab = ['0','1','2','3','4','5','6','7','8','9']
    texts_2, conf = recoglayer(img, boxes=boxes, subvocab=subvocab)
    img_2 = vzer.visualizer(img, contours=boxes, show=False, texts=texts_2)
    cv2.imwrite(img_name+'_recog_num.jpg', img_2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./test/images/image_1.jpg', required=True, help='path to input .jpg image')
    parser.add_argument('--data_path', type=str, default='./data', help='path to folder contains input .ttf font')
    args = parser.parse_args()
    test_pipeline(args)

if __name__ == '__main__':
    main()
