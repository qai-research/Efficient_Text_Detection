#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Huu Kim - Kimnh3
Created Date: May 24 14:31:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contains infer pipeline of whole model, format onnx
_____________________________________________________________________________
"""
import sys
sys.path.append("../../")
import argparse
from pipeline.layers import SlideWindow, Detectlayer, Recoglayer
from pipeline.util import Visualizer
from pathlib import Path
import cv2
from utils.runtime import start_timer, end_timer_and_print

def test_pipeline(args):
    # start_timer()
    img = cv2.imread(args.image_path)
    file_path = Path(args.image_path)

    img_name = file_path.stem
    # slidewindow = SlideWindow(window=(1280, 800))
    deteclayer = Detectlayer(model_name="test")
    # recoglayer = Recoglayer(model_name="test")
    vzer = Visualizer(data_path=args.data_path)

    # out = slidewindow(img)
    out=img
    boxes, heat = deteclayer(out)
    # texts_1, conf = recoglayer(img, boxes=boxes)
    img_1 = vzer.visualizer(img, contours=boxes, show=False)
    # img_1 = vzer.visualizer(img, contours=boxes, show=False, texts=texts_1)
    cv2.imwrite(img_name+'_recog.jpg', img_1)
    # end_timer_and_print("Inference using .pth model: ")

    # subvocab = ['0','1','2','3','4','5','6','7','8','9']
    # texts_2, conf = recoglayer(img, boxes=boxes, subvocab=subvocab)
    # img_2 = vzer.visualizer(img, contours=boxes, show=False, texts=texts_2)
    # cv2.imwrite(img_name+'_recog_num.jpg', img_2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./test/images/image_1.jpg', required=False, help='path to input .jpg image')
    parser.add_argument('--data_path', type=str, default='./data', help='path to folder contains input .ttf font')
    args = parser.parse_args()
    test_pipeline(args)

if __name__ == '__main__':
    main()
