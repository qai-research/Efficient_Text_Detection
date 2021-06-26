# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Huu Kim - Kimnh3
Created Date: May 24 17:41:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contain code for TensorRT engine inference
Refer from TensorRT example:
https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html
https://github.com/NVIDIA/TensorRT/blob/master/quickstart/SemanticSegmentation/tutorial-runtime.ipynb
_____________________________________________________________________________
"""

# Import required modules
import numpy as np
import argparse
import os
from pathlib import Path
import cv2
import sys

sys.path.append("../../")
from pipeline.util import Visualizer
from engine.config import setup, parse_base
from pipeline.layers_trt import SlideWindow, Detectlayer


# Main inference pipeline
def infer(data, exp, input, output):
    if not os.path.exists(output):
        os.mkdir(output)
    print("Reading input image from file {}".format(input))
    img_name = Path(input).stem
    # with cv2.imread(input) as image:
    #     image_width = image.width
    #     image_height = image.height
    image = cv2.imread(input)
    image_width = image.shape[0]
    image_height = image.shape[1]
    print("Running TensorRT inference for akaOCR " + "detec engine")

    deteclayer = Detectlayer(model_name=exp)
    # recoglayer = Recoglayer(model_name=exp)
    vzer = Visualizer(data_path=data)

    slidewindow = SlideWindow(window=(1280, 800))
    out = slidewindow(image)
    # image = np.asarray(image, dtype='float32')
    boxes, heat = deteclayer(out)

    # texts_1, conf = recoglayer(image, boxes=boxes)
    img_1 = vzer.visualizer(image, contours=boxes, show=False)
    print(os.path.join(output, img_name+'_detec_rt.jpg'))
    cv2.imwrite(os.path.join(output, img_name+'_detec_rt.jpg'), img_1)

def main():
    parser = argparse.ArgumentParser(description="Infer akaOCR model from TensorRT engine (Python API), visualize and save into output image")
    parser.add_argument("--data", required=False, help="Path to parent data folder, include font to visualize (parent of exp_<detec/recog>/<exp_name>)", default="../../data")
    parser.add_argument("--exp", required=False, help="Path to experiment folder name (exp_<detec/recog>/<exp_name>)", default="test")
    parser.add_argument("--input", required=False, help="Path to input image", default="../../test/images/image_1.jpg")
    parser.add_argument("--output", required=False, help="Path to folder that save the output image, should be of same parent as engine data path", default="../../data/infer_result")
    args = parser.parse_args()
    infer(args.data, args.exp, args.input, args.output)



if __name__ == '__main__':
    main()

# # segment output
# plt.imshow(Image.open(output_file))

# python inference_trt.py --