# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Huu Kim - Kimnh3
Created Date: Jun 12 16:22:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contain code for inferencing using NVIDIA Triton server
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
from pipeline.layers_triton import SlideWindow, Detectlayer, Recoglayer
# from pipeline.layers import Recoglayer
from engine.config import setup, parse_base
from utils.runtime import start_timer, end_timer_and_print
from utils.utility import initial_logger
logger = initial_logger()

# Main inference pipeline
def infer(FLAGS):
    if not os.path.exists(FLAGS.output):
        os.mkdir(FLAGS.output)

    logger.info("Reading input image from file {}".format(FLAGS.input))
    
    img_name = Path(FLAGS.input).stem
    # with cv2.imread(FLAGS.input) as image:
    #     image_width = image.width
    #     image_height = image.height
    image = cv2.imread(FLAGS.input)
    # image_width = image.shape[0]
    # image_height = image.shape[1]
    
    logger.info("Running inference using Triton server for detec -> recog pth")

    deteclayer = Detectlayer(model_name=FLAGS.exp)
    recoglayer = Recoglayer(model_name=FLAGS.exp)
    vzer = Visualizer(data_path=FLAGS.data)

    # slidewindow = SlideWindow(window=(1280, 800))
    # out = slidewindow(image)
    out=image
    # image = np.asarray(image, dtype='float32')
    boxes, heat = deteclayer(out, FLAGS)

    texts_1, conf = recoglayer(image, boxes=boxes)
    img_1 = vzer.visualizer(image, contours=boxes, show=False, texts=texts_1) # with recog
    # img_1 = vzer.visualizer(image, contours=boxes, show=False) # without recog
    img_new_name = os.path.join(FLAGS.output, img_name+'_infer_triton.jpg')
    logger.info('Infer successfully, saved result at: ',img_new_name)

    cv2.imwrite(img_new_name, img_1)


def main():
    parser = argparse.ArgumentParser(description="Infer akaOCR model from Triton server (GRPC/HTTP), visualize and save into output image")
    parser.add_argument("--data", required=False, help="Path to parent data folder, include font to visualize (parent of exp_<detec/recog>/<exp_name>)", default="../../data")
    parser.add_argument("--exp", required=False, help="Path to experiment folder name (exp_<detec/recog>/<exp_name>)", default="test")
    # parser.add_argument("--input", required=False, help="Path to input image", default="../../test/images/image_1.jpg")
    parser.add_argument("--output", required=False, help="Path to folder that save the output image, should be of same parent as engine data path", default="../../data/infer_result")
    
    # triton
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-a',
                        '--async',
                        dest="async_set",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use asynchronous inference API')
    parser.add_argument('--streaming',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use streaming inference API. ' +
                        'The flag is only available with gRPC protocol.')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=True,
                        help='Name of model')
    parser.add_argument(
        '-x',
        '--model-version',
        type=str,
        required=False,
        default="",
        help='Version of model. Default is to use latest version.')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-c',
                        '--classes',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument(
        '-s',
        '--scaling',
        type=str,
        choices=['NONE', 'INCEPTION', 'VGG'],
        required=False,
        default='NONE',
        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument('-i',
                        '--protocol',
                        type=str,
                        required=False,
                        default='HTTP',
                        help='Protocol (HTTP/gRPC) used to communicate with ' +
                        'the inference service. Default is HTTP.')
    parser.add_argument('--input',
                        type=str,
                        nargs='?',
                        default="../../test/images/image_1.jpg",
                        help='Input image / Input folder.')
    FLAGS = parser.parse_args()
    
    infer(FLAGS)



if __name__ == '__main__':
    main()

# # segment output
# plt.imshow(Image.open(output_file))

# python inference_triton.py -m='smz_detec' -x=1
# python inference_triton.py -m='smz_detec_onnx' -x=1
# python inference_triton.py -m='smz_detec_trt' -x=1