import os
import sys
import cv2
import json
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0,'../akaocr')
from akaocr.utils.text_gen import TextGenerator
from akaocr.SynthText.generator import generate
from akaocr.SynthText.TextToImage.fromfont import TextFontGenerator
from akaocr.SynthText.BackgroundProcessing.box_generator import BoxGenerator
from akaocr.SynthText.ImgtextProcessing.PerspectiveTransformation import Transform


def get_args():

    parser = argparse.ArgumentParser(description='Run SynthText')

    parser.add_argument('--backgrounds_path',
                         type = str, 
                         default = '.', 
                         help = 'The path of background directory, contains all background.')

    parser.add_argument('--vocab_path',
                         type = str, 
                         default = '.', 
                         help = 'The path of vocab files')

    parser.add_argument('--segment_path',
                         type = str,
                         default = '.', 
                         help = "The path to segmentation files (h5) generate with matlab. With an image didn't segment, it will be auto segment in another algorithism")

    #######################################

    parser.add_argument('--num_samples',
                         type = int, 
                         default = 100, 
                         help = 'The path of source text path')
                         
    parser.add_argument('--aug_percent',
                         type = float, 
                         default = 0.5, 
                         help = 'The path of source text path')

    parser.add_argument('--max_num_box',
                         type = int, 
                         type = 100, 
                         help = 'The path of source text path')

    parser.add_argument('--box_iter',
                         type = int,
                         default = 100, 
                         help = 'The path of source text path')

    parser.add_argument('--fixed_size',
                         type = tuple, 
                         default = None, 
                         help = 'The path of source text path')

    parser.add_argument('--weigh_random_range',
                         type = str, 
                         type = None, 
                         help = 'The path of source text path')

    parser.add_argument('--heigh_random_range',
                         type = str, 
                         type = None, 
                         help = 'The path of source text path')

    parser.add_argument('--method',
                         type = str, 
                         type = None, 
                         help = 'The path of source text path')

    #######################################

    parser.add_argument('--fonts_path',
                         type = str, 
                         default = '.', 
                         help = 'The path of font directory, just for True Tye Font (.ttf) files.')

    parser.add_argument('--source_text_path',
                         type = str, 
                         default = '.', 
                         help = 'The path of source text path')

    parser.add_argument('--random_color',
                         type = boolean,
                         default = False,  
                         help = 'The path of source text path')

    parser.add_argument('--font_color',
                         type = list,
                         default = [0,0,0],  
                         help = 'The path of source text path')

    parser.add_argument('--output_path',
                         type = str, 
                         default = '.', 
                         help = 'The path of source text path')

    opt = parser.parse_args()
    return opt

def main(opt):    
    for im_path in os.listdir(opt.backgrounds_path):
        target_image = os.path.join(opt.backgrounds_path,im_path)
        box_gen = BoxGenerator(target_image,
                               fixed_size           = opt.fixed_size, 
                               weigh_random_range   = opt.weigh_random_range, 
                               heigh_random_range   = opt.heigh_random_range, 
                               box_iter             = opt.box_iter, 
                               max_num_box          = opt.max_num_box,
                               num_samples          = opt.num_samples,
                               aug_percent          = opt.aug_percent,
                               segment_path         = opt.segment_path
                          )
                          
        a = time.time()
        for input_json in out:
            pic, out_json = generate(fonts_path         = opt.fonts_path, 
                                    font_size_range     = opt.font_size_range, 
                                    input_json          = opt.input_json, 
                                    target_image        = opt.target_image, 
                                    fixed_box           = opt.fixed_box, 
                                    output_path         = opt.output_path,
                                    source_text_path    = opt.source_text_path,
                                    random_color        = opt.random_color, 
                                    font_color          = opt.font_color)

if __name__ == '__main__':
    opt = get_args()