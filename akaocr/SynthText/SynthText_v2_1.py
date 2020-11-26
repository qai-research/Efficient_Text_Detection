import os
import sys
import cv2
import json
import time
import random
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../akaocr')
from utils.text_gen import TextGenerator

from SynthText.utils import get_args
from SynthText.generator import generate
from SynthText.TextToImage.fromfont import TextFontGenerator
from SynthText.BackgroundProcessing.box_generator import BoxGenerator
from SynthText.ImgtextProcessing.PerspectiveTransformation import Transform


class BlackList():

    def __init__(self, opt, is_return = False, is_random_sample = True): 
        self.opt = opt
        self.is_return = is_return
        self.is_random_sample = is_random_sample

    def run(self):        

        list_background = os.listdir(self.opt.backgrounds_path)
        if self.is_random_sample:
            samples = np.random.choice(len(list_background),size = self.opt.num_images)
        else:
            samples = list(range(len(list_background)))

        output_path = self.opt.output_path

        if self.opt.output_path is not None:
            _date = str(datetime.datetime.now().date()).replace("-",'')
            _time = str(datetime.datetime.now().time()).replace(":",'')[:6]
            output_path = "_".join([output_path,_date,_time])
            os.mkdir(output_path)
            os.mkdir(os.path.join(output_path,'images'))
            os.mkdir(os.path.join(output_path,'anotations'))

        for ind in set(samples):
            im_full_name = list_background[ind]
            if self.is_random_sample:
                num_samples = np.count_nonzero(samples==ind)
            else:
                num_samples = 1

            target_image = os.path.join(self.opt.backgrounds_path,im_full_name)
            box_gen = BoxGenerator(target_image,
                                fixed_size           = self.opt.fixed_size, 
                                weigh_random_range   = self.opt.weigh_random_range, 
                                heigh_random_range   = self.opt.heigh_random_range, 
                                box_iter             = self.opt.box_iter, 
                                max_num_box          = self.opt.max_num_box,
                                num_samples          = num_samples,
                                aug_percent          = self.opt.aug_percent,
                                segment              = self.opt.segment,
                                max_size             = self.opt.max_size
                            )
            out = box_gen.run()  
            for input_json in out:
                pic, out_json = generate(fonts_path         = self.opt.fonts_path, 
                                        font_size_range     = self.opt.font_size_range, 
                                        input_json          = input_json, 
                                        target_image        = target_image, 
                                        fixed_box           = self.opt.fixed_box, 
                                        output_path         = output_path,
                                        source_path         = self.opt.source_path,
                                        random_color        = self.opt.random_color, 
                                        font_color          = self.opt.font_color,
                                        min_text_length     = self.opt.min_text_length,
                                        max_text_length     = self.opt.max_text_length,
                                        is_object           = self.opt.is_object,
                                        max_size            = self.opt.max_size,
                                        method              = self.opt.method
                                        )
        if self.is_return:
            return output_path


class WhiteList():

    def __init__(self, opt):
        self.opt = opt

    def run(self):

        list_background = os.listdir(self.opt.backgrounds_path)
        samples = np.random.choice(len(list_background),size = self.opt.num_images)
        output_path = self.opt.output_path
        if self.opt.output_path is not None:
            _date = str(datetime.datetime.now().date()).replace("-",'')
            _time = str(datetime.datetime.now().time()).replace(":",'')[:6]
            output_path = "_".join([output_path,_date,_time])
            os.mkdir(output_path)
            os.mkdir(os.path.join(output_path,'images'))
            os.mkdir(os.path.join(output_path,'anotations'))

        for ind in set(samples):
            im_full_name = list_background[ind]
            num_samples = np.count_nonzero(samples==ind)
            print(num_samples)
            target_image = os.path.join(self.opt.backgrounds_path,im_full_name)
            img_name = os.path.splitext(os.path.basename(target_image))[0]
            ano_path = target_image.split("/")
            ano_folder_path = "/".join(ano_path[:-2])
            ano_path = os.path.join(ano_folder_path, 'anotations','%s.json'%img_name)

            a = time.time()

            pic, out_json = generate(fonts_path         = self.opt.fonts_path, 
                                    font_size_range     = self.opt.font_size_range, 
                                    input_json          = ano_path, 
                                    target_image        = target_image, 
                                    output_path         = output_path,
                                    source_path         = self.opt.source_path,
                                    random_color        = self.opt.random_color, 
                                    font_color          = self.opt.font_color,
                                    num_samples         = num_samples,
                                    min_text_length     = self.opt.min_text_length,
                                    max_text_length     = self.opt.max_text_length,
                                    is_object           = self.opt.is_object,
                                    method              = self.opt.method
                                    )          
            print(time.time()-a)                          

if __name__ == '__main__':
    opt = get_args()

    if opt.method == 'whitelist':
        runner = whitelist()
    elif opt.method == 'blacklist':
        runner = blacklist()
    else:
        raise KeyError("Do not have %s method"%opt.method)

    runner.opt = opt
    runner.run()