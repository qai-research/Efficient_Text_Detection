#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Vu Hoang Viet - VietVH9
Created Date: NOV - 2020
Project : AkaOCR core
_____________________________________________________________________________



The Module Has Been Build for Running SynthText as completed flow
_____________________________________________________________________________
"""
import os
import sys
import cv2
import json
import time
import argparse
import datetime
import matplotlib.pyplot as plt
from numpy import random, count_nonzero
from .gen.img_gen import ImageGenerator
from .gen.boxgen import BoxGenerator
from .gen.text_to_image import TextFontGenerator
from .pre.perspective import PerspectiveTransform


class BlackList:
    """"
    The blacklist method progress flow
    """

    def __init__(self, config, is_return=False, is_random_background=True, out_name='black'):
        self.config = config
        self.is_return = is_return
        self.is_random_background = is_random_background
        self.out_name = out_name

    def run(self):
        """
        Running the blacklist method
        """
        list_background = os.listdir(self.config.backgrounds_path)
        if self.is_random_background:
            samples = random.choice(len(list_background), size=self.config.num_images)
        else:
            samples = list(range(len(list_background)))

        output_path = self.config.output_path

        if self.config.output_path is not None:
            _date = str(datetime.datetime.now().date()).replace("-", '')
            _time = str(datetime.datetime.now().time()).replace(":", '')[:6]
            output_path = "_".join([output_path, self.out_name, _date, _time])
            if not os.path.exists(output_path):
                os.mkdir(output_path)
                os.mkdir(os.path.join(output_path, 'images'))
                os.mkdir(os.path.join(output_path, 'anotations'))

        for ind in set(samples):
            im_full_name = list_background[ind]
            if self.is_random_background:
                num_samples = count_nonzero(samples == ind)
            else:
                num_samples = 1

            target_image = os.path.join(self.config.backgrounds_path, im_full_name)
            box_gen = BoxGenerator(target_image,
                                   fixed_size=self.config.fixed_size,
                                   width_random_range=self.config.width_random_range,
                                   heigh_random_range=self.config.heigh_random_range,
                                   box_iter=self.config.box_iter,
                                   max_num_box=self.config.max_num_box,
                                   num_samples=num_samples,
                                   aug_percent=self.config.aug_percent,
                                   segment=self.config.segment,
                                   max_size=self.config.max_size
                                   )
            out = box_gen.run()
            for input_json in out:
                ImageGenerator(fonts_path=self.config.fonts_path,
                               font_size_range=self.config.font_size_range,
                               input_json=input_json,
                               target_image=target_image,
                               fixed_box=self.config.fixed_box,
                               output_path=output_path,
                               source_path=self.config.source_path,
                               random_color=self.config.random_color,
                               font_color=self.config.font_color,
                               min_text_length=self.config.min_text_length,
                               max_text_length=self.config.max_text_length,
                               is_object=self.config.is_object,
                               max_size=self.config.max_size,
                               method=self.config.method
                               )
        return output_path


class WhiteList:
    """"
    The whitelist method progress flow
    """

    def __init__(self, config, is_return=False, is_random_background=True, out_name='white'):
        self.config = config
        self.is_return = is_return
        self.is_random_background = is_random_background
        self.out_name = out_name

    def run(self):
        """
        Running the whitelist method
        """
        list_background = os.listdir(self.config.backgrounds_path)
        if self.is_random_background:
            samples = random.choice(len(list_background), size=self.config.num_images)
        else:
            samples = list(range(len(list_background)))
        output_path = self.config.output_path
        if self.config.output_path is not None:
            _date = str(datetime.datetime.now().date()).replace("-", '')[4:]
            _time = str(datetime.datetime.now().time()).replace(":", '')[:6]
            output_path = "_".join([output_path, self.out_name, _date, _time])
            if not os.path.exists(output_path):
                os.mkdir(output_path)
                os.mkdir(os.path.join(output_path, 'images'))
                os.mkdir(os.path.join(output_path, 'anotations'))

        for ind in set(samples):
            im_full_name = list_background[ind]
            num_samples = count_nonzero(samples == ind)
            print(num_samples)
            target_image = os.path.join(self.config.backgrounds_path, im_full_name)
            img_name = os.path.splitext(os.path.basename(target_image))[0]
            ano_path = target_image.split("/")
            ano_folder_path = "/".join(ano_path[:-2])
            ano_path = os.path.join(ano_folder_path, 'anotations', '%s.json' % img_name)
            ImageGenerator(fonts_path=self.config.fonts_path,
                           font_size_range=self.config.font_size_range,
                           input_json=ano_path,
                           target_image=target_image,
                           output_path=output_path,
                           source_path=self.config.source_path,
                           random_color=self.config.random_color,
                           font_color=self.config.font_color,
                           num_samples=num_samples,
                           min_text_length=self.config.min_text_length,
                           max_text_length=self.config.max_text_length,
                           is_object=self.config.is_object,
                           method=self.config.method
                           )
        return output_path

# if __name__ == '__main__':
#     config = get_args()
#
#     if config.method == 'whitelist':
#         runner = whitelist()
#     elif config.method == 'blacklist':
#         runner = blacklist()
#     else:
#         raise KeyError("Do not have %s method" % config.method)
#
#     runner.config = config
#     runner.run()
