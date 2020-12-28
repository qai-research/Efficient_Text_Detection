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
from multiprocessing.pool import Pool


class BlackList:
    """"
    The blacklist method progress flow
    """

    def __init__(self, config, is_return=False, is_random_background=True, out_name='black', num_cores=1):
        self.config = config
        self.is_return = is_return
        self.is_random_background = is_random_background
        self.out_name = out_name
        self.num_cores = num_cores

    def gen_img(self, ind):
        """
        @param ind: index of background image
        @return: None
        """
        im_full_name = self.list_background[ind]
        if self.is_random_background:
            num_samples = count_nonzero(self.samples == ind)
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
        print(out)
        for input_json in out:
            ImageGenerator(fonts_path=self.config.fonts_path,
                           font_size_range=self.config.font_size_range,
                           input_json=input_json,
                           target_image=target_image,
                           fixed_box=self.config.fixed_box,
                           output_path=self.output_path,
                           source_path=self.config.source_path,
                           random_color=self.config.random_color,
                           font_color=self.config.font_color,
                           min_text_length=self.config.min_text_length,
                           max_text_length=self.config.max_text_length,
                           char_spacing_range=self.config.char_spacing_range,
                           is_object=self.config.is_object,
                           max_size=self.config.max_size,
                           method=self.config.method,
                           aug_option=self.config.aug_option,
                           from_font=not self.config.is_handwriting,
                           handwriting_path=self.config.handwriting_path
                           )

    def run(self):
        """
        Running the blacklist method
        """
        begin = time.time()
        self.list_background = os.listdir(self.config.backgrounds_path)
        if self.is_random_background:
            self.samples = random.choice(len(self.list_background), size=self.config.num_images)
        else:
            self.samples = list(range(len(self.list_background)))
        print(set(self.samples))
        output_path = self.config.output_path

        if self.config.output_path is not None:
            _date = str(datetime.datetime.now().date()).replace("-", '')
            _time = str(datetime.datetime.now().time()).replace(":", '')[:6]
            self.output_path = "_".join([output_path, self.out_name, _date, _time])
            if not os.path.exists(output_path):
                os.mkdir(self.output_path)
                os.mkdir(os.path.join(self.output_path, 'images'))
                os.mkdir(os.path.join(self.output_path, 'anotations'))
        begin = time.time()
        # with Pool(self.num_cores) as pool:
        #     pool.map(self.gen_img, list(set(self.samples)))
        for i in list(set(self.samples)):
            self.gen_img(i)
        print(self.output_path)
        return self.output_path


class WhiteList:
    """"
    The whitelist method progress flow
    """

    def __init__(self, config, is_return=False, is_random_background=True, out_name='white', num_cores=1):
        self.config = config
        self.is_return = is_return
        self.is_random_background = is_random_background
        self.out_name = out_name
        self.num_cores = num_cores

    def run(self):
        """
        Running the whitelist method
        """
        self.list_background = os.listdir(self.config.backgrounds_path)
        if self.is_random_background:
            self.samples = random.choice(len(self.list_background), size=self.config.num_images)
        else:
            self.samples = list(range(len(list_background)))

        output_path = self.config.output_path

        if self.config.output_path is not None:
            _date = str(datetime.datetime.now().date()).replace("-", '')
            _time = str(datetime.datetime.now().time()).replace(":", '')[:6]
            self.output_path = "_".join([output_path, self.out_name, _date, _time])
            if not os.path.exists(output_path):
                os.mkdir(self.output_path)
                os.mkdir(os.path.join(self.output_path, 'images'))
                os.mkdir(os.path.join(self.output_path, 'anotations'))
        # with Pool(self.num_cores) as pool:
        #     pool.map(self.gen_img, list(set(self.samples)))
        for i in list(set(self.samples)):
            self.gen_img(i)
        print(self.output_path)
        return self.output_path

    def gen_img(self, ind):
        """
        @param ind: index of background image
        @return: None
        """
        im_full_name = self.list_background[ind]
        num_samples = count_nonzero(self.samples == ind)
        target_image = os.path.join(self.config.backgrounds_path, im_full_name)
        img_name = os.path.splitext(os.path.basename(target_image))[0]
        ano_path = target_image.split("/")
        ano_folder_path = "/".join(ano_path[:-2])
        ano_path = os.path.join(ano_folder_path, 'anotations', '%s.json' % img_name)
        ImageGenerator(fonts_path=self.config.fonts_path,
                       font_size_range=self.config.font_size_range,
                       input_json=ano_path,
                       target_image=target_image,
                       output_path=self.output_path,
                       source_path=self.config.source_path,
                       random_color=self.config.random_color,
                       font_color=self.config.font_color,
                       num_samples=num_samples,
                       min_text_length=self.config.min_text_length,
                       max_text_length=self.config.max_text_length,
                       char_spacing_range=self.config.char_spacing_range,
                       is_object=self.config.is_object,
                       method=self.config.method,
                       aug_option=self.config.aug_option,
                       from_font=not self.config.is_handwriting,
                       handwriting_path=self.config.handwriting_path
                       )
