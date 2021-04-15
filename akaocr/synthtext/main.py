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
import io
import sys
import cv2
import json
import time
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .gen.boxgen import BoxGenerator
from shutil import move as move_folder
from numpy import random, count_nonzero
from .gen.img_gen import ImageGenerator
from shutil import rmtree as remove_folder
from .gen.text_to_image import TextFontGenerator
from .gen.handwriting import HandWritingGenerator
from .pre.perspective import PerspectiveTransform
from multiprocessing import Process
from synthtext.utils.text_gen import TextGenerator


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
                           handwriting_path=self.config.handwriting_path,
                           text_gen_type = self.config.TextGenType,
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
                os.mkdir(os.path.join(self.output_path, 'annotations'))
        sample_set = list(set(self.samples))
        if self.num_cores <= 1:
            for i in sample_set:
                self.gen_img(i)
        else:
            for i in range(len(sample_set) // self.num_cores):
                procs = []
                for j in sample_set[i * self.num_cores:(i + 1) * self.num_cores]:
                    proc = Process(target=self.gen_img, args=(j,))
                    procs.append(proc)
                    proc.start()
                for proc in procs:
                    proc.join()
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
                os.mkdir(os.path.join(self.output_path, 'annotations'))
        sample_set = list(set(self.samples))
        if self.num_cores <= 1:
            for i in sample_set:
                self.gen_img(i)
        else:
            for i in range(len(sample_set) // self.num_cores):
                procs = []
                for j in sample_set[i * self.num_cores:(i + 1) * self.num_cores]:
                    proc = Process(target=self.gen_img, args=(j,))
                    procs.append(proc)
                    proc.start()
                for proc in procs:
                    proc.join()
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
        ano_path = os.path.join(ano_folder_path, 'annotations', '%s.json' % img_name)
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
                       handwriting_path=self.config.handwriting_path,
                       text_gen_type = self.config.TextGenType,
                       )


class RecogGen:
    """"
    The whitelist method progress flow
    """

    def __init__(self, config, is_return=False, is_random_background=True, out_name='white', vocab_path=None,
                 num_cores=1):
        self.config = config
        self.is_return = is_return
        self.is_random_background = is_random_background
        self.out_name = out_name
        self.num_cores = num_cores
        self.vocab_dict = {}
        self.vocab_path = vocab_path
        self.img_name_length = len(str(self.config.num_images))
        if self.vocab_path is not None:
            with open(vocab_path, "r", encoding='utf-8-sig') as f:
                vocab_group = json.loads(f.read())
            for group in vocab_group:
                for char in group:
                    if char != '':
                        self.vocab_dict[char] = group

        self.text_gen = TextGenerator(self.config.source_path,
                                      min_text_length=self.config.min_text_length,
                                      max_text_length=self.config.max_text_length,
                                      replace_percentage=1,
                                      text_gen_type = self.config.TextGenType,)

        if not self.config.is_handwriting:
            self.main_text_to_image_gen = TextFontGenerator(self.config.fonts_path,
                                                            self.config.font_size_range,
                                                            fixed_box=self.config.fixed_box,
                                                            random_color=self.config.random_color,
                                                            char_spacing_range=self.config.char_spacing_range,
                                                            font_color=self.config.font_color)
        else:
            self.main_text_to_image_gen = HandWritingGenerator(self.config.handwriting_path,
                                                               self.config.fonts_path,
                                                               read_batch=32,
                                                               char_spacing_range=self.config.char_spacing_range)

    def run(self):
        """
        Running the whitelist method
        """
        self.list_background = os.listdir(self.config.backgrounds_path)

        output_path = self.config.output_path

        if self.config.output_path is not None:
            _date = str(datetime.datetime.now().date()).replace("-", '')
            _time = str(datetime.datetime.now().time()).replace(":", '')[:6]
            self.output_path = "_".join([output_path, self.out_name, _date, _time])
            if not os.path.exists(output_path):
                os.mkdir(self.output_path)
                os.mkdir(os.path.join(self.output_path, 'images'))
                os.path.join(self.output_path, 'labels.txt')
        fw = open(os.path.join(self.output_path, 'labels.txt'), 'w', encoding='utf-8-sig')
        if self.num_cores <= 1:
            for ind in range(self.config.num_images):
                template = self.text_gen.generate()
                self.gen_img(template, ind)
                fw.write('images/%s.jpg\t' % str(ind).zfill(self.img_name_length))
                fw.write(template + "\n")
        else:
            for i in range(self.config.num_images // self.num_cores):
                procs = []
                for ind in range(i * self.num_cores, (i + 1) * self.num_cores):
                    template = self.text_gen.generate()
                    proc = Process(target=self.gen_img, args=(template, ind,))
                    print(ind, template)
                    fw.write('images/%s.jpg\t' % str(ind).zfill(self.img_name_length))
                    fw.write(template + "\n")
                    procs.append(proc)
                    proc.start()
                for proc in procs:
                    proc.join()
        fw.close()
        return self.output_path

    def gen_img(self, template, ind):
        """
        @param ind: index of background image
        @return: None
        """
        im_full_name = random.choice(self.list_background)
        target_image = os.path.join(self.config.backgrounds_path, im_full_name)
        bg = cv2.imread(target_image)
        h, w, _ = bg.shape
        img, _ = self.main_text_to_image_gen.generator(template)
        img[img!=255] = 0
        out_h, out_w, _ = np.array(img).shape
        try:
            bg_y = random.choice(range(h - out_h))
            bg_x = random.choice(range(w - out_w))

            bg = bg[bg_y:bg_y + out_h, bg_x:bg_x + out_w, :]

            out_img = cv2.bitwise_and(np.float32(img), np.float32(bg))

            img_name = str(ind).zfill(self.img_name_length)
            im_path = os.path.join(self.output_path, "images/%s.jpg" % img_name)
            cv2.imwrite(im_path, out_img)
        except ValueError:
            self.gen_img(template, ind)


if __name__ == "__main__":


        # get abspath of synthtext and data_folder, add them to sys.path

        parser = argparse.ArgumentParser()
        parser.add_argument("--data_folder",
                            help="The output path",
                            type = str,
                            default = None)
        parser.add_argument("-d","--is_detect", 
                            type = boolean,
                            default = True)
        parser.add_argument("-i","--input_csv_path", 
                            help="The config csv path",
                            type=str)
        parser.add_argument('-f',"--force_remove", 
                            help="increase output verbosity",
                            type=boolean,
                            default = False)                        
        args = parser.parse_args()
        args.background_folder = os.path.join(args.data_folder, 'backgrounds')
        args.source_folder = os.path.join(args.data_folde, 'sources')
        args.font_folder = os.path.join(args.data_folde, 'fonts')
        args.object_folder = os.path.join(args.data_folde, 'objects')
        args.outputs_folder = os.path.join(args.data_folde, 'outputs')


        current_path = os.path.abspath('')
        tree = current_path.split("/")        
        i = 1
        while True:
            ocr_path = os.path.join("/".join(tree[:-i]), 'akaocr')
            i += 1
            if os.path.exists(ocr_path) and 'synthtext' in os.listdir(ocr_path):
                break        
        sys.path.append(ocr_path)
        # Add libary of synthtext app
        from synthtext.apps.white import whiteapp
        from synthtext.apps.black import blackapp
        from synthtext.apps.recog import recogapp
        from synthtext.apps.doubleblack import doubleblackapp
        from synthtext.utils.data_loader import lmdb_dataset_loader
        from synthtext.utils.utils_func import check_valid, get_all_valid
        
        bg_df, source_df, font_df = get_all_valid(args.background_folder, args.source_folder, args.font_folder)

        input_config_file = pd.read_csv(args.input_csv_path)
        key = input_config_file.columns
        checked_df = check_valid(args.input_config_file, bg_df, source_df, font_df)
        key = key.insert(0, 'DETAIL')
        key = key.insert(0, 'STATUS')
        removed = False
        # Convert input dataframe to dictionanry
        input_config_dict  = []
        for ind, values in enumerate(checked_df.values):
            input_config_dict.append({k:v for k,v in zip(checked_df.columns, values)}  )

        # Check out path and remove if existed
        if force_remove and os.path.exists(args.output_path):
            os.remove(args.output_path)
        try:    
            os.mkdir(args.output_path)
            for input_dict in input_config_dict:
                
                if input_dict['STATUS'] is "INVALID":
                    continue
                begin_time = time.time()
                Method = input_dict['Method']

                if not is_detect:
                    local_output_path = recogapp(input_dict,args)

                elif Method == 'white':
                    local_output_path = whiteapp(input_dict,args)

                elif Method == 'black':
                    local_output_path = blackapp(input_dict,args)

                elif Method == 'double_black':
                    local_output_path = doubleblackapp(input_dict,args)
                else:
                    local_output_path = None

                for path in local_output_path:
                    if not os.path.exists(output_path):
                        os.mkdir(output_path)
                    if path is not None:
                        move_folder(path, output_path)
        except FileExistsError:
            raise FileExistsError('The folder %s exists.\nTry "-f" or "--force_remove" to remove existed folder, or change the output_path'%output_path)