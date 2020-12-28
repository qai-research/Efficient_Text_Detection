#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Vu Hoang Viet - VietVH9
Created Date: NOV - 2020
Project : AkaOCR core
_____________________________________________________________________________



The Module Has Been Build To Have Progress
_____________________________________________________________________________
"""

import os
import sys
import cv2
import json
import time
import random
import datetime
import numpy as np
from .boxgen import BoxGenerator
from .text_to_image import TextFontGenerator
from .handwriting import HandWritingGenerator
from synthtext.utils.text_gen import TextGenerator
from synthtext.utils.utils_func import Augmentator
from synthtext.pre.perspective import PerspectiveTransform


def ImageGenerator(fonts_path=None,
                   font_size_range=None,
                   input_json=None,
                   target_image=None,
                   random_color=False,
                   font_color=(0, 0, 0),
                   new_text_gen=False,
                   fixed_box=True,
                   output_path=None,
                   vocab_path=None,
                   method='white',
                   num_samples=1,
                   is_object=False,
                   source_path=None,
                   max_size=None,
                   is_return=False,
                   aug_option=None,
                   from_font=True,
                   **kwargs):
    """
    The image generator with each background and bounding box
    """
    # Load target label
    if type(input_json) == str:
        with open(input_json, 'r', encoding='utf-8-sig') as reader:
            input_json = json.loads(reader.read())

    target_points = [np.float32([[(char['x1'], char['y1']),
                                  (char['x2'], char['y2']),
                                  (char['x3'], char['y3']),
                                  (char['x4'], char['y4'])]]) for char in input_json['words']]

    if not is_object:
        if not method == 'white':
            new_text_gen = True
            # Create new text generator
            try:
                text_gen = TextGenerator(source_path,
                                         vocab_group_path=vocab_path,
                                         min_text_length=kwargs.pop('min_text_length'),
                                         max_text_length=kwargs.pop('max_text_length'),
                                         replace_percentage=1)
            except KeyError:
                try:
                    text_gen = TextGenerator(source_path,
                                             vocab_group_path=vocab_path,
                                             min_text_length=kwargs.pop('min_text_length'),
                                             replace_percentage=1)
                except KeyError:
                    text_gen = TextGenerator(source_path,
                                             vocab_group_path=vocab_path,
                                             replace_percentage=1)

        # Create font to images generator
        if from_font:
            main_text_to_image_gen = TextFontGenerator(fonts_path,
                                                       font_size_range,
                                                       fixed_box=fixed_box,
                                                       random_color=random_color,
                                                       char_spacing_range=None,
                                                       font_color=font_color)
        else:
            main_text_to_image_gen = HandWritingGenerator(kwargs.pop('handwriting_path'),
                                                          fonts_path,
                                                          read_batch=32,
                                                          char_spacing_range=None,
                                                          fixed_box=True)
        # Generator new text to replace old text
        source_images = []
        source_chars_coor = []
        for word in input_json['words']:
            print(word['text'])
            if new_text_gen:
                # Generator text with out any infomation of source text
                img, out_json = main_text_to_image_gen.generator(text_gen.generate())

            else:
                # Generator text with infomation of source text
                template = word['text']
                vocab_dict = {}
                if vocab_path is not None:
                    with open(vocab_path, "r", encoding='utf-8-sig') as f:
                        vocab_group = json.loads(f.read())
                    for group in vocab_group:
                        for char in group:
                            if char != '':
                                vocab_dict[char] = group
                for i in vocab_dict:
                    if i in template:
                        for _ in range(template.count(i)):
                            template = template.replace(i, random.choice(vocab_dict[i]), 1)
                img, out_json = main_text_to_image_gen.generator(template)
            source_images.append(np.array(img))
            source_chars_coor.append(out_json)
    else:
        assert source_path is not None

        source_images = []
        source_chars_coor = []
        for img_name in os.listdir(os.path.join(source_path, 'images')):
            basename = os.path.splitext(img_name)[0]
            im_path = os.path.join(source_path, "images/%s" % img_name)
            json_path = os.path.join(source_path, "labels/%s.json" % basename)
            img = cv2.imread(im_path)
            with open(json_path, 'r', encoding='utf-8-sig') as reader:
                img_info = json.loads(reader.read())
            source_images.append(np.array(img))
            source_chars_coor.append(img_info)
        sample_index = np.random.choice(range(len(source_images)), size=len(target_points))
        source_images = [source_images[i] for i in sample_index]
        source_chars_coor = [source_chars_coor[i] for i in sample_index]

    source_images, source_chars_coor = Augmentator(source_images, source_chars_coor, aug_option)
    # Preprocess the clean background
    trans = PerspectiveTransform(source_images, source_chars_coor, target_points, target_image, max_size=max_size)
    if method == 'white':
        trans.inpainting = True
    else:
        trans.inpainting = False
    for _ in range(num_samples):
        if is_object:
            sample_index = np.random.choice(range(len(source_images)), size=len(target_points))
            trans.source_images = [source_images[i] for i in sample_index]
            trans.source_chars_coor = [source_chars_coor[i] for i in sample_index]
        result_pic, out_json_trans = trans.fit()
        if output_path is not None:
            img_basename = os.path.basename(out_json_trans['file'])
            name = os.path.splitext(img_basename)[0]

            image_path = os.path.join(output_path, 'images/%s' % img_basename)
            json_path = os.path.join(output_path, 'anotations/%s.json' % name)

            cv2.imwrite(image_path, result_pic)
            with open(json_path, 'w', encoding='utf-8-sig') as fw:
                fw.write(json.dumps(out_json_trans))
    if is_return:
        return output_path
