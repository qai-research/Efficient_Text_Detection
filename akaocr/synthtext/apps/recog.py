"""
_____________________________________________________________________________
Created By  : Vu Hoang Viet - VietVH9
Created Date: NOV - 2020
Project : AkaOCR core
_____________________________________________________________________________



The Module Has Been Build for...
_____________________________________________________________________________
"""
import os
import io
import sys
import time
import config
import argparse
import streamlit as st
from synthtext.main import WhiteList
from shutil import rmtree as remove_folder
from synthtext.utils.utils_func import check_valid, get_all_valid
from synthtext.gen.text_to_image import TextFontGenerator
from synthtext.gen.handwriting import HandWritingGenerator


def recogapp(value):
    """
    Gen data with white method
    """
    Method, NumCores, Fonts, Backgrounds = value[:4]
    ObjectSources, TextSources, ImageSources, GenType, num_images, max_num_box = value[4:10]
    min_char_spacing, max_char_spacing, min_size, max_size, min_text_len, max_text_len, random_color = value[10:-7]
    max_height, max_width, shear_p, dropout_p, blur_p, status, detail = value[-7:]
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    opt.method = Method
    opt.backgrounds_path = os.path.join(config.background_folder, Backgrounds, 'images')

    opt.fonts_path = os.path.join(config.font_folder, Fonts)
    opt.font_size_range = (min_size, max_size)
    opt.fixed_box = True
    opt.num_images = num_images
    opt.output_path = os.path.join(config.outputs_folder, Backgrounds)
    opt.source_path = os.path.join(config.source_folder, TextSources)
    #############
    opt.is_handwriting = (GenType != 'font')
    opt.handwriting_path = os.path.join(config.source_folder, ImageSources)
    #############
    opt.random_color = (random_color == 1)
    opt.font_color = (0, 0, 0)
    opt.min_text_length = min_text_len
    opt.max_text_length = max_text_len
    opt.max_num_text = None
    opt.char_spacing_range = (float(min_char_spacing), float(max_char_spacing))
    opt.max_size = (max_height, max_width)
    opt.input_json = os.path.join(config.background_folder, Backgrounds, 'anotations')
    opt.aug_option = {'shear': {'p': shear_p,
                                'v': {"x": (-15, 15),
                                      "y": (-15, 15)
                                      }
                                },
                      'dropout': {'p': dropout_p,
                                  'v': (0.2, 0.3)
                                  },
                      'blur': {'p': blur_p,
                               'v': (0.0, 2.0)
                               }
                      }
    begin_time = time.time()
    results = []

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