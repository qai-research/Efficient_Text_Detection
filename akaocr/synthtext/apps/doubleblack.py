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
from . import config
import argparse
import streamlit as st
from shutil import rmtree as remove_folder
from synthtext.main import BlackList, WhiteList
from synthtext.utils.utils_func import check_valid, get_all_valid


def doubleblackapp(input_dict):
    """
    Gen data with double black method
    """
    Method = input_dict['Method']
    NumCores = input_dict['NumCores']
    Fonts = input_dict['Fonts']
    Backgrounds = input_dict['Backgrounds']
    ObjectSources = input_dict['ObjectSource']
    TextSources = input_dict['Textsources']
    TextGenType = input_dict['TextGenType']
    ImageSources = input_dict['ImageSources']
    GenType = input_dict['GenType']
    num_images = input_dict['NumImages']
    max_num_box = input_dict['MaxNumBox']
    min_char_spacing = input_dict['MinCharSpacing']
    max_char_spacing = input_dict['MaxCharSpacing']
    min_size = input_dict['MinFontSize']
    max_size = input_dict['MaxFontSize']
    min_text_len = input_dict['MinTextLengh']
    max_text_len = input_dict['MaxTextLengh']
    max_height = input_dict['MaxHeigh']
    max_width = input_dict['MaxWidth']
    random_color = input_dict['RandomColor']
    elastic_p = input_dict['ElasticP']
    shear_p = input_dict['ShearP']
    dropout_p = input_dict['DropoutP']
    blur_p = input_dict['BlurP']
    status = input_dict['STATUS']
    detail = input_dict['DETAIL']
    
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    opt.method = Method
    opt.TextGenType = TextGenType
    opt.backgrounds_path = os.path.join(config.background_folder, Backgrounds, 'images')

    opt.fonts_path = os.path.join(config.font_folder, Fonts)
    opt.font_size_range = (min_size, max_size)
    opt.fixed_box = True
    opt.num_images = num_images
    opt.output_path = os.path.join(config.outputs_folder, Backgrounds)
    opt.source_path = os.path.join(config.source_folder, str(TextSources))
    #############
    opt.is_handwriting = (GenType != 'font')
    opt.handwriting_path = os.path.join(config.source_folder, str(ImageSources))
    #############
    opt.random_color = (random_color == 1)
    opt.font_color = (0, 0, 0)
    opt.min_text_length = min_text_len
    opt.max_text_length = max_text_len
    opt.max_num_text = None
    opt.char_spacing_range = (float(min_char_spacing), float(max_char_spacing))
    opt.max_size = (max_height, max_width)
    opt.fixed_size = None
    opt.width_random_range = (min_size * min_text_len, min_size * max_text_len)
    opt.heigh_random_range = (min_size, max_size)
    opt.box_iter = 100
    opt.max_num_box = max_num_box
    opt.num_images = num_images
    opt.aug_percent = 0
    seg_path = os.path.join(config.background_folder, Backgrounds, 'seg.h5')
    opt.segment = seg_path if os.path.exists(seg_path) else None
    opt.segment = None
    opt.aug_option = {'shear': {'p': shear_p,
                                'v': {"x": (-15, 15),
                                      "y": (-15, 15)
                                      }
                                },
                      'elastic': {'p': elastic_p,
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
    # Fisrt black method running with text source
    begin_time = time.time()
    opt.method = 'black'
    opt.fixed_size = None
    opt.width_random_range = (min_size * min_text_len, min_size * max_text_len)
    opt.heigh_random_range = (min_size, max_size)
    opt.box_iter = 100
    opt.max_num_box = max_num_box * 10
    opt.num_images = num_images
    opt.aug_percent = 0
    opt.segment = None
    opt.source_path = os.path.join(config.data, 'source.txt')
    opt.is_object = False
    opt.char_spacing_range = (min_char_spacing, max_char_spacing)
    runner = BlackList(opt, num_cores=NumCores)
    new_backgrounds_path = runner.run()

    # Second black method running with text source if the object source is not existed.
    opt.max_num_box = max_num_box
    opt.backgrounds_path = os.path.join(new_backgrounds_path, 'images')
    opt.output_path = os.path.join(config.outputs_folder, Backgrounds)
    if ObjectSources == 0:
        opt.is_object = False
        opt.source_path = os.path.join(config.source_folder, TextSources)
    else:
        opt.is_object = True
        opt.is_handwriting = False
        opt.source_path = os.path.join(config.source_folder, ObjectSources)
    runner = BlackList(opt, is_random_background=False, out_name='double_black', num_cores=NumCores)
    output_path = runner.run()
    remove_folder(new_backgrounds_path)
    return [output_path]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='', help='path to config file')
    parser.add_argument('--output_path', type=str, default='', help='path to output folder')
    opt = parser.parse_args()
    bg_df, source_df, font_df = get_all_valid(config)
    config_file = pd.read_csv(opt.config_path)
    key = config_file.columns
    checked_df = check_valid(config_file, bg_df, source_df, font_df)
    for index, value in enumerate(checked_df.values):
        Method = value[0]
        status = value[-2]
        if status is "INVALID" or Method != 'doubleblack':
            continue
        local_output_path = doubleblackapp(value)
        for path in local_output_path:
            if not os.path.exists(opt.output_path):
                os.mkdir(opt.output_path)
            if path is not None:
                move_folder(path, opt.output_path)
