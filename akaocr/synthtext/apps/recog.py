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
from synthtext.main import RecogGen
from shutil import rmtree as remove_folder
from synthtext.utils.utils_func import check_valid, get_all_valid


def recogapp(input_dict):
    """
    Gen data with white method
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
    opt.input_json = os.path.join(config.background_folder, Backgrounds, 'annotations')
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

    opt.is_object = False
    opt.source_path = os.path.join(config.source_folder, str(TextSources))
    runner = RecogGen(opt, out_name='Recog', num_cores=NumCores)
    output_path = runner.run()
    return [output_path]

    # Create font to images generator
