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
from synthtext.main import RecogGen
from shutil import rmtree as remove_folder
from synthtext.utils.utils_func import check_valid, get_all_valid


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

    opt.is_object = False
    opt.source_path = os.path.join(config.source_folder, str(TextSources))
    runner = RecogGen(opt, out_name='Recog', num_cores=NumCores)
    output_path = runner.run()
    return [output_path]

    # Create font to images generator
