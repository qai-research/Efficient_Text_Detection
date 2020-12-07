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
from shutil import rmtree as remove_folder
from synthtext.main import BlackList, WhiteList


def doubleblackapp(value):
    Method, Fonts, Backgrounds, ObjectSources, TextSources, num_images, max_num_box = value[:7]
    char_spacing, min_size, max_size, min_text_len, max_text_len, random_color = value[7:-7]
    max_height, max_width, status, detail, shear_p, dropout_p, blur_p = value[-7:]
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    opt.method = Method
    opt.backgrounds_path = os.path.join(config.background_folder, Backgrounds, 'images')

    opt.fonts_path = os.path.join(config.font_folder, Fonts)
    opt.font_size_range = (min_size, max_size)
    opt.fixed_box = True
    opt.num_images = num_images
    opt.output_path = os.path.join(config.outputs_folder, Backgrounds)
    opt.source_path = os.path.join(config.source_folder, Textsources)
    opt.random_color = (random_color == 1)
    opt.font_color = (0, 0, 0)
    opt.min_text_length = min_text_len
    opt.max_text_length = max_text_len
    opt.max_num_text = None
    opt.max_size = (max_height, max_width)
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

    st.warning("Begin running %s Method SynthText with folder %s " %(opt.method,Backgrounds))
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
    runner = BlackList(opt, is_return=True)
    new_backgrounds_path = runner.run()

    # Second black method running with textsource if the object source is not existed.
    opt.max_num_box = max_num_box
    opt.backgrounds_path = os.path.join(new_backgrounds_path, 'images')
    opt.output_path = os.path.join(new_backgrounds_path, "Double_black")
    if ObjectSources == 0:
        opt.is_object = False
        opt.source_path = os.path.join(config.source_folder, Textsources)
    else:
        opt.is_object = True
        opt.source_path = os.path.join(config.source_folder, ObjectSources)
    runner = BlackList(opt, is_random_background=False, out_name='double_black')
    output_path = runner.run()
    remove_folder(new_backgrounds_path)
    st.write("Time for this process was %s seconds" % int(time.time() - begin_time))
    return output_path
