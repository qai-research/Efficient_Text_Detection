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
from synthtext.main import BlackList, WhiteList


def handwritingapp(value, source_df):
    Method, Fonts, Backgrounds, ObjectSources, Textsources, num_images, max_num_box = value[:7]
    char_spacing, min_size, max_size, min_text_len, max_text_len, random_color = value[:-4][7:]
    max_height, max_width, status, detail = value[-4:]
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
    if ObjectSources == 0:
        opt.is_object = False
        opt.source_path = os.path.join(config.source_folder, Textsources)
    else:
        opt.is_object = True
        opt.source_path = os.path.join(config.source_folder, Textsources)

    st.warning("Begin running SynthText with %s folder" % Backgrounds)
    begin_time = time.time()
    # GenText
    opt.input_json = os.path.join(config.background_folder, Backgrounds, 'anotations')

    runner = WhiteList(opt)
    output_path = runner.run()
    st.write("Time for this process was %s seconds" % int(time.time() - begin_time))
    return output_path
