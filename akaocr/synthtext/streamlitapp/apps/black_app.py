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


def blackapp(value, source_df, out_name='black'):
    """
    Gen data with black method
    """
    Method, Fonts, Backgrounds, ObjectSources, TextSources, num_images, max_num_box = value[:7]
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
    opt.source_path = os.path.join(config.source_folder, TextSources)
    opt.random_color = (random_color == 1)
    opt.font_color = (0, 0, 0)
    opt.min_text_length = min_text_len
    opt.max_text_length = max_text_len
    opt.max_num_text = None
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

    st.warning("Begin running %s Method SynthText with folder %s " %(opt.method,Backgrounds))
    begin_time = time.time()
    results = []
    if ObjectSources == '0':
        # Just running white method with TextSources if ObjectSources does not exists
        opt.is_object = False
        opt.source_path = os.path.join(config.source_folder, TextSources)
        runner = BlackList(opt, out_name='black')
        output_path = runner.run()
        results.append(output_path)
        st.write("Time for this process was %s seconds" % int(time.time() - begin_time))
    elif TextSources == '0':
        # Just running white method with ObjectSources if TextSources does not exists
        opt.is_object = True
        opt.source_path = os.path.join(config.source_folder, ObjectSources)
        runner = BlackList(opt, out_name='black')
        output_path = runner.run()
        results.append(output_path)
        st.write("Time for this process was %s seconds" % int(time.time() - begin_time))
    else:
        # Running white method with both ObjectSources and TextSources
        opt.num_images = num_images//2
        opt.is_object = False
        opt.source_path = os.path.join(config.source_folder, TextSources)
        runner = BlackList(opt, out_name='black')
        output_path = runner.run()
        results.append(output_path)
        opt.num_images = num_images - opt.num_images
        opt.is_object = True
        opt.source_path = os.path.join(config.source_folder, ObjectSources)
        runner = BlackList(opt, out_name='black')
        output_path = runner.run()
        results.append(output_path)
        st.write("Time for this process was %s seconds" % int(time.time() - begin_time))
    return results
