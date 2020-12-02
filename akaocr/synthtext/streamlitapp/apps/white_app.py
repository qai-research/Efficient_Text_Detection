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


def whiteapp(value, source_df, out_name='white'):
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
    opt.input_json = os.path.join(config.background_folder, Backgrounds, 'anotations')

    st.warning("Begin running %s Method SynthText with %s folder" %(opt.method,Backgrounds))
    begin_time = time.time()
    results = []
    if ObjectSources == '0':
        # Just running white method with TextSources if ObjectSources does not exists
        opt.is_object = False
        opt.source_path = os.path.join(config.source_folder, TextSources)
        runner = WhiteList(opt, out_name='white')
        output_path = runner.run()
        results.append(output_path)
        st.write("Time for this process was %s seconds" % int(time.time() - begin_time))
    elif TextSources == '0':
        # Just running white method with ObjectSources if TextSources does not exists
        opt.is_object = True
        opt.source_path = os.path.join(config.source_folder, ObjectSources)
        runner = WhiteList(opt, out_name='white')
        output_path = runner.run()
        results.append(output_path)
        st.write("Time for this process was %s seconds" % int(time.time() - begin_time))
    else:
        # Running white method with both ObjectSources and TextSources
        opt.num_images = num_images//2
        opt.is_object = False
        opt.source_path = os.path.join(config.source_folder, TextSources)
        runner = WhiteList(opt, out_name='white')
        output_path = runner.run()
        results.append(output_path)
        opt.num_images = num_images - opt.num_images
        opt.is_object = True
        opt.source_path = os.path.join(config.source_folder, ObjectSources)
        runner = WhiteList(opt, out_name='white')
        output_path = runner.run()
        results.append(output_path)
        st.write("Time for this process was %s seconds" % int(time.time() - begin_time))

    return results
