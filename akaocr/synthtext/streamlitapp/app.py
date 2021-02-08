#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Vu Hoang Viet - VietVH9
Created Date: NOV - 2020
Project : AkaOCR core
_____________________________________________________________________________



The Module Has Been Build For Running Streamlit App Flow
_____________________________________________________________________________
"""
import os
import io
import sys
import time
import argparse
import pandas as pd
import streamlit as st
from shutil import move as move_folder
from shutil import rmtree as remove_folder


def main():
    """
    The main function define the flow of streamlit app
    """

    # get abspath of synthtext and data_folder, add them to sys.path
    current_path = os.path.abspath('')
    tree = current_path.split("/")
    i = 1
    while True:
        data = os.path.join("/".join(tree[:-i]), 'data')
        i += 1
        if os.path.exists(data) and 'backgrounds' in os.listdir(data):
            break
    i = 1
    while True:
        ocr_path = os.path.join("/".join(tree[:-i]), 'akaocr')
        i += 1
        if os.path.exists(ocr_path) and 'synthtext' in os.listdir(ocr_path):
            break        
    css_path = os.path.join(ocr_path, 'synthtext/streamlitapp/style.css')
    background_folder = os.path.join(data, 'backgrounds')
    source_folder = os.path.join(data, 'sources')
    font_folder = os.path.join(data, 'fonts')
    object_folder = os.path.join(data, 'objects')
    outputs_folder = os.path.join(data, 'outputs')
    sys.path.append(ocr_path)
    # Add libary of synthtext app
    from synthtext.apps.white import whiteapp
    from synthtext.apps.black import blackapp
    from synthtext.apps.recog import recogapp
    from synthtext.apps.doubleblack import doubleblackapp
    from synthtext.utils.data_loader import lmdb_dataset_loader
    from synthtext.utils.utils_func import check_valid, get_all_valid
    from synthtext.utils.input_config_loader import InputConfigLoader 
    
    bg_df, source_df, font_df = get_all_valid(background_folder, source_folder, font_folder)

    st.markdown("<h3 style='text-align: left; color: Blue;'>Backgrounds List</h3>", unsafe_allow_html=True)
    empty_bg_df = st.empty()
    empty_bg_df.dataframe(bg_df)

    st.markdown("<h3 style='text-align: left; color: Blue;'>Text Source List</h3>", unsafe_allow_html=True)
    empty_src_df = st.empty()
    empty_src_df.dataframe(source_df)

    st.markdown("<h3 style='text-align: left; color: Blue;'>Fonts List</h3>", unsafe_allow_html=True)
    empty_font_df = st.empty()
    empty_font_df.dataframe(font_df)

    st.markdown("<h1 style='text-align: center; color: Blue;'>SynthText App</h1>", unsafe_allow_html=True)
    is_detect = (st.selectbox("Choose the purpose of synthtext", ['For Detect Training',
                                                                  'For Recog Training']) == 'For Detect Training'
                 )
    empty_upload = st.empty()
    file_buffer = empty_upload.file_uploader("UPLOAD CONFIG FILES")
    if file_buffer is not None:
        input_config_file = pd.read_csv(file_buffer)
        key = input_config_file.columns
        checked_df = config.check_valid(input_config_file, bg_df, source_df, font_df)
        key = key.insert(0, 'DETAIL')
        key = key.insert(0, 'STATUS')

        st.text("The uploaded config")
        st.dataframe(checked_df[key])
        outpath = st.text_input("Insert Output Name")
        removed = False
        input_config_dict  = {}
        for ind, values in enumerate(checked_df.values):
            input_config_dict[ind] = {k:v for k,v in zip(checked_df.columns, values)}  
        if outpath != '':
            output_path = os.path.join(config.data, 'outputs/%s' % outpath)
            if not removed and os.path.exists(output_path):
                empty1 = st.empty()
                empty2 = st.empty()
                empty1.warning(
                    "This directory already exists. Click REMOVE button to delete this directory,\\"
                    " or try with another name.")
                if empty2.button("REMOVE"):
                    remove_folder(output_path)
                    empty1.empty()
                    empty2.empty()
            if removed or not os.path.exists(output_path):
                empty1 = st.empty()
                empty2 = st.empty()
                empty1.write("The output with save at %s." % output_path)
                empty2.write("Press START for running synthtext app.")
                if st.button("START GEN"):
                    empty1.empty()
                    empty2.empty()
                    for index, value in enumerate(checked_df.values):

                        Method = value[0]
                        status = value[-2]
                        Backgrounds = value[5]
                        if status is "INVALID":
                            continue
                        begin_time = time.time()
                        st.warning("Begin running %s Method SynthText with folder %s " % (Method, Backgrounds))

                        if not is_detect:
                            local_output_path = recogapp(value)

                        elif Method == 'white':
                            local_output_path = whiteapp(value)

                        elif Method == 'black':
                            local_output_path = blackapp(value)

                        elif Method == 'double_black':
                            local_output_path = doubleblackapp(value)
                        else:
                            local_output_path = None

                        for path in local_output_path:
                            if not os.path.exists(output_path):
                                os.mkdir(output_path)
                            if path is not None:
                                move_folder(path, output_path)
                        st.write("Time for this process was %s seconds" % int(time.time() - begin_time))


if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)


    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


    local_css(config.css_path)
    main()
