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
import config
import argparse
import pandas as pd
import streamlit as st
from shutil import move as move_folder
from shutil import rmtree as remove_folder


def main():
    """
    The main function define the flow of streamlit app
    """

    sys.path.append(config.ocr_path)
    from synthtext.apps.white_app import whiteapp
    from synthtext.apps.black_app import blackapp
    from synthtext.apps.doubleblack_app import doubleblackapp
    from synthtext.utils.data_loader import lmdb_dataset_loader
    from synthtext.utils.utils_func import check_valid, get_all_valid

    bg_df, source_df, font_df = get_all_valid(config)

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
    empty_upload = st.empty()
    file_buffer = empty_upload.file_uploader("UPLOAD CONFIG FILES")
    if file_buffer is not None:
        config_file = pd.read_csv(file_buffer)
        key = config_file.columns
        checked_df = check_valid(config_file, bg_df, source_df, font_df)
        key = key.insert(0, 'DETAIL')
        key = key.insert(0, 'STATUS')

        st.text("The uploaded config")
        st.dataframe(checked_df[key])
        outpath = st.text_input("Insert Output Path")
        removed = False
        if outpath != '':
            output_path = os.path.join(config.data, 'outputs/%s' % outpath)
            if not removed:
                if os.path.exists(output_path):
                    empty1 = st.empty()
                    empty2 = st.empty()
                    empty1.warning(
                        "This directory already exists. Click submit to delete this, or try with another name.")
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
                        if status is "INVALID":
                            continue
                        begin_time = time.time()
                        st.warning("Begin running %s Method SynthText with folder %s " % (opt.method, Backgrounds))

                        if Method == 'white':
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
