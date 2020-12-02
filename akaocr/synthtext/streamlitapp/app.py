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

sys.path.append("/home/vietvh9/Project/OCR_Components/ocr-components/akaocr")
from synthtext.streamlitapp.apps.white_app import whiteapp
from synthtext.streamlitapp.apps.black_app import blackapp
from synthtext.streamlitapp.apps.doubleblack_app import doubleblackapp


def main():
    """
    The main function define the flow of streamlit app
    """
    st.markdown("<h1 style='text-align: center; color: Blue;'>SynthText App</h1>", unsafe_allow_html=True)

    # CREATE BACKGROUND DATAFRAME
    existed_background = sorted(
        [os.path.join(config.background_folder, name) for name in os.listdir(config.background_folder)])
    whitelist_background = [path for path in existed_background if
                            os.path.isdir(path) and 'anotations' in os.listdir(path)]
    blacklist_background = [path for path in existed_background if os.path.isdir(path)]
    bg_df = {"NAME": [],
             "METHOD": [],
             "SIZE": [],
             "PATH": []
             }

    for path in existed_background:

        if not len(os.listdir(path)) > 0:
            continue

        if path in blacklist_background:
            bg_df['NAME'].append(os.path.basename(path))
            bg_df['METHOD'].append('black')
            bg_df['SIZE'].append(len(os.listdir(path + "/images")))
            bg_df['PATH'].append(path)

        if path in whitelist_background:
            bg_df['NAME'].append(os.path.basename(path))
            bg_df['METHOD'].append('white')
            bg_df['SIZE'].append(len(os.listdir(path + "/images")))
            bg_df['PATH'].append(path)

    st.markdown("<h3 style='text-align: left; color: Blue;'>Backgrounds List</h3>", unsafe_allow_html=True)
    bg_df = pd.DataFrame(bg_df, columns=["NAME", "METHOD", "SIZE", "PATH"])
    empty_bg_df = st.empty()
    empty_bg_df.dataframe(bg_df)

    # CREATE SOURCE DATAFRAME
    existed_source = sorted([os.path.join(config.source_folder, name) for name in os.listdir(config.source_folder)])
    source_df = {"NAME": [],
                 "SIZE": [],
                 "PATH": [],
                 "TYPE": []
                 }
    for path in existed_source:
        if os.path.isfile(path) and path.endswith('.txt'):
            source_df["NAME"].append(os.path.basename(path))
            with open(path, 'r', encoding='utf8') as fr:
                source_df["SIZE"].append(len(fr.read().split("\n")))
            source_df["PATH"].append(path)
            source_df["TYPE"].append("Text")

        elif os.path.isdir(path) and 'images' in os.listdir(path):
            length = len(os.listdir(os.path.join(path, 'images')))
            source_df["NAME"].append(os.path.basename(path))
            source_df["SIZE"].append(length)
            source_df["PATH"].append(path)
            source_df["TYPE"].append("Object")

    st.markdown("<h3 style='text-align: left; color: Blue;'>Text Source List</h3>", unsafe_allow_html=True)
    source_df = pd.DataFrame(source_df, columns=["NAME", "TYPE", "SIZE", "PATH"])
    empty_src_df = st.empty()
    empty_src_df.dataframe(source_df)

    # CREATE FONT DATAFRAME
    existed_font = sorted([os.path.join(config.font_folder, name) for name in os.listdir(config.font_folder)])

    font_df = {"NAME": [],
               "SIZE": [],
               "PATH": []
               }
    for path in existed_font:
        if os.path.isdir(path):
            font_df["NAME"].append(os.path.basename(path))
            font_df["SIZE"].append(len(os.listdir(path)))
            font_df["PATH"].append(path)
    st.markdown("<h3 style='text-align: left; color: Blue;'>Fonts List</h3>", unsafe_allow_html=True)
    font_df = pd.DataFrame(font_df, columns=["NAME", "SIZE", "PATH"])
    empty_font_df = st.empty()
    empty_font_df.dataframe(font_df)

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

                        if Method == 'white':
                            local_output_path = whiteapp(value, source_df, out_name='white')

                        elif Method == 'black':
                            local_output_path = blackapp(value, source_df, out_name='black')

                        elif Method == 'double_black':
                            local_output_path = doubleblackapp(value, source_df, out_name='double_black')

                        else:
                            local_output_path = None

                        for path in local_output_path:
                            if not os.path.exists(output_path):
                                os.mkdir(output_path)
                            if path is not None:
                                move_folder(path, output_path)


def check_valid(dataframe, bg_df, source_df, fonts_df):
    """
    Check if the valid of input dataframe
    """
    df = dataframe.copy()
    results = {}
    for index, value in enumerate(dataframe.values):
        results[index] = {"Status": "valid", "Error": []}
        Method, Fonts, Backgrounds, ObjectSources, Textsources = value[:5]
        if Backgrounds not in bg_df['NAME'].values:
            results[index]['Error'].append('Invalid Backgrounds Folder')
        else:
            info = bg_df[bg_df['NAME'] == Backgrounds]
            st.dataframe(info)
            if Method == 'white' and 'white' not in info['METHOD'].values:
                results[index]['Error'].append('Invalid Method')
            if Fonts not in fonts_df['NAME'].values:
                results[index]['Error'].append('Fonts Folder Is Not Existed.')
            if Textsources != '0' and Textsources not in source_df['NAME'].values:
                results[index]['Error'].append('The TextSources Is Not Existed.')
            if ObjectSources != '0' and ObjectSources not in source_df['NAME'].values:
                results[index]['Error'].append('The ObjectSources Is Not Existed.')
        if len(results[index]['Error']) is not 0:
            results[index]["Status"] = "INVALID"
    df['STATUS'] = [results[i]["Status"] for i in range(len(results))]
    df['DETAIL'] = [results[i]["Error"] for i in range(len(results))]
    return df


if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)


    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


    local_css(config.css_path)
    main()
