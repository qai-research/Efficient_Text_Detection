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
import pandas as pd
import streamlit as st
from .white import whiteapp
from .black import blackapp
from .doubleblack import doubleblackapp
from shutil import rmtree as remove_folder
from synthtext.utils.utils_func import check_valid, get_all_valid


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
        if status is "INVALID":
            continue
        if Method == 'white':
            local_output_path = whiteapp(value)
        elif Method == 'black':
            local_output_path = blackapp(value)
        elif Method == 'doubleblack':
            local_output_path = doubleblackapp(value)
        else:
            continue

        for path in local_output_path:
            if not os.path.exists(opt.output_path):
                os.mkdir(opt.output_path)
            if path is not None:
                move_folder(path, opt.output_path)
