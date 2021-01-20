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


def white_handwriting(value):
    """
    Gen data with white method
    """
    Method, NumCores, Fonts, Backgrounds, ObjectSources, TextSources, num_images, max_num_box = value[:8]
    char_spacing, min_size, max_size, min_text_len, max_text_len, random_color = value[8:-7]
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
    opt.source_path = os.path.join(config.source_folder, str(TextSources))
    opt.random_color = (random_color == 1)
    opt.font_color = (0, 0, 0)
    opt.min_text_length = min_text_len
    opt.max_text_length = max_text_len
    opt.max_num_text = None
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
    results = []
    if str(ObjectSources) == '0':
        # Just running white method with TextSources if ObjectSources does not exists
        opt.is_object = False
        opt.source_path = os.path.join(config.source_folder, str(TextSources))
        runner = WhiteList(opt, out_name='white', num_cores=NumCores)
        output_path = runner.run()
        results.append(output_path)
    elif str(TextSources) == '0':
        # Just running white method with ObjectSources if TextSources does not exists
        opt.is_object = True
        opt.source_path = os.path.join(config.source_folder, str(ObjectSources))
        runner = WhiteList(opt, out_name='white', num_cores=NumCores)
        output_path = runner.run()
        results.append(output_path)
    else:
        # Running white method with both ObjectSources and TextSources
        opt.num_images = num_images // 2
        opt.is_object = False
        opt.source_path = os.path.join(config.source_folder, str(TextSources))
        runner = WhiteList(opt, out_name='white', num_cores=NumCores)
        output_path = runner.run()
        results.append(output_path)
        opt.num_images = num_images - opt.num_images
        opt.is_object = True
        opt.source_path = os.path.join(config.source_folder, str(ObjectSources))
        runner = WhiteList(opt, out_name='white', num_cores=NumCores)
        output_path = runner.run()
        results.append(output_path)

    return results


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
            local_output_path = doubleblack(value)
            for path in local_output_path:
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                if path is not None:
                    move_folder(path, opt.output_path)