import os
import io
import sys
import time
import argparse
import pandas as pd
import streamlit as st
from shutil import move as move_folder
from shutil import rmtree as remove_folder

current_path = os.path.abspath('')
tree = current_path.split("/")        
i = 1
while True:
    ocr_path = os.path.join("/".join(tree[:-i]), 'akaocr')
    i += 1
    if os.path.exists(ocr_path) and 'synthtext' in os.listdir(ocr_path):
        break        
sys.path.insert(0,ocr_path)
# Add libary of synthtext app
from synthtext.apps.white import whiteapp
from synthtext.apps.black import blackapp
from synthtext.apps.recog import recogapp
from synthtext.apps.doubleblack import doubleblackapp
from synthtext.utils.data_loader import lmdb_dataset_loader
from synthtext.utils.utils_func import check_valid, get_all_valid


if __name__ == "__main__":


        # get abspath of synthtext and data_folder, add them to sys.path

        parser = argparse.ArgumentParser()
        parser.add_argument("-d","--data_dir",
                            help="The data folder path",
                            type = str,
                            default = None)
        parser.add_argument("-s","--save_dir",
                            help="The output folder path",
                            type = str,
                            default = None)
        parser.add_argument("-r", "--is_recog", 
                            help="1 to generate the recog data, 0 to generate the detect data",
                            type = int,
                            default = 0)
        parser.add_argument("-i","--input_csv_path", 
                            help="The config csv path",
                            type=str)
        parser.add_argument("-f","--force_remove", 
                            help="1 to force remove the output path if exists, 0 to nothing",
                            type = int,
                            default = 1)                        
        args = parser.parse_args()
        args.background_folder = os.path.join(args.data_dir, 'backgrounds')
        args.source_folder = os.path.join(args.data_dir, 'sources')
        args.font_folder = os.path.join(args.data_dir, 'fonts')
        args.object_folder = os.path.join(args.data_dir, 'objects')
        args.outputs_folder = os.path.join(args.data_dir, 'outputs')
        bg_df, source_df, font_df = get_all_valid(args.background_folder, args.source_folder, args.font_folder)

        args.input_config_file = pd.read_csv(args.input_csv_path)
        key = args.input_config_file.columns
        checked_df = check_valid(args.input_config_file, bg_df, source_df, font_df)
        key = key.insert(0, 'DETAIL')
        key = key.insert(0, 'STATUS')
        removed = False
        # Convert input dataframe to dictionanry
        input_config_dict  = []
        for ind, values in enumerate(checked_df.values):
            input_config_dict.append({k:v for k,v in zip(checked_df.columns, values)}  )

        # Check out path and remove if existed
        if args.force_remove == 1 and os.path.exists(args.save_dir):
            remove_folder(args.save_dir)

        try:    
            os.mkdir(args.save_dir)
            for input_dict in input_config_dict:
                
                if input_dict['STATUS'] is "INVALID":
                    continue
                Method = input_dict['Method']
                if args.is_recog == 1:
                    print(args.is_recog)
                    local_output_path = recogapp(input_dict,args)

                elif Method == 'white':
                    local_output_path = whiteapp(input_dict,args)

                elif Method == 'black':
                    local_output_path = blackapp(input_dict,args)

                elif Method == 'double_black':
                    local_output_path = doubleblackapp(input_dict,args)
                else:
                    local_output_path = None
                for path in local_output_path:
                    if not os.path.exists(args.save_dir):
                        os.mkdir(args.save_dir)
                    if path is not None:
                        move_folder(path, args.save_dir)
        except FileExistsError:
            raise FileExistsError('The folder %s exists.\n Try "-f 1" or "--force_remove 1" to remove existed folder, or change the output_path'%args.save_dir)