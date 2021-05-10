import os
from pathlib import Path
import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm
from detectolmdb import process_data
# from lmdb_maker import create_dataset
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='path to the ground truth files')
    parser.add_argument('--output', required=True, help='path to the output folder')
    parser.add_argument('--expand', default=0.35, help='random expand boxes')
    parser.add_argument('--todetec', type=int, default=1, help='create detec lmdb')
    parser.add_argument('--torecog', type=int, default=1, help='create recog lmdb')

    parser.add_argument('--character', type=str, help='character label', default=None)
    parser.add_argument('--check_valid', action='store_true', help='whether to check valid images', default=True)
    parser.add_argument('--map_size', type=int, default=50073741824, help='maximum size database may grow to')

    opt = parser.parse_args()

    output_folder_path = Path(opt.output)
    if not output_folder_path.exists():
        output_folder_path.mkdir()
    input_folder_path = Path(opt.input)

    folder_list = [x for x in input_folder_path.iterdir() if x.is_dir()]
    for fo in folder_list:
        fo = Path(fo)
        imp = fo.joinpath('images')
        anp = fo.joinpath('annotations')
        if not imp.exists() or not anp.exists():
            continue
        print(f'processing {str(fo.name)}')
        out_name = fo.name.replace("ST_", "")
        # out_croplmdb = output_folder_path.joinpath(fo.name)
        # out_crop
        # if out_croplmdb.exists():
        #     print('pass', out_croplmdb.stem)
        #     continue
        process_data(input_folder=str(fo), name=out_name, todetec=opt.todetec, torecog=opt.torecog, output_folder=str(output_folder_path),
                    characters=opt.character, expand=opt.expand, db_map_size=opt.map_size)

if __name__ == '__main__':
    main()