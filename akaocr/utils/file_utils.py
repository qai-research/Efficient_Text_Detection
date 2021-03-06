#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Mon November 03 10:00:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain read, write func of various data type
_____________________________________________________________________________
"""

import six
import lmdb
import json
import shutil
import configparser
import numpy as np
from pathlib import Path
from PIL import Image
from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathHandler
from iopath.common.file_io import PathManager as PathManagerBase

from utils.utility import initial_logger
logger = initial_logger()


PathManager = PathManagerBase()
"""
This is a detectron2 project-specific PathManager.
We try to stay away from global PathManager in fvcore as it
introduces potential conflicts among other libraries.
"""


class CustomHandler(PathHandler):
    """borrow handler from detectron2"""

    # PREFIX = "detectron2://"
    # S3_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"
    PREFIX = "dump"
    S3_PREFIX = "dump"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.S3_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


# PathManager.register_handler(HTTPURLHandler())
# PathManager.register_handler(OneDrivePathHandler())
PathManager.register_handler(CustomHandler())


class Constants:
    def __init__(self, config_path):
        """
        Read config to a namespace
        :param config_path: path to the config file
        """
        config_p = Path(config_path)
        if not config_p.is_file():
            raise ValueError("Config file not found")
        config_parser = configparser.ConfigParser()
        config_parser.read(config_path)
        self.config = config_parser['DEFAULT']
        print('loaded config from : ', str(config_p.resolve()))


def read_vocab(file_path):
    """
    Read vocab to list of character
    :param file_path: path to vocab file
    :return: list of character
    """
    with open(file_path, "r", encoding='utf-8-sig') as f:
        list_lines = f.read().strip().split("\n")
    return list_lines


def read_json_annotation(js_label):
    """
    Read word and charboxes from json
    :param js_label: json variable
    :return: word_list, charbox_list
    """
    js_words = js_label["words"]
    character_boxes = []
    words = []
    for tex in js_words:  # loop through each word
        bboxes = []
        words.append(tex['text'])
        for ch in tex['chars']:
            bo = [
                [ch['x1'], ch['y1']],
                [ch['x2'], ch['y2']],
                [ch['x3'], ch['y3']],
                [ch['x4'], ch['y4']]
            ]
            bo = np.array(bo)
            bboxes.append(bo)
        character_boxes.append(np.array(bboxes))
    return words, character_boxes


def copy_files(src, des, move=False, clean_des=False):
    """Copy or move file(or list of file) of Path object(from pathlib)

    Args:
        src: source file(or list of source file)
        des: destination folder
        move: move file or copy file
        clean_des: clean destination folder or not

    """
    if clean_des:
        shutil.rmtree(des, ignore_errors=True)
    des_path = Path(des)
    des_path.mkdir(exist_ok=True, parents=True)

    if isinstance(src, list):
        if move:
            for file in src:
                shutil.move(str(file), str(des_path.joinpath(file.name)))
        else:
            for file in src:
                shutil.copyfile(str(file), str(des_path.joinpath(file.name)))
    else:
        if move:
            shutil.move(str(src), str(des_path.joinpath(src.name)))
        else:
            shutil.copyfile(str(src), str(des_path.joinpath(src.name)))


class LmdbReader:
    def __init__(self, root, rgb=False, get_func=None):
        """
        Read lmdb dataset to variable
        :param root: path to lmdb database
        :param rgb: to use color image
        """
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)

        self.cursor = self.env.begin(write=False)
        try:
            self.num_samples = int(self.cursor.get('num-samples'.encode()))
        except KeyError:
            self.num_samples = int(self.cursor.stat()['entries'] / 2 - 1)
        self.rgb = rgb
        self.get_func = get_func

    def lmdb_loader(self, index):
        """
        read binary image from lmdb file with custom key and convert to PIL image
        :param index: index of item to be fetch
        :return: Pillow image, string label
        """
        label_key = 'label-{:09d}'.format(index).encode()
        label = self.cursor.get(label_key).decode('utf-8')
        img_key = 'image-{:09d}'.format(index).encode()
        img_buf = self.cursor.get(img_key)

        buf = six.BytesIO()
        buf.write(img_buf)
        buf.seek(0)
        try:
            if self.rgb:
                img = Image.open(buf).convert('RGB')  # for color image
            else:
                img = Image.open(buf).convert('L')
            return img, label
        except IOError:
            logger.warning(f'Corrupted image for {index}')
            # return None and ignore in dataloader
            return None

    def get_item(self, index):
        if self.get_func is None:
            return self.lmdb_loader(index)
        else:
            logger.error("get func type on found")
