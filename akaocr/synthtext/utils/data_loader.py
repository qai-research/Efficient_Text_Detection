#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Vu Hoang Viet - VietVH9
Created Date: NOV - 2020
Project : AkaOCR core
_____________________________________________________________________________



The Module Has Been Build To Have Progress
_____________________________________________________________________________
"""

import argparse

import lmdb
from PIL import Image
from pip._vendor import six
import numpy as np


# noinspection PyPep8Naming
class lmdb_dataset_loader:
    """
    Load image from lmdb dataset
    """

    def __init__(self, lmdb_path, batch_size=32):
        self.env = lmdb.open(lmdb_path,
                             max_readers=batch_size,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        self.cursor = self.env.begin(write=False)
        self.key_dict = [name for name in self.cursor.get("char-list".encode())]

    def random_sample(self, char):
        """
        Get random images of char
        @param char: str
        @return: image of char (PIL image)
        """
        hexchar = hex(ord(char))
        hexchar = str(hexchar)[2:].zfill(4)
        a = "%s_num" % hexchar
        _num = int(self.cursor.get(a.encode()))
        index = np.random.choice(range(_num))
        index = str(index).zfill(6)
        img_key = '{}_{}'.format(hexchar, index).encode()
        img_buf = self.cursor.get(img_key)
        buf = six.BytesIO()
        buf.write(img_buf)
        buf.seek(0)
        img = Image.open(buf)
        return img
