import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data as data
import scipy.io as scio
import re
import argparse
import itertools
import random
from PIL import Image
import json
import six
import lmdb
import codecs
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
from tqdm import tqdm
import warnings

# import constants
from utils.data import lmdbproc


def read_lmdb(root):
    """
    Read lmdb dataset to variable
    :param root: path to lmdb database
    :return: env, length embedded in database
    """
    env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        warnings.warn(
            'cannot create lmdb from {}'.format(root)
        )
        sys.exit(0)

    with env.begin(write=False) as txn:
        try:
            num_samples = int(txn.get('num-samples'.encode()))
        except:
            num_samples = int(txn.stat()['entries']/2 - 1)
    return env, num_samples


def lmdb_loader(lmdb_file, index, rgb=False):
    """
    read binary image from lmdb file with custom key and convert to PIL image
    :param lmdb_file: cursor to lmdb dataset (file opened)
    :param index: index of item to be fetch
    :return:
    """
    label_key = 'label-{:09d}'.format(index).encode()
    label = lmdb_file.get(label_key)
    img_key = 'image-{:09d}'.format(index).encode()
    img_buf = lmdb_file.get(img_key)

    buf = six.BytesIO()
    buf.write(img_buf)
    buf.seek(0)
    try:
        if rgb:
            img = Image.open(buf).convert('RGB')  # for color image
        else:
            img = Image.open(buf).convert('L')
        return img, label
    except IOError:
        print(f'Corrupted image for {index}')
        # return None and ignore in dataloader
        return None


class LmdbDataset(Dataset):
    """
    Base loader for lmdb type dataset
    """
    def __init__(self, root, type_handler=None):
        """
        :param root: path to lmdb dataset
        :param label_handler: type of label processing
        """
        self.env, self.num_samples = read_lmdb(root)
        self.labelproc = lmdbproc.LabelHandler(type=type_handler)
        self.lmdb_file = self.env.begin(write=False)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        image, label = lmdb_loader(self.lmdb_file, index)
        return image, label
