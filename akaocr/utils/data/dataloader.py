# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Mon November 03 10:00:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain base loader for lmdb type data + wrapper
_____________________________________________________________________________
"""

import torch
import logging
from pathlib import Path
import random
from torch.utils.data import Dataset, ConcatDataset, Subset

import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import copy
from utils.file_utils import LmdbReader
from utils.file_utils import Constants, read_vocab
from utils.data import collates, label_handler
from utils.runtime import Color, colorize
from utils.utility import initial_logger
logger = initial_logger()


class LmdbDataset(Dataset):
    """
    Base loader for lmdb type dataset
    """
    def __init__(self, root, rgb=False, labelproc=None):
        """
        :param root: path to lmdb dataset
        :param rgb: process color image
        :param label_handler: type of label processing
        """
        if labelproc is None:
            logger.warning(f"You don\'t have label handler for loading {root}")
        logger.info(f"load dataset from : {root}")

        self.labelproc = labelproc
        self.lmdbreader = LmdbReader(root, rgb)

    def __len__(self):
        return self.lmdbreader.num_samples

    def __getitem__(self, index):
        index += 1
        assert index <= len(self), 'index range error'
        image, label = self.lmdbreader.get_item(index)
        if self.labelproc is not None:
<<<<<<< akaocr/utils/data/dataloader.py
            label = self.labelproc(label)
=======
            label = self.labelproc.process_label(label)
<<<<<<< HEAD
>>>>>>> akaocr/utils/data/dataloader.py
        return image, label
<<<<<<< akaocr/utils/data/dataloader.py
<<<<<<< HEAD

def get_seg(path):
    f = h5py.File(path, 'r')
    for i in range(len(list(f['mask'].keys()))):
        name = list(f['mask'].keys())[i]
        val = np.array(f['mask'][name])
        img_name = name.split('.')[0] + '_seg.jpg'
        plt.imsave(img_name, val)
        l=[]
        m = np.max(val)
        for j in range(m):
            num = sum(sum(val==j+1))
            if num >2000:     
                tem = copy.deepcopy(val)
                tem[tem!=j] = 0
                yield tem
=======
>>>>>>> d63f4ed173251eaede0c7c790bf34d09a3712175
=======
        return image, label
>>>>>>> 983e19d5ce27d61d5d708f2b893d039a924fe13d
=======


class LoadDataset:
    def __init__(self, cfg, vocab=None):
        """
        Method to load dataset
        :param cfg: config name space
        :param vocab: path to the vocab file
        """
        self.cfg = cfg
        self.vocab = vocab

    def load_dataset_recog_ocr(self, root):
        """
        load method for recognition data
        :param root: path to lmdb dataset
        :return: dataloader
        """
        # try:
        #     chars = read_vocab(self.vocab)
        # except TypeError:
        #     raise Exception(f"vocab path {self.vocab} not found")
        chars = self.vocab
        labelproc = label_handler.TextLableHandle(character=chars,
                                                  sensitive=self.cfg.MODEL.SENSITIVE,
                                                  unknown=self.cfg.SOLVER.UNKNOWN)
        try:
            dataset = LmdbDataset(root, rgb=self.cfg.MODEL.RGB, labelproc=labelproc)
        except Exception:
            logger.warning(f"can't read recog LMDB database from {root}")
            return None
        align_collate = collates.AlignCollate(img_h=int(self.cfg.MODEL.IMG_H), img_w=int(self.cfg.MODEL.IMG_W),
                                              keep_ratio_with_pad=self.cfg.MODEL.PAD)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=int(self.cfg.SOLVER.BATCH_SIZE),
            shuffle=True,
            num_workers=int(self.cfg.SOLVER.WORKERS),
            collate_fn=align_collate,
            pin_memory=True)
        return data_loader

    def load_dataset_detec_heatmap(self, root):
        """
        load method for detection data
        :param root: path to lmdb dataset
        :return: dataloader
        """
        labelproc = label_handler.JsonLabelHandle()
        try:
            dataset = LmdbDataset(root, rgb=self.cfg.MODEL.RGB, labelproc=labelproc)
        except Exception:
            logger.warning(f"can't read detec LMDB database from {root}")
            return None

        gaussian_collate = collates.GaussianCollate(self.cfg.MODEL.MIN_SIZE, self.cfg.MODEL.MAX_SIZE)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=int(self.cfg.SOLVER.BATCH_SIZE),
            shuffle=True,
            num_workers=int(self.cfg.SOLVER.WORKERS),
            collate_fn=gaussian_collate,
            pin_memory=True)
        return data_loader

    def _load_multiple_dataset(self, cfg, selected_data=None):
        """
        Wrapper to load multiple dataset
        :param cfg: config namespace
        :param selected_data: list of selected data from lake
        :return: list of datasets
        """
        root_path = Path(cfg.SOLVER.DATA_SOURCE)
        list_dataset = list()
        for dataset_name in selected_data:
            dataset_path = root_path.joinpath(dataset_name)
            if cfg._BASE_.MODEL_TYPE == "ATTEN_BASE":
                dataset = self.load_dataset_recog_ocr(str(dataset_path))
            elif cfg._BASE_.MODEL_TYPE == "HEAT_BASE":
                dataset = self.load_dataset_detec_heatmap(str(dataset_path))
            else:
                raise ValueError(f"invalid mode type_dataset : {cfg._BASE_.MODEL_TYPE} for _load_multiple_dataset")
            list_dataset.append(dataset)
        return list_dataset


<<<<<<< akaocr/utils/data/dataloader.py
class LoadDatasetIterator:
    def __init__(self, cfg, data, selected_data=None):
        """
        Infinite iterator to load multiple dataset
        :param cfg: config namespace
        :param selected_data: list of selected data from lake
        """
        root_path = Path(data)
        self.idi = 0
        self.list_dataset = list()
        self.list_iterator = list()
        self.filled_selected_data = list()
        loader = LoadDataset(cfg, vocab=cfg.MODEL.VOCAB)
        for dataset_name in selected_data:
            dataset_path = root_path.joinpath(dataset_name)
            if cfg._BASE_.MODEL_TYPE == "ATTEN_BASE":
                dataset = loader.load_dataset_recog_ocr(str(dataset_path))
            elif cfg._BASE_.MODEL_TYPE == "HEAT_BASE":
                dataset = loader.load_dataset_detec_heatmap(str(dataset_path))
            else:
                raise ValueError(f"invalid model type : {cfg._BASE_.MODEL_TYPE} in config")
            if dataset is not None:
                self.list_dataset.append(dataset)
                self.list_iterator.append(iter(dataset))
                self.filled_selected_data.append(dataset_name)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.idi > len(self.list_iterator) - 1:
                self.idi = 0
            try:
                data_loader_iter = self.list_iterator[self.idi]
                data = data_loader_iter.next()
                self.idi += 1
                return data
            except StopIteration:
                self.list_iterator[self.idi] = iter(self.list_dataset[self.idi])
                logger.info(f"finish on dataloader from {self.filled_selected_data[self.idi]}")
            except ValueError:
                self.logger.warning(f"Getting data from dataloader failed")

=======
>>>>>>> akaocr/utils/data/dataloader.py
>>>>>>> akaocr/utils/data/dataloader.py
