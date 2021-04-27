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

import json
import random
import torch
import logging
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset, ConcatDataset, Subset
from utils.file_utils import LmdbReader
from utils.file_utils import Constants, read_vocab
from utils.data import collates, label_handler
from utils.runtime import Color, colorize
from utils.utility import initial_logger
from utils.augmentation import Augmentation

logger = initial_logger()


class LmdbDataset(Dataset):
    """
    Base loader for lmdb type dataset
    """

    def __init__(self, root, rgb=False, labelproc=None, augmentation=None):
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
        self.augmentation = augmentation

    def __len__(self):
        return self.lmdbreader.num_samples

    def __getitem__(self, index):
        index += 1
        assert index <= len(self), 'index range error'
        image, label = self.lmdbreader.get_item(index)
        if self.labelproc is not None:
            label = self.labelproc(label)
        if label is None:
            return None
        if self.augmentation is not None:
            image, label = self.augmentation.augment([image], [label])
        return image, label


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
        chars = self.vocab
        labelproc = label_handler.TextLableHandle(character=chars,
                                                  sensitive=self.cfg.MODEL.SENSITIVE,
                                                  unknown=self.cfg.SOLVER.UNKNOWN,
                                                  max_length=self.cfg.MODEL.MAX_LABEL_LENGTH)
        try:
            dataset = LmdbDataset(root, rgb=self.cfg.MODEL.RGB, labelproc=labelproc, augmentation=None)
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
        option = {'shear': {'p': 0.8, 'v': {'x': (-15, 15), 'y': (-15, 15)}},
                  'scale': {'p': 0.8, 'v': {"x": (0.8, 1.2), "y": (0.8, 1.2)}},
                  'translate': {'p': 0.8, 'v': {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
                  'rotate': {'p': 0.8, 'v': (-45, 45)},
                  'dropout': {'p': 0.6, 'v': (0.0, 0.5)},
                  'blur': {'p': 0.6, 'v': (0.0, 2.0)},
                  'elastic': {'p': 0.85}}
        augmentation = Augmentation(self.cfg, option=option)
        try:
            dataset = LmdbDataset(root, rgb=self.cfg.MODEL.RGB, labelproc=labelproc, augmentation=augmentation)
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
                logger.debug(len(self.list_iterator))
                logger.debug(self.idi)
                data_loader_iter = self.list_iterator[self.idi]
                data = data_loader_iter.next()
                self.idi += 1
                return data
            except StopIteration:
                self.list_iterator[self.idi] = iter(self.list_dataset[self.idi])
                logger.info(f"exhaust dataloader from {self.filled_selected_data[self.idi]} : reload")
            except ValueError:
                logger.warning(f"Getting data from dataloader failed")


class LoadDatasetDetecBBox():
    def __init__(self, data, cfg):
        """
        Load detection data with bouding boxes label
        Args:
            data: path to LMDB data source
            cfg: config name space
        """
        self.lmdbreader = LmdbReader(data, rgb=cfg.MODEL.RGB)
        self.index_list = random.sample(range(1, self.get_length() + 1), self.get_length())

    def get_length(self):
        return self.lmdbreader.num_samples

    def get_item(self, index):
        img, label = self.lmdbreader.get_item(self.index_list[index])
        img = np.array(img)
        label = json.loads(label)
        return img, label
