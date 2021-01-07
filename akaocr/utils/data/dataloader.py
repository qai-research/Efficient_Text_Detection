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
        assert index <= len(self), 'index range error'
        image, label = self.lmdbreader.get_item(index+1)
        if self.labelproc is not None:
            label = self.labelproc(label)
        return image, label


class LoadDataset:
    def __init__(self, config, vocab=None):
        """
        Factory method to load dataset
        :param config: config name space
        :param vocab: path to the vocab file
        """
        # config = Constants(config)
        self.config = config
        self.vocab = vocab

    def load(self, root, load_type="recog", selected_data=None):
        """
        Load different type of dataset
        :param root: path to dataset root
        :param load_type: type of dataset to load
        :param selected_data: list of dataset to load multiple dataset
        :return: dataset object(s)
        """
        if load_type == "recog":
            return self._load_dataset_recog_ocr(root)
        elif load_type == "detec":
            return self._load_dataset_detec_heatmap(root)
        elif load_type == "mrecog":
            return self._load_multiple_dataset(root, selected_data=selected_data, type_dataset="recog")
        elif load_type == "mdetec":
            return self._load_multiple_dataset(root, selected_data=selected_data, type_dataset="detec")
        else:
            raise ValueError(f"invalid mode load_type : {load_type} in LoadDataset")

    def _load_dataset_recog_ocr(self, root):
        """
        load method for recognition data
        :param root: path to lmdb dataset
        :return: dataloader
        """
        try:
            chars = read_vocab(self.vocab)
        except TypeError:
            raise Exception(f"vocab path {self.vocab} not found")
        labelproc = label_handler.TextLableHandle(character=chars,
                                                  sensitive=self.config.MODEL.SENSITIVE,
                                                  unknown=self.config.SOLVER.UNKNOWN)
        try:
            dataset = LmdbDataset(root, rgb=self.config.MODEL.RGB, labelproc=labelproc)
        except Exception:
            logger.warning(f"can't read recog LMDB database from {root}")
            return None
        align_collate = collates.AlignCollate(img_h=int(self.config.MODEL.IMG_H), img_w=int(self.config.MODEL.IMG_W),
                                              keep_ratio_with_pad=self.config.MODEL.PAD)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=int(self.config.SOLVER.BATCH_SIZE),
            shuffle=True,
            num_workers=int(self.config.SOLVER.WORKERS),
            collate_fn=align_collate,
            pin_memory=True)
        return data_loader

    def _load_dataset_detec_heatmap(self, root):
        """
        load method for detection data
        :param root: path to lmdb dataset
        :return: dataloader
        """
        labelproc = label_handler.JsonLabelHandle()
        try:
            dataset = LmdbDataset(root, rgb=self.config.MODEL.RGB, labelproc=labelproc)
        except Exception:
            logger.warning(f"can't read detec LMDB database from {root}")
            return None

        gaussian_collate = collates.GaussianCollate(int(self.config.MODEL.MIN_SIZE), int(self.config.MODEL.MAX_SIZE))
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=int(self.config.SOLVER.BATCH_SIZE),
            shuffle=True,
            num_workers=int(self.config.SOLVER.WORKERS),
            collate_fn=gaussian_collate,
            pin_memory=True)
        return data_loader

    def _load_multiple_dataset(self, root, selected_data=None, type_dataset="recog"):
        """
        Wrapper to load multiple dataset
        :param root: path to dataset lake
        :param selected_data: list of selected data from lake
        :param type_dataset: type of dataset to load
        :return: list of datasets
        """
        root_path = Path(root)
        list_dataset = list()
        for dataset_name in selected_data:
            dataset_path = root_path.joinpath(dataset_name)
            if type_dataset == "recog":
                dataset = self._load_dataset_recog_ocr(str(dataset_path))
            elif type_dataset == "detec":
                dataset = self._load_dataset_detec_heatmap(str(dataset_path))
            else:
                raise ValueError(f"invalid mode type_dataset : {type_dataset} for _load_multiple_dataset")
            list_dataset.append(dataset)
        return list_dataset


class LoadDatasetIterator:
    def __init__(self, root, selected_data=None, load_type="recog", config=None, vocab=None):
        """
        Infinite iterator to load multiple dataset
        :param root: path to dataset lake
        :param selected_data: list of selected data from lake
        :param load_type: type of dataset to load
        :param config: config namespace
        :param load_type: type of dataset to load
        """
        root_path = Path(root)
        self.list_dataset = list()
        self.list_iterator = list()
        self.selected_data = selected_data
        loader = LoadDataset(config, vocab=vocab)
        for dataset_name in selected_data:
            dataset_path = root_path.joinpath(dataset_name)
            if load_type == "recog":
                dataset = loader._load_dataset_recog_ocr(str(dataset_path))
            elif load_type == "detec":
                dataset = loader._load_dataset_detec_heatmap(str(dataset_path))
            else:
                raise ValueError(f"invalid mode load_type : {load_type} for _load_multiple_dataset")
            if dataset is not None:
                self.list_dataset.append(dataset)
                self.list_iterator.append(iter(dataset))

    def __iter__(self):
        return self

    def __next__(self):
        idi = 0
        while True:
            if idi > len(self.list_iterator) - 1:
                idi = 0
            try:
                data_loader_iter = self.list_iterator[idi]
                data = data_loader_iter.next()
                return data
            except StopIteration:
                self.list_iterator[idi] = iter(self.list_dataset[idi])
                logger.info(f"finish on dataloader from {self.selected_data[idi]}")
            except ValueError:
                self.logger.warning(f"Getting data from dataloader failed")

