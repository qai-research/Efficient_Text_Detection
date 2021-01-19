# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Wed December 30 14:48:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain methods build function
_____________________________________________________________________________
"""
from pathlib import Path

from utils.data.dataloader import LoadDatasetIterator
from models.detec.heatmap import HEAT
from models.recog.atten import Atten
from engine.config import setup, dict2namespace
from utils.utility import initial_logger

logger = initial_logger()


def build_dataloader(cfg, data, selected_data=None):
    """
    Build data iterator for training
    :param cfg: config Namespace
    :param data: source folder for dataloader
    :param selected_data: list target dataset's name want to load(inside root)
    :return: data iterator
    """
    data_path = Path(data)
    if selected_data is None:
        datalist = [str(dataset.name) for dataset in data_path.iterdir()]
    else:
        datalist = selected_data
    iterator = LoadDatasetIterator(cfg, data, selected_data=datalist)
    iterator = iter(iterator)
    return iterator


def build_model(config):
    """Build model from config"""
    if config._BASE_.MODEL_TYPE == "HEAT_BASE":
        model = HEAT()
    elif config._BASE_.MODEL_TYPE == "ATTEN_BASE":
        model = Atten(config)
    else:
        raise ValueError("invalid _BASE_.MODEL_TYPE in config file(accept HEAT_BASE and ATTEN_BASE)")
    model.to(device=config.SOLVER.DEVICE)
    return model
