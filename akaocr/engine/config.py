# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Tue December 29 13:00:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain config methods
_____________________________________________________________________________
"""

import os
import yaml
import shutil
import argparse
from pathlib import Path
from argparse import Namespace
import torch
from engine.solver.default import _C
from utils.utility import initial_logger
logger = initial_logger()
from utils.file_utils import read_vocab
from models.modules.converters import AttnLabelConverter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.full_load(f)
    return config


def parse_args(add_help=True):
    parser = argparse.ArgumentParser(add_help=add_help)
    # params for prediction engine
    parser.add_argument("-e", "--exp", type=str, default="test")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("-w", "--weight", type=str, default=None)
    parser.add_argument("-g", "--gpu", nargs="+")
    parser.add_argument("--data", type=str, default="../data")
    return parser.parse_args()


def setup(tp="recog"):
    """setup config environment and working space"""
    args = parse_args()
    data_path = Path(args.data)
    exp_exist = True
    if tp == "recog":
        exp_path = data_path.joinpath("exp_recog", args.exp)
        exp_config_path = str(exp_path.joinpath(args.exp + "_detec_config.yaml"))
        if not exp_path.exists():
            config = "../data/attention_resnet_base_v1.yaml"
            logger.warning(f"Experiment {args.exp} do not exist.")
            logger.warning("Creating new experiment folder")
            exp_exist = False
            exp_path.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config, exp_config_path)
        else:
            config = exp_config_path
            logger.warning(f"Experiment {args.exp} exist")
            logger.warning(f"Use current experiment folder")

    elif tp == "detec":
        exp_path = data_path.joinpath("exp_detec", args.exp)
        exp_config_path = str(exp_path.joinpath(args.exp + "_recog_config.yaml"))
        if not exp_path.exists():
            config = "../data/heatmap_1fpn_v1.yaml"
            logger.info(f"Experiment {args.exp} do not exist")
            logger.info("Creating new experiment folder")
            exp_exist = False
            exp_path.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config, exp_config_path)
        else:
            config = exp_config_path
            logger.info(f"Experiment {args.exp} exist")
            logger.info(f"Use current experiment folder")
    else:
        logger.error("Training type not in [detec, recog]")
        raise ValueError("Invalid type for setup")

    if exp_exist:
        model_path, iteration = get_model_weight(str(exp_path))
    else:
        model_path, iteration = None, 0

    if args.config is not None:
        config = args.config
        assert not (
                    exp_exist and model_path is not None), f"Experiment {args.exp} exist so you can not use custom config"
        shutil.copyfile(config, exp_config_path)
    #########################################

    if args.weight:
        if model_path is None or Path(args.weight).parent == exp_path:
            model_path = args.weight
            iteration = 0
        else:
            assert not model_path, f"Weight exist for experiment folder: {str(exp_path)}. Please remove old " \
                                   f"model weights before load new weight"

    cfg = get_cfg_defaults()

    logger.info(f"Load config from : {config}")
    cfg.merge_from_file(config)

    cfg.SOLVER.START_ITER = iteration
    cfg.SOLVER.WEIGHT = model_path
    cfg.SOLVER.GPU = args.gpu
    cfg.SOLVER.DATA = args.data
    cfg.SOLVER.EXP = str(exp_path)
    if cfg.MODEL.VOCAB is not None:
        cfg.MODEL.VOCAB = os.path.join(cfg.SOLVER.DATA, "vocabs", cfg.MODEL.VOCAB)   
    if cfg._BASE_.MODEL_TYPE == "ATTEN_BASE":
        cfg.SOLVER.DEVICE = str(device)
        if cfg.MODEL.VOCAB:  # vocabulary is given
            with open(cfg.MODEL.VOCAB, "r", encoding='utf-8-sig') as f:   
                chars = f.read().strip().split("\n")
                cfg["character"] = chars
        else:  # use character list instead
            cfg["character"] = list(cfg["character"])
        if cfg.SOLVER.UNKNOWN:
            cfg["character"].append(cfg.SOLVER.UNKNOWN)
        cfg["character"].sort()
        if 'CTC' in cfg.MODEL.PREDICTION:
            converter = CTCLabelConverter(cfg["character"])
        else:
            converter = AttnLabelConverter(cfg["character"], device=cfg.SOLVER.DEVICE)
        cfg.MODEL.VOCAB = read_vocab(cfg.MODEL.VOCAB)
        cfg.MODEL.NUM_CLASS = len(converter.character)
    return cfg


def get_model_weight(exp_path, cp_type="best"):
    """
    Extract weight from experiment folder
    :param exp_path: experiment path
    :param cp_type: type of weight to extract (best weigth or last weight)
    :return:
    """
    exp_path = Path(exp_path)
    best_path = None
    list_cp = list()
    for cp in exp_path.glob("*.pth"):
        if cp.name == "best_accuracy.pth":
            best_path = cp
        else:
            cpid = cp.name.replace(".pth", "")
            cpid = cpid.replace("iter_", "")
            if cpid.isnumeric():
                list_cp.append(int(cpid))

    if len(list_cp) == 0:
        logger.info(f"No weight found in {str(exp_path)}")
        return None, 0
    max_iter = max(list_cp)

    if cp_type == "best":
        if best_path is None:
            logger.warning(f"Best weight not found in {str(exp_path)}. Use latest checkpoint instead")
        else:
            logger.info(f"Best checkpoint {str(best_path)} found")
            return str(best_path), max_iter

    # if not return best path
    latest_model = exp_path.joinpath("iter_" + str(max_iter) + ".pth")
    logger.info(f"Latest checkpoints {str(latest_model)} found")
    return str(latest_model), max_iter

def dict2namespace(di):
    for d in di:
        if type(di[d]) is dict:
            di[d] = Namespace(**di[d])
    di = Namespace(**di)
    return di
