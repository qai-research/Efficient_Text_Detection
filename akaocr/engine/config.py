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

import yaml
import shutil
import argparse
from pathlib import Path
from argparse import Namespace

from utils.utility import initial_logger

logger = initial_logger()


def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.full_load(f)

    return config


def parse_args(add_help=True):
    def str2bool(v):
        return v.lower() in ("True", "true", "t", "1")

    parser = argparse.ArgumentParser(add_help=add_help)
    # params for prediction engine
    parser.add_argument("-e", "--exp", type=str, default="test")
    parser.add_argument("-c", "--continue_train", type=str2bool, default=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("-w", "--weight", type=str, default=None)
    parser.add_argument('-g', '--gpu', nargs='+')
    parser.add_argument("--data", type=str, default="../data")
    return parser.parse_args()


def setup(tp="recog"):
    args = parse_args()
    data_path = Path(args.data)
    exp_exist = True
    if tp == "recog":
        config = "../data/attention_resnet_base_v1.yaml"
        exp_path = data_path.joinpath("exp_recog", args.exp)
        if not exp_path.exists():
            logger.info(f"Experiment {args.exp} do not exist.")
            logger.info("Creating new experiment folder")
            exp_exist = False
            exp_path.mkdir(exist_ok=True)
            exp_config_path = str(exp_path.joinpath(args.exp + "_detec_config.yaml"))
            shutil.copyfile(config, exp_config_path)
        else:
            logger.info(f"Experiment {args.exp} exist")
            logger.info(f"Use current experiment folder")

    elif tp == "detec":
        config = "../data/heatmap_1fpn_v1.yaml"
        exp_path = data_path.joinpath("exp_detec", args.exp)
        if not exp_path.exists():
            logger.info(f"Experiment {args.exp} do not exist")
            logger.info("Creating new experiment folder")
            exp_exist = False
            exp_path.mkdir(exist_ok=True)
            exp_config_path = str(exp_path.joinpath(args.exp + "_recog_config.yaml"))
            shutil.copyfile(config, exp_config_path)
        else:
            logger.info(f"Experiment {args.exp} exist")
            logger.info(f"Use current experiment folder")
    else:
        logger.error("Training type not in [detec, recog]")
        raise ValueError("Invalid type for setup")

    if args.config is not None:
        config = args.config
        assert not exp_exist, f"Experiment {args.exp} exist so you can not use custom config"
        shutil.copyfile(config, exp_config_path)
    #########################################

    config_data = load_yaml_config(config)
    model_path, iteration = get_model_weight(str(exp_path))

    if args.weight:
        if model_path is None or Path(args.weight).parent == exp_path:
            model_path = args.weight
            iteration = 0
        else:
            assert not model_path, f"Weight exist for experiment folder: {str(exp_path)}. Please remove old " \
                                   f"model weight before load new weight"
    config_data["SOLVER"]["START_ITER"] = iteration
    config_data["SOLVER"]["WEIGHT"] = model_path
    config_data["SOLVER"]["GPU"] = args.gpu
    config_data["SOLVER"]["DATA"] = args.data
    config_data["SOLVER"]["EXP"] = args.exp
    config_data = dict2namespace(config_data)
    return config_data


def get_model_weight(exp_path, cp_type="latest"):
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

    if cp_type == "best":
        if best_path is None:
            logger.warning(f"Best weight not found in {str(exp_path)}. Use latest checkpoint instead")
        else:
            return str(best_path), 0
    # if not return best path
    if len(list_cp) == 0:
        logger.info(f"No weight found in {str(exp_path)}")
        return None
    max_iter = max(list_cp)
    latest_model = exp_path.joinpath("iter_"+str(max_iter) + ".pth")
    logger.info(f"Latest checkpoints {str(latest_model)} found")
    return str(latest_model), max_iter


def dict2namespace(di):
    for d in di:
        if type(di[d]) is dict:
            di[d] = Namespace(**di[d])
    di = Namespace(**di)
    return di
