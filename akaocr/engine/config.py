# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Tue December 29 13:00:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

Build initial folder tree and setup environment for running a training experiment
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
from utils.file_utils import read_vocab, copy_files
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


def parse_base():
    """
    Basic parse arguments.

    Returns: parse object to call or add more arguments

    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-e", "--exp", type=str, default="test", help="experiment name to save training result or "
                                                                      "model source for inference")
    parser.add_argument("-c", "--config", type=str, default=None, help="path to custom config define by user, this "
                                                                       "parameter will overwrite the default path to "
                                                                       "config(./data/config/config_base_v1.yaml for "
                                                                       "training the first time or overwrite the "
                                                                       "config in experiment folder for continue "
                                                                       "training or inference)")
    parser.add_argument("-w", "--weight", type=str, default=None, help="path to custom weight define by user, this "
                                                                       "parameter can be use to overwrite default "
                                                                       "path to weight in experiment folder or to "
                                                                       "provide pre-train model for training new "
                                                                       "model")
    parser.add_argument("-g", "--gpu", nargs="+", help="specify GPU or GPUs that train the model(Exp: -g 1, -g 0 2 4)")
    parser.add_argument("-d", "--data", type=str, default="../data", help="path to data path to get train data and save"
                                                                          "experiment results")

    return parser


def setup(tp="recog", args=None):
    """setup config environment and working space"""
    data_path = Path(args.data)

    if tp == "recog":
        config = str(data_path.joinpath("attention_resnet_base_v1.yaml"))
        exp_path = data_path.joinpath("exp_recog", args.exp)
    elif tp == "detec":
        config = str(data_path.joinpath("heatmap_1fpn_v1.yaml"))
        exp_path = data_path.joinpath("exp_detec", args.exp)
    else:
        logger.error("Training type not in [detec, recog]")
        raise ValueError("Invalid type for setup")

    backup_folder = exp_path.joinpath("backup")
    exp_config_path = str(exp_path.joinpath(args.exp + "_config.yaml"))

    lcf = list(exp_path.glob("*.yaml"))
    if len(lcf) > 0:
        config = lcf[0]

    checkpoint = None
    iteration = 0

    if not exp_path.exists():
        exp_path.mkdir(parents=True)
        # default config for EATS
        logger.info(f"Experiment {args.exp} do not exist.")
        logger.info("Creating new experiment folder")
        if args.config:
            config = args.config
            logger.warning(f"New config provided by flag -c/--config {config}")
        else:
            logger.info(f"Default config copy from {config}")
        shutil.copyfile(config, exp_config_path)
        if args.weight:
            checkpoint = args.weight
            logger.info(f"User checkpoint provided by flag -w/--weight {checkpoint}")
        else:
            logger.info(f"No checkpoint provided by -w/--weight.")
    else:
        logger.info(f"Experiment {args.exp} exist")
        logger.info(f"Use current experiment folder")
        exist_checkpoint = list(exp_path.glob("*.pth"))
        if args.weight:
            if len(exist_checkpoint) == 0:
                checkpoint = args.weight
            elif Path(args.weight).parent == exp_path:
                checkpoint_name = Path(args.weight).name
                copy_files(exist_checkpoint, backup_folder)
                checkpoint = str(backup_folder.joinpath(checkpoint_name))
            else:
                assert f"Weight exist for experiment folder: {str(exp_path)}. Please remove old " \
                       f"model weights before load new weight"
            logger.info(f"User checkpoint provided by flag -w/--weight {args.weight}")
        else:
            _, cp_name = get_exp_weight(str(exp_path))
            if cp_name is not None:
                copy_files(exist_checkpoint, backup_folder, move=True, clean_des=True)
                checkpoint = str(backup_folder.joinpath(cp_name))
        if args.config is not None:
            config = args.config
            assert not (
                    checkpoint is not None and not args.weight), f"Experiment {args.exp} and checkpoint exist so you " \
                                                                 f"can not use custom config "
            shutil.copyfile(config, exp_config_path)
            logger.warning(f"New config provided by flag -c/--config {config}")
            logger.warning(f"Overwrite current config in {str(exp_path)}")

    cfg = get_cfg_defaults()

    logger.info(f"Load config from : {config}")
    cfg.merge_from_file(config)

    cfg.SOLVER.START_ITER = iteration
    cfg.SOLVER.WEIGHT = checkpoint
    cfg.SOLVER.GPU = args.gpu
    cfg.SOLVER.DATA = args.data
    cfg.SOLVER.EXP = str(exp_path)
    if cfg.MODEL.VOCAB is not None:
        cfg.MODEL.VOCAB = os.path.join(cfg.SOLVER.DATA, "vocabs", cfg.MODEL.VOCAB)
    if cfg._BASE_.MODEL_TYPE == "ATTEN_BASE":
        cfg.SOLVER.DEVICE = str(device)
        if cfg.MODEL.VOCAB:  # vocabulary is given
            cfg.MODEL.VOCAB = read_vocab(cfg.MODEL.VOCAB)
            cfg["character"] = cfg.MODEL.VOCAB
        else:  # use character list instead
            cfg["character"] = list(cfg["character"])
        if cfg.SOLVER.UNKNOWN:
            cfg["character"].append(cfg.SOLVER.UNKNOWN)
        cfg["character"].sort()
        if 'CTC' in cfg.MODEL.PREDICTION:
            converter = CTCLabelConverter(cfg["character"])
        else:
            converter = AttnLabelConverter(cfg["character"], device=cfg.SOLVER.DEVICE)
        # cfg.MODEL.VOCAB = read_vocab(cfg.MODEL.VOCAB)
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


def get_exp_weight(exp_path):
    """Extract weight from experiment folder.

    If exist any checkpoint in the experiment folder this method will return the path to checkpoint.
    If multiple checkpoint exist => ask user for which checkpoint to load.
    Move all the exist checkpoint to experiment_folder/backup.

    Args:
        exp_path: Path to the current experiment folder

    Returns:
        None if do not found any checkpoint
        String path_to_checkpoint if exist checkpoint in experiment folder

    """
    exp_path = Path(exp_path)
    list_checkpoint = list(exp_path.glob("*.pth"))
    # list_checkpoint = sorted(list_checkpoint)
    if len(list_checkpoint) == 0:
        logger.info(f"No checkpoint found in experiment folder {str(exp_path)}")
        found_checkpoint = None
        cp_name = None
        return found_checkpoint, cp_name
    elif len(list_checkpoint) == 1:
        logger.info(f"Single checkpoint exist in folder {str(exp_path)}")
        cp_name = list_checkpoint[0].name
    else:
        logger.warning(f"Multiple checkpoint found in experiment folder {str(exp_path)}")
        logger.warning("Available option is : ")
        for cp in list_checkpoint:
            print("===> ", str(cp.name))
        print(f"Please enter the checkpoint name that you want to load :", end=" ")
        cp_name = input()

        # extract checkpoint of largest iteration
        if not cp_name:
            list_index = list()
            for cp in list_checkpoint:
                cp_index = ''.join([n for n in str(cp.name) if n.isdigit()])
                if cp_index.isdigit():
                    list_index.append(int(cp_index))
                else:
                    list_index.append(0)
            id_get = list_index.index(max(list_index))
            cp_name = list_checkpoint[id_get].name
            logger.warning(f"No checkpoint chosen. Get the checkpoint : " + cp_name)

    found_checkpoint = str(exp_path.joinpath(cp_name))
    return found_checkpoint, cp_name

def dict2namespace(di):
    for d in di:
        if type(di[d]) is dict:
            di[d] = Namespace(**di[d])
    di = Namespace(**di)
    return di
