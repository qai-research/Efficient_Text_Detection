# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Mon November 03 10:00:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain unit test for dataloader
_____________________________________________________________________________
"""
import argparse
import sys
import torch
import cv2
from pathlib import Path

sys.path.append("../")
from utils.data.dataloader import LmdbDataset, LoadDataset, LoadDatasetIterator
from utils.file_utils import Constants, read_vocab
from utils.data import collates, label_handler
from utils.visproc import save_heatmap
from pre.image import ImageProc
from engine.config import setup, parse_base
from engine.build import build_dataloader


def test_dataloader_detec(root, config_path, image_name="demo_single_dataset_load"):
    """Test LmdbDataset for detec"""
    constants = Constants(config_path)
    constants = constants.config
    # labelproc = collates.LabelHandler(label_type=label_type)
    labelproc = label_handler.JsonLabelHandle()

    _dataset = LmdbDataset(root, rgb=constants.getboolean('rgb'), labelproc=labelproc)

    _gaussian_collate = collates.GaussianCollate(int(constants["min_size"]), int(constants["max_size"]))
    _data_loader = torch.utils.data.DataLoader(
        _dataset, batch_size=int(constants['batch_size']),
        shuffle=True,
        num_workers=int(constants['workers']),
        collate_fn=_gaussian_collate,
        pin_memory=True)

    iterator = iter(_data_loader)
    x = iterator.next()
    save_vis_heatmap_from_model(x, image_name=image_name)


def test_dataloader_recog(root, config_path, vocab=None):
    """Test LmdbDataset for recog"""
    constants = Constants(config_path)
    constants = constants.config
    chars = read_vocab(vocab)
    labelproc = label_handler.TextLableHandle(character=chars,
                                              sensitive=constants.getboolean('sensitive'),
                                              unknown=constants["unknown"])

    _dataset = LmdbDataset(root, rgb=constants.getboolean('rgb'), labelproc=labelproc)

    _align_collate = collates.AlignCollate(img_h=int(constants['img_h']), img_w=int(constants['img_w']),
                                           keep_ratio_with_pad=constants.getboolean('pad'))

    _data_loader = torch.utils.data.DataLoader(
        _dataset, batch_size=int(constants['batch_size']),
        shuffle=True,
        num_workers=int(constants['workers']),
        collate_fn=_align_collate,
        pin_memory=True)

    iterator = iter(_data_loader)
    x = iterator.next()
    print(x[1])


def test_load_dataset(root, config_path, load_type="recog", vocab=None):
    """test wrapper for load data method"""
    data_path = Path(root, vocab=vocab)
    datalist = [str(dataset.name) for dataset in data_path.iterdir()]
    loader = LoadDataset(config_path=config_path, vocab=vocab)

    dataset = loader.load(root, load_type=load_type, selected_data=datalist)
    print(dataset)
    return dataset


def test_load_iterator(root, config_path, load_type="recog", vocab=None):
    data_path = Path(root, vocab=vocab)
    datalist = [str(dataset.name) for dataset in data_path.iterdir()]

    iterator = LoadDatasetIterator(root, selected_data=datalist, load_type=load_type,
                                   config_path=config_path, vocab=vocab)
    print(iterator)
    iterator = iter(iterator)
    print(iterator)
    data = next(iterator)
    if load_type == "recog":
        print(data[1])
    elif load_type == "detec":
        save_vis_heatmap_from_model(data, image_name="demo_multiple_dataset_iterator")
    return iterator


def save_vis_heatmap_from_model(x, image_name="demo"):
    img = x[0][0].permute(1, 2, 0).numpy()
    img = ImageProc.denormalize_mean_variance(img)
    img = cv2.resize(img, (350, 350))
    re = x[1][0].numpy()
    af = x[2][0].numpy()
    save_heatmap(img, [], re, af, image_name=image_name)


def test_build_dataset():
    pass


def main():
    parser = parse_base()
    parser.add_argument('--data_recog', type=str, help='path to recog data')
    parser.add_argument('--data_detec', type=str, help='path to detect data')
    args = parser.parse_args()

    if args.data_recog is not None:
        config = setup("recog", args)
        print(config)
        build_dataloader(config, args.data_recog)

    if args.data_detec is not None:
        config = setup("detec", args)
        print(config)
        build_dataloader(config, args.data_detec)


if __name__ == '__main__':
    main()
