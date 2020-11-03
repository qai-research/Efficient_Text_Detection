import sys
sys.path.append("../")
import os
import torch

from utils.data.dataloader import LmdbDataset
from utils.data.lmdbproc import AlignCollate
from utils.file_utils import Constants
# import constants

def test_dataloader(root, config_path):
    constants = Constants(config_path)
    constants = constants.config

    _align_collate = AlignCollate(img_h=int(constants['img_h']), img_w=int(constants['img_w']),
                                  keep_ratio_with_pad=constants.getboolean('pad'))
    _dataset = LmdbDataset(root)
    _data_loader = torch.utils.data.DataLoader(
        _dataset, batch_size=int(constants['batch_size']),
        shuffle=True,
        num_workers=int(constants['workers']),
        collate_fn=_align_collate,
        pin_memory=True)

    iterator = iter(_data_loader)
    image, text = iterator.next()
    print(image.shape, text)

if __name__ == '__main__':
    root = "/home/bacnv6/data/train_data/recog/CR_sample"
    config_path = '../data/recog_constants.ini'
    test_dataloader(root, config_path)