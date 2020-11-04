import sys

sys.path.append("../")
import os
import torch

from utils.data.dataloader import LmdbDataset
from utils.data.lmdbproc import AlignCollate
from utils.file_utils import Constants, read_vocab
from utils.data import lmdbproc


def test_dataloader(root, config_path, vocab, label_type="norm"):
    constants = Constants(config_path)
    constants = constants.config
    chars = read_vocab(vocab)
    labelproc = lmdbproc.LabelHandler(type=label_type, character=chars,
                                      sensitive=constants.getboolean('sensitive'),
                                      unknown=constants["unknown"])

    _dataset = LmdbDataset(root, labelproc=labelproc)

    _align_collate = AlignCollate(img_h=int(constants['img_h']), img_w=int(constants['img_w']),
                                  keep_ratio_with_pad=constants.getboolean('pad'))
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
    root_recog = "/home/bacnv6/data/train_data/recog/CR_DOC_WL_v4_30000"
    root_detec = '/home/bacnv6/data/train_data/detec/ST_DOC_WL_v4_30000'
    config_path = '../data/recog_constants.ini'
    vocab = '../data/vocabs/char_jpn_v2.txt'
    # vocab = '../data/vocabs/char_eng.txt'
    test_dataloader(root_detec, config_path, vocab, label_type="loadj")
