
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import copy
from utils.file_utils import LmdbReader
from torch.utils.data import Dataset, ConcatDataset, Subset

class LmdbDataset(Dataset):
    """
    Base loader for lmdb type dataset
    """

    def __init__(self, root, rgb=False, labelproc=None):
        """
        :param root: path to lmdb dataset
        :param label_handler: type of label processing
        """
        # self.env, self.num_samples = read_lmdb(root)

        if labelproc is None:
            warn("You don\'t have label handler")

        self.labelproc = labelproc
        self.lmdbreader = LmdbReader(root, rgb)

    def __len__(self):
        return self.lmdbreader.num_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        image, label = self.lmdbreader.lmdb_loader(index)
        if self.labelproc is not None:
            label = self.labelproc.process_label(label)
        return image, label

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