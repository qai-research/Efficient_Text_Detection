import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import copy

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