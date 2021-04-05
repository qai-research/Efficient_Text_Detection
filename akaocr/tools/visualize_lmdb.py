import os
import argparse
import random
import cv2
import json
import lmdb
import six
import numpy as np
import inspect
import sys

from tqdm import tqdm
from PIL import Image

################
# add detection folder to sys path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
# parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

from pipeline.utils import Visualizer


def get_item_lmdb(env, index):
    config = dict()
    if index == 0:
        index = 1
    with env.begin(write=False) as txn:
        label_key = 'label-{:09d}'.format(index).encode()
        if label_key is None:
            index = random.randint(1, max(1, index - 1))
            label_key = 'label-{:09d}'.format(index).encode()
        label = txn.get(label_key).decode('utf-8')
        img_key = 'image-{:09d}'.format(index).encode()
        img_buf = txn.get(img_key)

        buf = six.BytesIO()
        buf.write(img_buf)
        buf.seek(0)
        try:
            if "rgb" in config:
                rgb = config["rgb"]
            else:
                rgb = True
            if rgb:
                image = Image.open(buf).convert('RGB')  # for color image
            else:
                image = Image.open(buf).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            index = random.randint(1, index)
            return get_item_lmdb(env, index)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    label_json = json.loads(label)
    return image, label_json


def visualize(img, data, save_dir, mode):
    filename = data['file']
    vzer = Visualizer()
    words = data['words']
    contours_chars = list()
    contours_words = list()
    text_list = list()
    for text in words:
        cwords = [
            [text['x1'], text['y1']],
            [text['x2'], text['y2']],
            [text['x3'], text['y3']],
            [text['x4'], text['y4']],
        ]
        contours_words.append(np.array(cwords))
        for ch in text['chars']:
            cchars = [
                [ch['x1'], ch['y1']],
                [ch['x2'], ch['y2']],
                [ch['x3'], ch['y3']],
                [ch['x4'], ch['y4']],
            ]
            contours_chars.append(np.array(cchars))
            text_list.append(ch['text'])

    contours_words = np.array(contours_words)
    contours_chars = np.array(contours_chars)
    if mode == 'words':
        img = vzer.visualizer(img, contours=contours_words, show=False, texts=text_list,
                              font='../data/default_vis_font.ttf',
                              font_size=18)
    elif mode == 'chars':
        img = vzer.visualizer(img, contours=contours_chars, show=False, texts=text_list,
                              font='../data/default_vis_font.ttf', font_size=18)
    cv2.imwrite(os.path.join(save_dir, filename), img)


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lmdb', required=True, help='Path to the folder include mdb files')
    parser.add_argument('-t', '--total', default=10, help='Total sample images visualize')
    parser.add_argument('-o', '--output', required=True, help='Output dir to save sample images visualize')
    parser.add_argument('-m', '--mode', required=True, default='char',
                        help='Draw bounding box for Character/Words',
                        choices=['chars', 'words'])

    return parser.parse_args()


def main(opt):
    root_dir = opt.lmdb
    out_dir = opt.output
    total = opt.total
    mode = opt.mode
    roots = [os.path.join(root_dir, root) for root in os.listdir(root_dir) if os.path.join(root_dir, root)]
    for root in roots:
        save_dir = os.path.join(out_dir, os.path.basename(os.path.normpath(root)))
        vsl_dir = os.path.join(save_dir, 'Visualize')
        os.makedirs(vsl_dir, exist_ok=True)

        env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            length = int(txn.get('num-samples'.encode()))
        if total > length:
            total = lenght

        for index in tqdm(range(1, total)):
            try:
                image, data = get_item_lmdb(env, index)
                visualize(image, data, vsl_dir, mode)
            except:
                print('Error :', index)


if __name__ == '__main__':
    opt = get_argparse()
    main(opt)
