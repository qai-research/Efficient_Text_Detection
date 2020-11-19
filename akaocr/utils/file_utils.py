import six
import lmdb
import json
import configparser
import numpy as np
from pathlib import Path
from PIL import Image
from utils.runtime import warn, error

class Constants:
    def __init__(self, config_path):
        config_p = Path(config_path)
        if not config_p.is_file():
            raise ValueError("Config file not found")
        config_parser = configparser.ConfigParser()
        config_parser.read(config_path)
        self.config = config_parser['DEFAULT']
        print('loaded config from : ', str(config_p.resolve()))


def read_vocab(file_path):
    with open(file_path, "r", encoding='utf-8-sig') as f:
        list_lines = f.read().strip().split("\n")
    return list_lines


def read_json_annotation(js_label):
    js_words = js_label["words"]
    character_boxes = []
    words = []
    for tex in js_words:  # loop through each word
        bboxes = []
        words.append(tex['text'])
        for ch in tex['chars']:
            bo = [
                [ch['x1'], ch['y1']],
                [ch['x2'], ch['y2']],
                [ch['x3'], ch['y3']],
                [ch['x4'], ch['y4']]
            ]
            bo = np.array(bo)
            bboxes.append(bo)
        character_boxes.append(np.array(bboxes))
    return words, character_boxes


class LmdbReader:
    def __init__(self, root, rgb=False):
        """
        Read lmdb dataset to variable
        :param root: path to lmdb database
        """
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            warn('cannot create lmdb from {}'.format(root))

        self.cursor = self.env.begin(write=False)
        try:
            self.num_samples = int(self.cursor.get('num-samples'.encode()))
        except:
            self.num_samples = int(self.cursor.stat()['entries'] / 2 - 1)
        self.rgb = rgb

    def lmdb_loader(self, index):
        """
        read binary image from lmdb file with custom key and convert to PIL image
        :param index: index of item to be fetch
        :param rgb: type of image color or gray
        :return:
        """
        label_key = 'label-{:09d}'.format(index).encode()
        label = self.cursor.get(label_key).decode('utf-8')
        img_key = 'image-{:09d}'.format(index).encode()
        img_buf = self.cursor.get(img_key)

        buf = six.BytesIO()
        buf.write(img_buf)
        buf.seek(0)
        try:
            if self.rgb:
                img = Image.open(buf).convert('RGB')  # for color image
            else:
                img = Image.open(buf).convert('L')
            return img, label
        except IOError:
            print(f'Corrupted image for {index}')
            # return None and ignore in dataloader
            return None
