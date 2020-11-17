import os
import json
import math
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

from utils.transforms.gaussian import GaussianTransformer
from utils.transforms.heatproc import transform2heatmap


class LabelHandler:
    """
    Class contain all pre-process of label from LMDB database
    """
    def __init__(self, type="norm", character=None, sensitive=True, unknown="?"):
        """
        :param type: transformation type of label # norm/json
        """
        self.type = type
        self.character = character
        self.sensitive = sensitive
        self.unknown = unknown

    @staticmethod
    def normalize_label(label, character, sensitive=True, unknown="?"):
        """
        pre-process label before training (cleaning + masking)
        :param label: input label (string)
        :param character: input vocab (list)
        :param sensitive: training with upper case data
        :param unknown: char to replace unknown chars(do not show up in vocab)
        :return: clean label
        """
        if not sensitive:  # case insensitive
            label = label.lower()
        if unknown:
            label = list(label)
            for i in range(len(label)):
                if label[i] not in character and label[i] != " ":
                    label[i] = unknown
            label = "".join(label)
        return label

    @staticmethod
    def load_json(label):
        return json.loads(label)

    def process_label(self, label):
        if self.type == "norm":
            return self.normalize_label(label, self.character, self.sensitive, self.unknown)
        elif self.type == "json":
            return self.load_json(label)
        else:
            return label


class AlignCollate(object):
    """
    Resizing batch of raw images then convert them into Tensors, ready to feed into the model.
    """
    def __init__(self, img_h=32, img_w=254, keep_ratio_with_pad=True):
        """
        Custom collate function for normalize image
        :param img_h: image height
        :param img_w: image width
        :param keep_ratio_with_pad: pad image with empty
        """
        self.img_h = img_h
        self.img_w = img_w
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.img_w
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.img_h, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.img_h * ratio) > self.img_w:
                    resized_w = self.img_w
                else:
                    resized_w = math.ceil(self.img_h * ratio)

                resized_image = transform(image.resize((resized_w, self.img_h), Image.BICUBIC))
                resized_images.append(resized_image)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.img_w, self.img_h))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


class ResizeNormalize(object):
    """Resizing input images by stretching them"""

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)  # [0, 1] => [-1, 1]
        return img


class NormalizePAD(object):
    """Resizing input images by padding with zeros"""

    def __init__(self, max_size, pad_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = pad_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return pad_img


class GaussianCollate(object):
    """
    Return label in heatmap representation
    """
    def __init__(self, min_size, max_size, rgb=True):
        self.gaussian_transformer = GaussianTransformer(img_size=512, region_threshold=0.35, affinity_threshold=0.15)
        # self.toTensor = transforms.ToTensor()
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)
        # img = self.toTensor(img)
        # img.sub_(0.5).div_(0.5)  # [0, 1] => [-1, 1]
        images_proc = list()
        regions_proc = list()
        affinities_proc = list()
        confidences_proc = list()
        for img, label in zip(images, labels):
            img = np.array(img)
            heat_data = transform2heatmap(img, label,
                                          self.gaussian_transformer,
                                          self.min_size, self.max_size)

            img, region_scores, affinity_scores, confidence_mask, confidences = heat_data
            img = torch.from_numpy(img).float().permute(2, 0, 1)
            region_scores_torch = torch.from_numpy(region_scores / 255).float()
            affinity_scores_torch = torch.from_numpy(affinity_scores / 255).float()
            confidence_mask_torch = torch.from_numpy(confidence_mask).float()

            images_proc.append(img)
            regions_proc.append(region_scores_torch)
            affinities_proc.append(affinity_scores_torch)
            confidences_proc.append(confidence_mask_torch)

        image_tensors = torch.cat([t.unsqueeze(0) for t in images_proc], 0)
        region_tensors = torch.cat([t.unsqueeze(0) for t in regions_proc], 0)
        affinity_tensors = torch.cat([t.unsqueeze(0) for t in affinities_proc], 0)
        confidence_tensors = torch.cat([t.unsqueeze(0) for t in confidences_proc], 0)
        return image_tensors, region_tensors, affinity_tensors, confidence_tensors
