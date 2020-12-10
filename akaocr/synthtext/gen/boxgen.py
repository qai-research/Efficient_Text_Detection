#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# noinspection SpellCheckingInspection
"""
_____________________________________________________________________________
Created By  : Vu Hoang Viet - VietVH9
Created Date: NOV - 2020
Project : AkaOCR core
_____________________________________________________________________________



The Module Has Been Build For Box Generator In Background Images (Blacklist Method)
_____________________________________________________________________________
"""

import os
import cv2
import pdb
import h5py
import copy
import math
import time
import random
import datetime
import logging
import operator
import numpy as np
import itertools as it
import multiprocessing
from scipy import optimize
from functools import reduce
from matplotlib import pyplot as plt
from pre.image import ImageProc
from synthtext.utils.utils_func import resize_with_char


def clockwise_sorted(coords):
    """
    Sort 4 points as top-left, top-right, bottom-right, bottom-left
    """
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    return np.int32([sorted(coords,
                            key=lambda coord:
                            (-135 - math.degrees(
                                math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)])


def normalize(xy):
    """
    Set the normalize data
    """
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = xy
    if abs(x3 - x1) < abs(y3 - y1):
        return (x4, y4), (x1, y1), (x2, y2), (x3, y3)
    return (x1, y1), (x2, y2), (x3, y3), (x4, y4)


def rotate(xy, theta):
    """
    Rotate augmentation
    """
    def _rotate_(xy, theta):
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        return (
            xy[0] * cos_theta - xy[1] * sin_theta,
            xy[0] * sin_theta + xy[1] * cos_theta
        )

    def _translate_(xy, offset):
        return xy[0] + offset[0], xy[1] + offset[1]

    (x1, y1), (_, _), (x3, y3), (_, _) = xy
    w = abs(x3 - x1)
    h = abs(y3 - y1)
    offset = (x1, y1)
    new_p = [(0, 0), (w, 0), (w, h), (0, h)]
    return [_translate_(_rotate_(xy, theta), offset) for xy in new_p]


class BoxGenerator:
    """
    The class is built with the task of generating box random on the segment regions defined.
    """

    def __init__(self, img_path,
                 fixed_size=None,
                 width_random_range=None,
                 heigh_random_range=None,
                 box_iter=None,
                 aug_percent=0,
                 num_samples=100,
                 max_num_box=100,
                 segment=None,
                 threshold=0.01,
                 max_size=None):
        """
        """
        # Read image from path and resize this image to maxsize with out changing of ratio
        image = cv2.imread(img_path)
        if max_size is not None:
            w, h = max_size
            if image.size > h * w * 3:
                old_h, old_w, _ = image.shape
                scale_percent = min([h / old_h, w / old_w])
                new_h, new_w = int(old_h * scale_percent), int(old_w * scale_percent)
                self.image = cv2.resize(image, (new_w, new_h))
            else:
                self.image = image
        else:
            self.image = image
        # Setup size of box, if fixed_size is not None, get random size for box
        self.fixed_size = fixed_size
        self.width_random_range = width_random_range
        self.heigh_random_range = heigh_random_range
        if self.fixed_size is None:
            self.min_box_w, self.max_box_w = self.width_random_range
            self.min_box_h, self.max_box_h = self.heigh_random_range
        else:
            self.min_box_w, self.min_box_h = self.fixed_size
        # Get param of class
        self.threshold = threshold
        self.min_box_size = self.min_box_w * self.min_box_h
        self.box_iter = box_iter
        self.num_samples = num_samples
        self.max_num_box = max_num_box
        self.aug_percent = aug_percent
        self.target_name = os.path.basename(img_path)
        self.target_base_name = os.path.splitext(self.target_name)[0]
        self.target_base_type = os.path.splitext(self.target_name)[1][1:]
        self.out_size = (self.image.shape[1], self.image.shape[0])
        # If the segment file is exists, read segment file and output is the segment of images
        if segment is not None:
            if type(segment) == str:
                f = h5py.File(segment, 'r')
                assert self.target_name in f['mask'].keys()
            else:
                f = segment.copy()
            val = np.array(f['mask'][self.target_name])
            m = np.max(val)
            self.segments = []
            for j in range(m):
                num = sum(sum(val == j + 1))
                if num > 2000:
                    tem = copy.deepcopy(val)
                    tem[tem != j] = 0
                    self.segments.append(tem)
            self.box_gen = self.box_generator_existed_masker
        else:
            self.box_gen = self.box_generator_with_masker_gen

    def mask_marker(self):
        """
        Create the segments of image without segment files
        """
        dump_img = self.image.copy()

        gray = cv2.cvtColor(dump_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        sure_bg = cv2.dilate(closing, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)
        ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)

        markers += 1
        markers[unknown == 255] = 0
        return cv2.watershed(dump_img, markers)

    def random_box(self, dump_marker, box_w, box_h):
        """
        Gen the random box to segment area with size is box_w and box_h
        """
        # Find all valid pixel (white dot on picture)
        all_valid_pixel = np.asarray(np.where(dump_marker[:, :, 0] == 255)).T
        # Number of trials to gen a box on segment area
        count = 100 if self.box_iter is None else self.box_iter
        while True:
            if count == 0:
                return None
            count -= 1
            # Randomly select a point in the valid area, create the rectangle box with box_w and box_h.
            y1, x1 = random.choice(all_valid_pixel)
            box_coordinates = [(x1, y1), (x1 + box_w, y1), (x1 + box_w, y1 + box_h), (x1, y1 + box_h)]
            # Make the created box transform.
            trans_box_coordinates = self.transform(box_coordinates)
            # Check that the position of the box is not outside the area of the image.
            if np.min(trans_box_coordinates) < 0 \
                    or max([x for x, y in trans_box_coordinates]) >= self.out_size[0] \
                    or max([y for x, y in trans_box_coordinates]) >= self.out_size[1]:
                continue
            # Check that no pixel of the box overlaps the marker black area
            if np.all(dump_marker[self.get_inside(trans_box_coordinates)] == 255):
                break
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = trans_box_coordinates
        if x1 == x3 or y1 == y3:
            return None
        m = max(abs(y3 - y1), abs(x3 - x1)) / min(abs(y3 - y1), abs(x3 - x1))
        x1 = np.random.randint(x1 - m, x1 + 1)
        y1 = np.random.randint(y1 - m, y1 + 1)
        x2 = np.random.randint(x2, x2 + m + 1)
        y2 = np.random.randint(y2 - m, y2 + 1)
        x3 = np.random.randint(x3, x3 + m + 1)
        y3 = np.random.randint(y3, y3 + m + 1)
        x4 = np.random.randint(x4 - m, x4 + 1)
        y4 = np.random.randint(y4, y4 + m + 1)
        return (x1, y1), (x2, y2), (x3, y3), (x4, y4)

    def get_inside(self, box_coordinates):
        """
        Find all black pixel of the box give by box_coordinates
        """
        empty_img = np.zeros_like(self.image)
        empty_img = cv2.fillPoly(empty_img, np.int32([box_coordinates]), [255, 255, 255])
        return np.where(empty_img == [255, 255, 255])

    def get_box_size(self, dump_marker):
        """
        Get the maximum box size valid in area
        """
        all_valid_pixel = np.asarray(np.where(dump_marker[:, :, 0] == 255)).T
        if self.fixed_size is None:
            max_w = max(self.width_random_range)
            max_h = max(self.heigh_random_range)
            if max_w * max_h >= len(all_valid_pixel):
                box_w = len(all_valid_pixel) / self.out_size[1]
                box_h = len(all_valid_pixel) / self.out_size[0]
            else:
                box_w = np.random.randint(*self.width_random_range)
                box_h = np.random.randint(*self.heigh_random_range)
        else:
            box_w, box_h = self.fixed_size

        return int(box_w), int(box_h)

    def box_generator_with_masker_gen(self):
        """
        Generate a new box in case there are no segments
        """
        # Get segment of image
        markers = self.mask_marker()
        # Convert image to black and white with 3 chanels
        img = self.image.copy()
        img[markers == 1] = [255, 255, 255]
        img[markers != 1] = [0, 0, 0]
        results = []
        for i in range(self.max_num_box):
            box_w, box_h = self.get_box_size(img)
            if box_w * box_h == 0:
                continue
            p = self.random_box(img, box_w, box_h)
            if p is not None:
                img = cv2.fillPoly(img, np.int32([p]), [0, 0, 0])
                results.append(p)
        return results

    @property
    def box_generator_existed_masker(self):
        """
        Generate a new box in case there are the segments existed.
        """
        results = []
        for i, markers in enumerate(self.segments):
            h, w = markers.shape
            dump_marker = np.zeros((h, w, 3))
            dump_marker[markers == 0] = [255, 255, 255]
            dump_marker = resize_with_char(dump_marker, self.out_size)
            num_box = 0
            img = self.image.copy()
            img[dump_marker != 255] = 255
            img[dump_marker == 255] = 0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (thresh, black_and_white) = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)

            black_percent = np.sum(black_and_white) / black_and_white.size
            if black_percent < self.threshold:
                p = self.gen_fixed_box(black_and_white)
                if p is not None:
                    cv2.fillPoly(img, np.int32([p]), [0, 0, 0])
                    results.append(self.normalize(p))
                    num_box += 1
            else:
                while True:
                    box_w, box_h = self.get_box_size(img)
                    if box_w * box_h == 0:
                        continue
                    p = self.random_box(img, box_w, box_h)
                    if p is not None:
                        img = cv2.fillPoly(img, np.int32([p]), [0, 0, 0])
                        results.append(p)
                        num_box += 1
                    if random.random() < num_box / self.max_num_box:
                        break
                    if len(results) > self.max_num_box:
                        break
                    if random.random() > 5 * black_percent:
                        break

            if len(results) > self.max_num_box:
                break
        return results

    def transform(self, box_coordinate):
        """
        Augmentation
        """
        if np.random.random() < self.aug_percent:
            d = np.random.choice(range(-90, 90))
            theta = math.radians(d)
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = self.rotate(box_coordinate, theta)
            return (x1, y1), (x2, y2), (x3, y3), (x4, y4)
        else:
            return box_coordinate

    def gen_fixed_box(self, matrix, iters=50, thresh=0.5):
        """
        Gen box with small box
        """
        all_valid_pixcel = np.asarray(np.where(matrix == 1)).T
        if len(all_valid_pixcel) == 0:
            return None
        max_square = 0
        results = None
        for i in range(iters):
            c = random.choice(all_valid_pixcel)
            c_y, c_x = int(c[0]), int(c[1])
            m_x = min([x for (y, x) in all_valid_pixcel if y == c_y])
            m_y = min([y for (y, x) in all_valid_pixcel if x == c_x])
            M_x = max([x for (y, x) in all_valid_pixcel if y == c_y])
            M_y = max([y for (y, x) in all_valid_pixcel if x == c_x])
            if (M_y - M_x) * (m_y - m_x) / len(all_valid_pixcel) < 0.3:
                continue
            while np.any(matrix[m_y:M_y + 1, m_x:M_x + 1] == 0):
                if random.random() < 0.5:
                    m_x += 1
                if random.random() < 0.5:
                    M_x -= 1
                if random.random() < 0.5:
                    m_y += 1
                if random.random() < 0.5:
                    M_y -= 1
            s = np.sum(matrix[m_y:M_y + 1, m_x:M_x + 1])
            if s > max_square and s / matrix.size > thresh:
                d = np.random.choice(range(-2, 2))
                theta = math.radians(d)
                results = self.rotate(((m_x, m_y), (M_x, m_y), (M_x, M_y), (m_x, M_y)), theta)
        return results

    def run(self):
        """
        Run the generator flow
        """
        self.mask_marker()
        results = []
        for _ in range(self.num_samples):
            uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':',
                                                                                                                    '.')
            name = ".".join([self.target_base_name, uniq_filename, self.target_base_type])
            out_json = {"file": name,
                        "width": self.out_size[0],
                        "height": self.out_size[1],
                        "depth": 3,
                        "words": []}
            for (x1, y1), (x2, y2), (x3, y3), (x4, y4) in self.box_gen():
                if (x3 - x1) == 0 or (y3 - y1) == 0:
                    pass
                else:
                    word_len = max(abs((y3 - y1) // (x3 - x1)), abs((x3 - x1) // (y3 - y1)))
                    word_out = {'text': 'x' * int(word_len),
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'x3': x3,
                                'y3': y3,
                                'x4': x4,
                                'y4': y4,
                                'chars': []}
                    out_json['words'].append(word_out)
            results.append(out_json)

        return results
