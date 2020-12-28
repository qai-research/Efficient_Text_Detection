#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Vu Hoang Viet - VietVH9
Created Date: NOV - 2020
Project : AkaOCR core
_____________________________________________________________________________



The Module Has Been Build for...
_____________________________________________________________________________
"""
import os
import sys
import cv2
import json
import glob
import random
import pygame
import numpy as np
import pygame.locals
import pygame.freetype
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from .text_to_image import TextFontGenerator
from synthtext.utils.data_loader import lmdb_dataset_loader
from synthtext.pre.perspective import PerspectiveTransform, new_coordinate


class HandWritingGenerator:
    """
    Generator text image from font with PyGame and handwriting images
    """

    def __init__(self, lmdb_path, fonts_path, read_batch=32, font_size_range=None, char_spacing_range=None,
                 fixed_box=True):
        print("Start")
        self.data = lmdb_dataset_loader(lmdb_path, batch_size=read_batch)
        print("Loaded")
        self.possition_gen = TextFontGenerator(fonts_path,
                                               font_size_range=font_size_range,
                                               fixed_box=False,
                                               char_spacing_range=char_spacing_range
                                               )
        self.fixed_box = fixed_box

    def generator(self, source_word):
        """
        @type source_word: str
        @param source_word: word string
        @return: image and bouding box for each character
        """
        target_img, target_poss = self.possition_gen.generator(source_word)
        if self.fixed_box is True:
            return self.fixed_box_gen(target_img, target_poss)
        else:
            return self.none_fixed_box_gen(target_img, target_poss)

    def none_fixed_box_gen(self, target_img, target_poss):
        """
        Gen image with small bounding box for each character
        """
        target_img = np.array(target_img)
        results = np.ones_like(target_img) * 255
        H, W, _ = results.shape
        for index, info in enumerate(target_poss['text']):
            char = info['char']
            x1 = info['x1']
            y1 = info['y1']
            x2 = info['x2']
            y2 = info['y2']
            x3 = info['x3']
            y3 = info['y3']
            x4 = info['x4']
            y4 = info['y4']
            target_coor = np.float32([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]])
            source_img = self.self.char_img(char)(char)
            h, w = np.array(source_img).shape
            source_coor = np.float32([[(0, 0), (w, 0), (w, h), (0, h)]])
            trans_matrix = cv2.getPerspectiveTransform(source_coor, target_coor)
            word_img = cv2.warpPerspective(source_img,
                                           trans_matrix,
                                           (W, H),
                                           borderValue=(255, 255, 255))
            img2 = np.ones_like(results) * 255
            img2[:, :, 0] = word_img
            img2[:, :, 1] = word_img
            img2[:, :, 2] = word_img
            results = cv2.bitwise_and(img2, results)
        return results, target_poss

    def fixed_box_gen(self, target_img, target_poss):
        """
        Gen image with big bounding box for each character
        """
        target_img = np.array(target_img)
        results = np.ones_like(target_img) * 255
        H, W, _ = results.shape
        for index, info in enumerate(target_poss['text']):
            char = info['char']
            x1 = info['x1']
            y1 = info['y1']
            x2 = info['x2']
            y2 = info['y2']
            x3 = info['x3']
            y3 = info['y3']
            x4 = info['x4']
            y4 = info['y4']
            target_coor = np.float32([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]])
            source_img = self.char_img(char)
            h, w = np.array(source_img).shape
            source_coor = np.float32([[(0, 0), (w, 0), (w, h), (0, h)]])
            trans_matrix = cv2.getPerspectiveTransform(source_coor, target_coor)
            word_img = cv2.warpPerspective(np.float32(source_img),
                                           trans_matrix,
                                           (W, H),
                                           borderValue=(255, 255, 255))
            img2 = np.ones_like(results) * 255
            img2[:, :, 0] = word_img
            img2[:, :, 1] = word_img
            img2[:, :, 2] = word_img
            results = cv2.bitwise_and(img2, results)
            target_poss['text'][index]['y1'] = 0
            target_poss['text'][index]['y2'] = 0
            target_poss['text'][index]['y3'] = h
            target_poss['text'][index]['y4'] = h
        return results, target_poss

    def char_img(self, char):
        try:
            source_img = self.data.random_sample(char)
        except Exception:
            source_img, _ = self.possition_gen.generator(char)
            h, w, _ = np.array(source_img).shape
            source_img = np.reshape(np.array(source_img)[:, :, 0], (h, w))
        return source_img
