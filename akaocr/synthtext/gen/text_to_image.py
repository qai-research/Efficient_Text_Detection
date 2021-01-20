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


class TextFontGenerator:
    """
    Generator text image from font with Pillow and True Font Type
    """

    def __init__(self, fonts_path,
                 font_size_range,
                 fixed_box=True,
                 random_color=False,
                 font_color=(0, 0, 0),
                 char_spacing_range=None):
        self.fonts_list = glob.glob(os.path.join(fonts_path, "*.ttf"))
        self.fonts_list.extend(glob.glob(os.path.join(fonts_path, "*.TTF")))
        self.font_size_range = font_size_range
        self.fixed_box = fixed_box
        self.random_color = random_color
        self.font_color = font_color
        self.char_spacing_range = char_spacing_range

    def generator(self, source_word):
        """
        Gen image with bounding box for each character
        """
        try:
            font_path = random.choice(self.fonts_list)
            pygame.freetype.init()

            if self.font_size_range is not None:
                l, h = self.font_size_range
                font_size = np.random.randint(low=l, high=h)
            else:
                font_size = np.random.randint(10, 50)
            font_renderer = pygame.freetype.Font(font_path, size=font_size, font_index=0, resolution=0, ucs4=False)
            if self.char_spacing_range is not None:
                l, h = self.char_spacing_range
                char_spacing_factor = round(random.uniform(l, h), 1)
            else:
                char_spacing_factor = 0
            if self.fixed_box is True:
                img, out_json = self.fixed_box_gen(font_renderer, source_word, char_spacing_factor)
            else:
                img, out_json = self.none_fixed_box_gen(font_renderer, source_word, char_spacing_factor)

            pygame.freetype.quit()
        except TypeError:
            img, out_json = self.generator(source_word)
        return img, out_json

    def fixed_box_gen(self, font_renderer, word, char_spacing_factor=0):
        """
        Gen image with big bounding box for each character
        """
        if self.random_color is False:
            color = self.font_color
        else:
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        width = font_renderer.get_rect('O').w
        heigh = font_renderer.get_rect('Gg').h + 1
        char_spacing = int(char_spacing_factor * width)
        fsize = (width + char_spacing) * (len(word) + 10), heigh * 2
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
        surf.fill((255, 255, 255))
        out_json = {"words": word,
                    "text": []}

        re = font_renderer.get_rect(word[0])
        old_x = re.x
        old_w = re.w
        y_change = heigh
        re.y = y_change - re.y
        font_renderer.render_to(surf, re, word[0], fgcolor=color)
        first_x = 0
        first_y = 0
        img_w = re.w + 1
        img_h = first_y + re.h + 1
        if word[0] != " ":
            char_json = {'char': word[0],
                         'x1': re.x,
                         'y1': 0,
                         'x2': re.x + re.w,
                         'y2': 0,
                         'x3': re.x + re.w,
                         'y3': 0,
                         'x4': re.x,
                         'y4': 0}
            out_json['text'].append(char_json)
        for char in word[1:]:
            re = font_renderer.get_rect(char)
            re.x = old_x + old_w + char_spacing
            re.y = y_change - re.y
            font_renderer.render_to(surf, re, char, fgcolor=color)
            old_x = re.x
            old_w = re.w
            if char != " ":
                char_json = {'char': char,
                             'x1': re.x,
                             'y1': 0,
                             'x2': re.x + re.w,
                             'y2': 0,
                             'x3': re.x + re.w,
                             'y3': 0,
                             'x4': re.x,
                             'y4': 0}
                out_json['text'].append(char_json)
                img_w = max(img_w, re.x + re.w - first_x + 2)
                img_h = max(img_h, re.y + re.h)
        for ind, _ in enumerate(out_json['text']):
            out_json['text'][ind]['y3'] = img_h - 1
            out_json['text'][ind]['y4'] = img_h - 1

        raw_str = pygame.image.tostring(surf, 'RGB', False)
        image = Image.frombytes('RGB', surf.get_size(), raw_str)
        image = image.crop((first_x, first_y, img_w, img_h))
        pygame.freetype.quit()
        return image, out_json

    def none_fixed_box_gen(self, font_renderer, word, char_spacing_factor=0):
        """
        Gen image with small bounding box for each character
        """
        if self.random_color is False:
            color = self.font_color
        else:
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        width = font_renderer.get_rect('O').w
        heigh = font_renderer.get_rect('Gg').h + 1
        char_spacing = int(char_spacing_factor * width)
        fsize = (width + char_spacing) * (len(word) + 10), heigh * 2
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)
        surf.fill((255, 255, 255))
        out_json = {"words": word,
                    "text": []}

        re = font_renderer.get_rect(word[0])
        old_x = re.x
        old_w = re.w
        y_change = heigh
        re.y = y_change - re.y
        font_renderer.render_to(surf, re, word[0], fgcolor=color)
        first_x = 0
        first_y = 0
        img_w = re.w + 1
        img_h = first_y + re.h + 1
        if word[0] != " ":
            char_json = {'char': word[0],
                         'x1': re.x,
                         'y1': re.y,
                         'x2': re.x + re.w,
                         'y2': re.y,
                         'x3': re.x + re.w,
                         'y3': re.y + re.h,
                         'x4': re.x,
                         'y4': re.y + re.h}
            out_json['text'].append(char_json)
        for char in word[1:]:
            re = font_renderer.get_rect(char)
            re.x = old_x + old_w + char_spacing
            re.y = y_change - re.y
            font_renderer.render_to(surf, re, char, fgcolor=color)
            old_x = re.x
            old_w = re.w
            if char != " ":
                char_json = {'char': char,
                             'x1': re.x,
                             'y1': re.y,
                             'x2': re.x + re.w,
                             'y2': re.y,
                             'x3': re.x + re.w,
                             'y3': re.y + re.h,
                             'x4': re.x,
                             'y4': re.y + re.h}
                out_json['text'].append(char_json)
                img_w = max(img_w, re.x + re.w - first_x + 2)
                img_h = max(img_h, re.y + re.h)

        raw_str = pygame.image.tostring(surf, 'RGB', False)
        image = Image.frombytes('RGB', surf.get_size(), raw_str)
        image = image.crop((first_x, first_y, img_w, img_h))
        return image, out_json
