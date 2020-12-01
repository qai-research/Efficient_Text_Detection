#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Vu Hoang Viet - VietVH9
Created Date: NOV - 2020
Project : AkaOCR core
_____________________________________________________________________________



The Module Has Been Build For Convert Text Image To Polygon Bounding Box
_____________________________________________________________________________
"""
import os
import sys
import cv2
import json
import datetime
import numpy as np


def new_coordinate(p, matrix):
    """
    Get new coordinate after transform from old coordinate
    """
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    return int(px), int(py)


def tranform_matrix(source_coor, target_coor):
    """
    Get transform matrix
    """
    return cv2.getPerspectiveTransform(source_coor, target_coor)


class PerspectiveTransform:
    """
    Module convert
    """

    def __init__(self,
                 source_images,
                 source_chars_coor,
                 target_points,
                 target_image_path,
                 max_size=None,
                 inpainting=False,
                 fixed_ratio=False):

        # Equivalent element number standards are ensured
        assert len(source_images) == len(target_points)
        assert len(source_images) == len(source_chars_coor)

        self.source_images = source_images
        self.source_chars_coor = source_chars_coor
        self.target_points = target_points
        target_image = cv2.imread(target_image_path)

        # Resize target image to the maxsize of picture with out changing of ratio
        if max_size is not None:
            w, h = max_size
            if target_image.size > h * w * 3:
                old_h, old_w, _ = target_image.shape
                scale_percent = min([old_h / h, old_w / w])
                new_h, new_w = int(old_h * scale_percent), int(old_w * scale_percent)
                self.target_image = cv2.resize(target_image, (new_w, new_h))
            else:
                self.target_image = target_image
        else:
            self.target_image = target_image

        self.target_base_name = os.path.splitext(os.path.basename(target_image_path))[0]
        self.target_base_type = os.path.splitext(os.path.basename(target_image_path))[1][1:]

        # Use inpainting
        if inpainting:
            self.target_image = self.inpainting()
        self.out_size = (self.target_image.shape[1], self.target_image.shape[0])
        self.fixed_ratio = fixed_ratio

    def inpainting(self):
        """
        Delete the object and change the color of the deleted area
        """
        clearned_target_image = self.target_image.copy()
        cv2.fillPoly(clearned_target_image, np.int32(self.target_points), [255, 255, 255])
        mask_img = np.uint8(np.zeros(clearned_target_image.shape))
        cv2.fillPoly(mask_img, np.int32(self.target_points), [255, 255, 255])
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        return cv2.inpaint(clearned_target_image, mask_img, 3, flags=cv2.INPAINT_NS)

    def transform(self, image, trans_matrix):
        """
        Transformer to trans image with trans_matrix has ben created before
        """
        return cv2.warpPerspective(image,
                                   trans_matrix,
                                   self.out_size,
                                   borderValue=(255, 255, 255))

    def fit(self, name=None):
        """
        Run the transfrom function
        """
        result_pic = self.target_image.copy()
        uniq_filename = str(datetime.datetime.now().date()).replace('-', '') + '_' + str(
            datetime.datetime.now().time()).replace(':', '')
        if name is None:
            name = ".".join([self.target_base_name, uniq_filename, self.target_base_type])
        out_json = {
            "file": name,
            "width": self.out_size[0],
            "height": self.out_size[1],
            "depth": 3,
            "words": []}
        for img, char_coor, target_point in zip(self.source_images, self.source_chars_coor, self.target_points):
            h, w, _ = np.asarray(img).shape
            source_point = np.float32([[(0, 0), (w, 0), (w, h), (0, h)]])
            matrix = tranform_matrix(source_point, target_point)
            word_img = self.transform(img, matrix)
            alpha = np.random.uniform(0.7, 0.9)
            mapped = cv2.bitwise_and(word_img, result_pic)
            result_pic = cv2.addWeighted(mapped, alpha, result_pic, 1 - alpha, 0, result_pic)
            word_out = {'text': char_coor['words'],
                        'x1': int(target_point[0][0][0]), 'y1': int(target_point[0][0][1]),
                        'x2': int(target_point[0][1][0]), 'y2': int(target_point[0][1][1]),
                        'x3': int(target_point[0][2][0]), 'y3': int(target_point[0][2][1]),
                        'x4': int(target_point[0][3][0]), 'y4': int(target_point[0][3][1]),
                        'chars': []}
            for char_dict in char_coor['text']:
                out_char = {'char': char_dict['char'],
                            'x1': (new_coordinate((char_dict['x1'], char_dict['y1']), matrix))[0],
                            'y1': (new_coordinate((char_dict['x1'], char_dict['y1']), matrix))[1],
                            'x2': (new_coordinate((char_dict['x2'], char_dict['y2']), matrix))[0],
                            'y2': (new_coordinate((char_dict['x2'], char_dict['y2']), matrix))[1],
                            'x3': (new_coordinate((char_dict['x3'], char_dict['y3']), matrix))[0],
                            'y3': (new_coordinate((char_dict['x3'], char_dict['y3']), matrix))[1],
                            'x4': (new_coordinate((char_dict['x4'], char_dict['y4']), matrix))[0],
                            'y4': (new_coordinate((char_dict['x4'], char_dict['y4']), matrix))[1]}
                word_out['chars'].append(out_char)
            out_json['words'].append(word_out)
        return result_pic, out_json
