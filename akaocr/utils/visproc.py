#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Mon November 03 10:00:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain visualize modules for various data type
_____________________________________________________________________________
"""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from pre.image import ImageProc


def json2contour(json_data):
    contours_words = list()
    contours_chars = list()
    text_list = list()
    for wo in json_data['words']:
        # print(wo)
        # print(wo.keys())
        if set(list(wo.keys())) != {'text', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'chars'} \
                and set(list(wo.keys())) != {'text', 'type', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'chars'}:
            print('invalid format', wo.keys())
            continue
        text_list.append(wo["text"])
        cwords = [
            [wo['x1'], wo['y1']],
            [wo['x2'], wo['y2']],
            [wo['x3'], wo['y3']],
            [wo['x4'], wo['y4']],
        ]
        contours_words.append(np.array(cwords))
        for ch in wo['chars']:
            cchars = [
                [ch['x1'], ch['y1']],
                [ch['x2'], ch['y2']],
                [ch['x3'], ch['y3']],
                [ch['x4'], ch['y4']],
            ]
            contours_chars.append(np.array(cchars))
    contours_words = np.array(contours_words)
    contours_chars = np.array(contours_chars)
    return contours_words, contours_chars, text_list


def draw_text(image, text, x, y, font_size=18, color=(0, 0, 0), font=None, thickness=3):
    """
    Function to draw text on image
    :param image: image source
    :param text: text
    :param x: x coordinate
    :param y: y coordinate
    :param font_size: font size
    :param color: text color
    :param font: font path
    :param thickness: text thickness
    :return: image with text visualized
    """
    img = np.copy(image)

    if font is None:
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size,
                                                    thickness=thickness)[0]
        text_offset_x = x
        text_offset_y = y

        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y),
                      (text_offset_x + text_width + 2, text_offset_y - text_height - 2))

        cv2.rectangle(img, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
        cv2.putText(img, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, color, thickness)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(im_pil)
        u_font = ImageFont.truetype(font, font_size)
        draw.text((x, y), text, font=u_font, fill=color)

        img = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
    return img


def visualizer(image_ori, contours=None, boxes=None, lines=None, bcolor=(0, 255, 0), texts=None,
               font='./data/default_vis_font.ttf',
               font_size=30, thick=2, windows=None, show=True, name='demo', tcolor=(255, 0, 0), gt_text=None,
               gt_color=(0, 0, 255)):
    """
    Function for visualize OCR result
    :param image_ori: image source
    :param contours: list of contours to visualize
    :param boxes: list of boxes to visualize
    :param lines: list of line to visualize
    :param bcolor: color of boxes or contours
    :param texts: list of text inline with boxes and contours
    :param font: path to font
    :param font_size:
    :param thick: thickness of boxes and contours
    :param windows: size of visualize window
    :param show: show image or not
    :param name: window name
    :param tcolor: text color
    :param gt_text: ground truth text to compare with OCR text
    :param gt_color: color of text match ground truth
    :return: image visualized
    """
    image = image_ori.copy()
    imshape = image.shape
    if len(imshape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if windows is None:
        windows = imshape[:2]

    if contours is not None:
        image = cv2.drawContours(image, contours.astype(int), -1, bcolor, thick)
        if texts is not None:
            if gt_text is None:
                for con, tex in zip(contours, texts):
                    image = self.draw_text(image, tex, con[3][0], con[3][1], font=font, font_size=font_size,
                                           color=tcolor)
            else:
                for con, tex, gt in zip(contours, texts, gt_text):
                    if tex == gt:
                        image = self.draw_text(image, tex, con[3][0], con[3][1], font=font, font_size=font_size,
                                               color=gt_color)
                    else:
                        image = self.draw_text(image, tex, con[3][0], con[3][1], font=font, font_size=font_size,
                                               color=tcolor)

    elif boxes is not None:
        for b in boxes:
            image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), bcolor, thick)
        if texts is not None:
            if gt_text is None:
                for box, tex in zip(boxes, texts):
                    image = self.draw_text(image, tex, box[0], box[3], font=font, font_size=font_size, color=tcolor)
            else:
                for box, tex, gt in zip(boxes, texts, gt_text):
                    if tex == gt:
                        image = self.draw_text(image, tex, box[0], box[3], font=font, font_size=font_size,
                                               color=gt_color)
                    else:
                        image = self.draw_text(image, tex, box[0], box[3], font=font, font_size=font_size,
                                               color=tcolor)

    if lines is not None:
        for li in lines:
            li = li[0]
            image = cv2.line(image, (li[0], li[1]), (li[2], li[3]), (255, 0, 0), thick, cv2.LINE_AA)

    if show:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, windows[0], windows[1])
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image


def save_heatmap(image, bboxes, region_scores,
                 affinity_scores, output_path="output", image_name="demo"):
    output_image = np.uint8(image.copy())
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    if len(bboxes) > 0:
        for i in range(len(bboxes)):
            _bboxes = np.int32(bboxes[i])
            for j in range(_bboxes.shape[0]):
                cv2.polylines(output_image, [np.reshape(_bboxes[j], (-1, 1, 2))], True, (0, 0, 255))

    target_gaussian_heatmap_color = ImageProc.cvt2_heatmap_img(region_scores)
    target_gaussian_affinity_heatmap_color = ImageProc.cvt2_heatmap_img(affinity_scores)
    heat_map = np.concatenate([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color], axis=1)
    output = np.concatenate([output_image, heat_map], axis=1)
    out_path = Path(output_path).joinpath("%s_input.jpg" % Path(image_name).stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), output)
