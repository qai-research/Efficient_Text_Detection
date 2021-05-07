#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Huu Kim - huukim911
Created Date: May 5, 2021 3:04pm GMT+0700
Project : AkaOCR core
_____________________________________________________________________________

This file contains function to convert from data source of Cocotext to desire groundtruth format
_____________________________________________________________________________

Refer API from https://github.com/bgshih/coco-text (coco_text.py)
Cocotext just has annot for train set only, split train/test/eval by yourself

Data source: 
    #refer: https://vision.cornell.edu/se3/coco-text-2/
    #annot: https://vision.cornell.edu/se3/wp-content/uploads/2019/05/COCO_Text.zip
    #image
            http://images.cocodataset.org/zips/train2014.zip
            http://images.cocodataset.org/zips/val2014.zip
            http://images.cocodataset.org/zips/test2014.zip
"""

import coco_text
import shutil
import json
import os
import argparse
from tqdm import tqdm
from pathlib import Path

subset = ['train','test', 'val']

def convert(src_path, annot_path, des_path):
    ct = coco_text.COCO_Text(annot_path)
    print(ct.info())
    for ss in subset:
        des_img_path = os.path.join(des_path, ss, 'images')
        des_jsn_path = os.path.join(des_path, ss, 'annotations')
        #remove old if exist for clean data
        if os.path.exists(des_img_path) and os.path.isdir(des_img_path):
            shutil.rmtree(des_img_path)
        if os.path.exists(des_jsn_path) and os.path.isdir(des_jsn_path):
            shutil.rmtree(des_jsn_path)
        #create folder output if not exist
        Path(des_img_path).mkdir(parents=True, exist_ok=True)
        Path(des_jsn_path).mkdir(parents=True, exist_ok=True)
        
        if ss == 'train':
            imgs = ct.getImgIds(imgIds=ct.train)
        elif ss == 'test':
            imgs = ct.getImgIds(imgIds=ct.test)
        else:
            imgs = ct.getImgIds(imgIds=ct.val)

        img_item = {}

        for img in tqdm(imgs):
            annIds = ct.getAnnIds(imgIds=img)
            anns = ct.loadAnns(annIds)
            img_prop = ct.loadImgs(img)

            img_item["file"] = img_prop[0]["file_name"]
            img_item["width"] = img_prop[0]["width"]
            img_item["height"] = img_prop[0]["height"]
            img_item["depth"] = 1
            words = []
            for ann in anns:
                word = {}
                if 'utf8_string' in list(ann.keys()):
                    left, top, width, height = ann['bbox']
                    x1, y1 = left, top
                    x2, y2 = x1+width, y1
                    x3, y3 = x2, y2+height
                    x4, y4 = x3-width, y3
                    
                    #append word
                    word["text"] = ann['utf8_string']
                    word["x1"] = round(x1,1)
                    word["y1"] = round(y1,1)
                    word["x2"] = round(x2,1)
                    word["y2"] = round(y2,1)
                    word["x3"] = round(x3,1)
                    word["y3"] = round(y3,1)
                    word["x4"] = round(x4,1)
                    word["y4"] = round(y4,1)
                    word["char"] = []
                    words.append(word)

            img_item["words"] = words
            src_img = os.path.join(src_path, img_item["file"])
            des_img = os.path.join(des_img_path, img_item["file"])
            des_jsn = os.path.join(des_jsn_path, img_item["file"][:img_item["file"].rfind('.')]+'.json')

            shutil.copyfile(src_img, des_img)
            with open(des_jsn, 'w') as js:
                json.dump(img_item, js, indent=4)

        print("Done convert set, check at: ", des_img_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', required=True, help='path to source data path')
    parser.add_argument('--annot_path', required=True, help='path to annot (.json) file of cocotext')
    parser.add_argument('--des_path', required=True, help='path to destination output path (contain images and annotations subfolder)')

    opt = parser.parse_args()
    convert(src_path=opt.src_path, annot_path=opt.annot_path, des_path=opt.des_path)

if __name__ == '__main__':
    main()

#src_path='D:/kimnh3/dataset/ocr/raw_download/cocotext/2014/train2014'
#annot_path='D:/kimnh3/dataset/ocr/raw_download/cocotext/annot-text/COCO_Text/COCO_Text.json'
#des_path='D:/kimnh3/dataset/ocr/converted/cocotext/'