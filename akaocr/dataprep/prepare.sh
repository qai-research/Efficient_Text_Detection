#!/bin/bash
# -*- coding: utf-8 -*-
# """
# _____________________________________________________________________________
# Created By  : Nguyen Huu Kim - huukim911
# Created Date: May 6, 2021 2:25pm GMT+0700
# Project : AkaOCR core
# _____________________________________________________________________________

# This file contain end-to-end sh script to prepare, convert data into desired format for training/testing
# _____________________________________________________________________________
# """

## Download
#COCO-Text
#annot
wget https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip
#image
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/zips/test2014.zip

#SCUT-CTW1500 dataset
#download annot xml
wget https://drive.google.com/open?id=13sNLo3s8hO8_2ldkVapL7Q7LRBp8Yr-g
#download image
wget https://1drv.ms/u/s!Aplwt7jiPGKilH4XzZPoKrO7Aulk

#Totaltext
wget https://drive.google.com/file/d/1bC68CzsSVTusZVvOkk7imSZSbgD1MqK2/view
wget https://drive.google.com/file/d/1v-pd-74EkZ3dWe6k0qppRtetjdPQ3ms1/view
#unzip annot and image

#clone repo
git clone https://github.com/qai-research/Efficient_Text_Detection.git
#cd to repo
cd Efficient_Text_Detection/akaocr
#install packages:
pip install -r akaocr/requirements.txt

#run convert dataset for cocotext (ex. on 72)
python dataprep/convert_cocotext.py --src_path D:/kimnh3/dataset/ocr/raw_download/cocotext/2014/train2014 --annot_path D:/kimnh3/dataset/ocr/raw_download/cocotext/annot-text/COCO_Text/COCO_Text.json --des_path D:/kimnh3/dataset/ocr/converted/cocotext/train
#for ctw1500
python convert_ctw1500.py --img_path D:/kimnh3/dataset/ocr/raw_download/SCUT-CTW1500/v1-xml/train-1000/train_images --ann_path D:/kimnh3/dataset/ocr/raw_download/SCUT-CTW1500/v1-xml/train-1000/ctw1500_train_labels --des_path D:/kimnh3/dataset/ocr/converted/scut-ctw1500/train