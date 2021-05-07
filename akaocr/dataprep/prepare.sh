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

# Cocotext (https://rrc.cvc.uab.es/?ch=4&com=downloads)
# Image (train2014.zip)
wget -O train2014.zip http://msvocds.blob.core.windows.net/coco2014/train2014.zip
# Annot (COCO_Text.zip)
wget -O COCO_Text.zip https://s3.amazonaws.com/cocotext/COCO_Text.zip
# unzip
unzip train2014.zip
unzip COCO_Text.zip
# run convert dataset for cocotext (ex. on 72)
python code/convert_cocotext.py --src_path train2014 --annot_path COCO_Text.json --des_path cocotext_gt/train
# create lmdb from output gt (cocotext/train)

# SCUT-CTW1500 dataset
# train
wget -O train_images.zip https://universityofadelaide.box.com/shared/static/py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip
wget -O train_labels.zip https://universityofadelaide.box.com/shared/static/jikuazluzyj4lq6umzei7m2ppmt3afyw.zip

#test
wget -O test_images.zip https://universityofadelaide.box.com/shared/static/t4w48ofnqkdw7jyc4t11nsukoeqk9c3d.zip
wget -O test_labels.zip https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5/download

# unzip
unzip train_images.zip
unzip train_labels.zip
# for ctw1500
python convert_ctw1500.py --img_path train_images --ann_path ctw1500_train_labels --des_path ctw1500/train