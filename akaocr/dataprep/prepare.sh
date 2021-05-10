#!/bin/bash
# -*- coding: utf-8 -*-
# """
# _____________________________________________________________________________
# Created By  : Nguyen Huu Kim - huukim98@gmail.com
# Created Date: May 6, 2021 2:25pm GMT+0700
# Project : AkaOCR core
# _____________________________________________________________________________
# This file contain end-to-end sh script to prepare, convert data into desired format for training/testing
# _____________________________________________________________________________
# """

# mkdir data training folders
mkdir -p ../data/train/cocotext ../data/test/cocotext
mkdir -p ../data/train/icdar13 ../data/test/icdar13
mkdir -p ../data/train/icdar15 ../data/test/icdar15
# Cocotext (https://rrc.cvc.uab.es/?ch=4&com=downloads)
# Image (train2014.zip)
wget -nc -O train2014.zip http://msvocds.blob.core.windows.net/coco2014/train2014.zip
# Annot (COCO_Text.zip)
wget -nc -O COCO_Text.zip https://s3.amazonaws.com/cocotext/COCO_Text.zip
# unzip
unzip train2014.zip
unzip COCO_Text.zip
# run convert dataset(ex. on 72)
python code/convert_cocotext.py --src_path train2014 --annot_path COCO_Text.json --des_path cocotext
#convert lmdb
python code/seq_detectolmdb.py --torecog 0 --input cocotext --output ../data/cocotext

# ICDAR13
wget -nc -O Challenge2_Training_Task12_Images.zip https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task12_Images.zip
wget -nc -O Challenge2_Training_Task1_GT.zip https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task1_GT.zip

wget -nc -O Challenge2_Test_Task12_Images.zip https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task12_Images.zip
wget -nc -O Challenge2_Test_Task1_GT.zip https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task1_GT.zip

# unzip
unzip Challenge2_Training_Task12_Images.zip -d Challenge2_Training_Task12_Images
unzip Challenge2_Training_Task1_GT.zip -d Challenge2_Training_Task1_GT
unzip Challenge2_Test_Task12_Images.zip -d Challenge2_Test_Task12_Images
unzip Challenge2_Test_Task1_GT.zip -d Challenge2_Test_Task1_GT
# run convert dataset (ex. on 72)
python code/convert_icdar13.py --train_img Challenge2_Training_Task12_Images --train_gt Challenge2_Training_Task1_GT --test_img Challenge2_Test_Task12_Images --test_gt Challenge2_Test_Task1_GT --des_path icdar13
#convert lmdb
python code/seq_detectolmdb.py --torecog 0 --input icdar13/train --output ../data/icdar13

# ICDAR15
wget -nc -O ch4_training_images.zip https://rrc.cvc.uab.es/downloads/ch4_training_images.zip
wget -nc -O ch4_training_localization_transcription_gt.zip https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip

wget -nc -O ch4_test_images.zip https://rrc.cvc.uab.es/downloads/ch4_test_images.zip
wget -nc -O Challenge4_Test_Task1_GT.zip https://rrc.cvc.uab.es/downloads/Challenge4_Test_Task1_GT.zip

# unzip
unzip ch4_training_images.zip -d ch4_training_images
unzip ch4_training_localization_transcription_gt.zip -d ch4_training_localization_transcription_gt
unzip ch4_test_images.zip -d ch4_test_images
unzip Challenge4_Test_Task1_GT.zip -d Challenge4_Test_Task1_GT

# run convert dataset (ex. on 72)
python code/convert_icdar15.py --train_img ch4_training_images --train_gt ch4_training_localization_transcription_gt --test_img ch4_test_images --test_gt Challenge4_Test_Task1_GT --des_path icdar15
#convert lmdb
python code/seq_detectolmdb.py --torecog 0 --input icdar15/train --output ../data/icdar15
