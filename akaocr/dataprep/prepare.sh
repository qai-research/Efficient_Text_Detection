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

# set flag variable for status
export failed=0

# # Cocotext (https://rrc.cvc.uab.es/?ch=4&com=downloads)
# # Image (train2014.zip)
# wget -nc -O train2014.zip http://msvocds.blob.core.windows.net/coco2014/train2014.zip
# # Annot (COCO_Text.zip)
# wget -nc -O COCO_Text.zip https://s3.amazonaws.com/cocotext/COCO_Text.zip
# # unzip
# unzip -n train2014.zip
# unzip -n COCO_Text.zip
# # run convert dataset(ex. on 72)
# python code/convert_cocotext.py --src_path train2014 --annot_path COCO_Text.json --des_path cocotext || export failed=1
# #convert lmdb
# mkdir -p ../data/cocotext
# python code/seq_detectolmdb.py --torecog 0 --input cocotext --output ../data/cocotext || export failed=1

# # ICDAR13
# wget -nc -O Challenge2_Training_Task12_Images.zip https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task12_Images.zip
# wget -nc -O Challenge2_Training_Task1_GT.zip https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task1_GT.zip
# wget -nc -O Challenge2_Training_Task2_GT.zip https://rrc.cvc.uab.es/downloads/Challenge2_Training_Task2_GT.zip
# wget -nc -O Challenge2_Test_Task12_Images.zip https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task12_Images.zip
# wget -nc -O Challenge2_Test_Task1_GT.zip https://rrc.cvc.uab.es/downloads/Challenge2_Test_Task1_GT.zip


# # unzip
# unzip -n Challenge2_Training_Task12_Images.zip -d Challenge2_Training_Task12_Images
# unzip -n Challenge2_Training_Task1_GT.zip -d Challenge2_Training_Task1_GT
# unzip -n Challenge2_Training_Task2_GT.zip -d Challenge2_Training_Task2_GT
# unzip -n Challenge2_Test_Task12_Images.zip -d Challenge2_Test_Task12_Images
# unzip -n Challenge2_Test_Task1_GT.zip -d Challenge2_Test_Task1_GT

# # run convert dataset (ex. on 72)
# python code/convert_icdar13_char.py --train_img Challenge2_Training_Task12_Images --train_gt Challenge2_Training_Task1_GT --train_ch_gt Challenge2_Training_Task2_GT --test_img Challenge2_Test_Task12_Images --test_gt Challenge2_Test_Task1_GT --des_path icdar13 || export failed=1
# #convert lmdb
# mkdir -p ../data/icdar13
# python code/seq_detectolmdb.py --torecog 0 --input icdar13 --output ../data/icdar13 || export failed=1

# # ICDAR15
# wget -nc -O ch4_training_images.zip https://rrc.cvc.uab.es/downloads/ch4_training_images.zip
# wget -nc -O ch4_training_localization_transcription_gt.zip https://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip

# wget -nc -O ch4_test_images.zip https://rrc.cvc.uab.es/downloads/ch4_test_images.zip
# wget -nc -O Challenge4_Test_Task1_GT.zip https://rrc.cvc.uab.es/downloads/Challenge4_Test_Task1_GT.zip

# # unzip
# unzip -n ch4_training_images.zip -d ch4_training_images
# unzip -n ch4_training_localization_transcription_gt.zip -d ch4_training_localization_transcription_gt
# unzip -n ch4_test_images.zip -d ch4_test_images
# unzip -n Challenge4_Test_Task1_GT.zip -d Challenge4_Test_Task1_GT

# # run convert dataset (ex. on 72)
# python code/convert_icdar15.py --train_img ch4_training_images --train_gt ch4_training_localization_transcription_gt --test_img ch4_test_images --test_gt Challenge4_Test_Task1_GT --des_path icdar15 || export failed=1
# #convert lmdb
# mkdir -p  ../data/icdar15
# python code/seq_detectolmdb.py --torecog 0 --input icdar15 --output ../data/icdar15 || export failed=1


# Synthtext800k
# download
wget -nc -O SynthText.zip https://thor.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip
# unzip
unzip -n SynthText.zip
# run convert dataset (ex. on 72)
python code/convert_synthtext.py --data_path SynthText --des_path synthtext800k || export failed=1
#convert lmdb (for loop sub synthtext folders)
cd synthtext800k
pwd
mkdir synthtext_train
mkdir synthtext_test
ls
pwd
mv synthtext_p9 synthtext_test/.
mv synthtext_p* synthtext_train/.
ls
ls synthtext_test
ls synthtext_train
# create for 1 test sets (ST_synthtext_p9)
cd synthtext_test
pwd
mkdir -p ../../../data/synthtext/synthtext_test/synthtext_p9
python ../../code/seq_detectolmdb.py --torecog 0 --input synthtext_p9 --output ../../../data/synthtext/synthtext_test/synthtext_p9 || export failed=1

cd ../synthtext_train

# create for 8 train sets (ST_synthtex_p1...8)
for st in *; do
    echo $st
    mkdir -p ../../../data/synthtext/synthtext_train/$st
    python ../../code/seq_detectolmdb.py --torecog 0 --input $st --output ../../../data/synthtext/synthtext_train/$st || export failed=1
done

cd ../../

# print complete message
if [ ${failed} -ne 0 ]; then
        echo "Prepare failed, check error on the terminal history above..."
      else
        echo "Prepare 4 datasets completed, check at akaocr/data/..."
      fi
