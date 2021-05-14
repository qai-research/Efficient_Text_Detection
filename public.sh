#!/bin/bash
# -*- coding: utf-8 -*-
# """
# _____________________________________________________________________________
# Created By  : Nguyen Huu Kim - Kimnh3
# Created Date: Apr 27, 2021 10:00am GMT+0700
# Project : AkaOCR core
# _____________________________________________________________________________

# This file contain sh script to update feature for public repo
# _____________________________________________________________________________
# """
git remote -v
git remote add public https://github.com/qai-research/Efficient_Text_Detection.git
git checkout qaicom
git status
git checkout master -- akaocr/
git reset HEAD akaocr/models/detec/resnet_fpn_heatmap.py akaocr/models/modules/backbones/Resnet_Extractor.py akaocr/synthtext/* akaocr/data/*
git status
git commit -m "Update features"
git push origin qaicom
git push -f public qaicom