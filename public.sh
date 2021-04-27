#!/bin/bash
git remote -v
git remote add public https://github.com/qai-research/Efficient_Text_Detection.git
cd ocr-components
git checkout qaicom
git status
git checkout master -- akaocr/
git reset HEAD akaocr/models/detec/resnet_fpn_heatmap.py akaocr/models/modules/backbones/Resnet_Extractor.py akaocr/synthtext/* akaocr/data/*
git status
git commit -m "Update code"
git push origin qaicom
git push -f public qaicom