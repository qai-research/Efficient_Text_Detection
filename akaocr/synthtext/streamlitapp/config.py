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

current_path = os.path.abspath('')
tree = current_path.split("/")
i = 1
while True:
    data = os.path.join("/".join(tree[:-i]), 'data')
    i += 1
    if os.path.exists(data) and 'backgrounds' in os.listdir(data):
        break
css_path = os.path.join(current_path,'synthtext/streamlitapp/style.css')
font_folder = os.path.join(data, 'fonts')
source_folder = os.path.join(data, 'sources')
object_folder = os.path.join(data, 'objects')
outputs_folder = os.path.join(data, 'outputs')
background_folder = os.path.join(data, 'backgrounds')
