#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Tue December 29 09:30:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain unit test for config
_____________________________________________________________________________
"""

import sys
sys.path.append("../")

from engine.config import setup

config = setup("recog")
print(config)
config = setup("detec")
print(config)
# print(type(config.MODEL.RGB))
# if config.MODEL.RGB:
#     print("dump")
# setup("dump")
