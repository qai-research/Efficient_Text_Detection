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

from engine.config import setup, parse_base

def config_test():
    parse = parse_base()
    args = parse.parse_args()
    config_recog = setup("recog", args)
    # print(config_recog)
    # print("Vocab length is : ", len(config_recog.MODEL.VOCAB))
    config_detec = setup("detec", args)
    # print(config_detec)
    return config_recog, config_detec
