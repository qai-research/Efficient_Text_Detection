# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Fri November 25 16:34:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain sample training function
_____________________________________________________________________________
"""

import sys
sys.path.append("../")

import os
import torch

from engine.config import setup


def main():
    args = setup("detec")


if __name__ == '__main__':
    main()