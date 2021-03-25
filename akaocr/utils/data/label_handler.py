#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Viet Bac - Bacnv6
Created Date: Mon November 03 10:00:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain label file reader + pre-process
_____________________________________________________________________________
"""

import json


class TextLableHandle:
    def __init__(self, character=None, sensitive=True, unknown="?", max_length=None):
        self.character = character
        self.sensitive = sensitive
        self.unknown = unknown
        self.max_length = max_length

    @staticmethod
    def normalize_label(label, character, sensitive=True, unknown="?", max_length=None):
        """
        pre-process label before training (cleaning + masking)
        :param label: input label (string)
        :param character: input vocab (list)
        :param sensitive: training with upper case data
        :param unknown: char to replace unknown chars(do not show up in vocab)
        :return: clean label
        """
        if not sensitive:  # case insensitive
            label = label.lower()
        if unknown:
            label = list(label)
            for i in range(len(label)):
                if label[i] not in character and label[i] != " ":
                    label[i] = unknown
            label = "".join(label)
        if max_length is None:
            return label
        else:
            if len(label) > max_length:
                label = None
            return label

    def __call__(self, label):
        return self.normalize_label(label, self.character, self.sensitive, self.unknown, self.max_length)


class JsonLabelHandle:
    """
    Load json label
    """

    @staticmethod
    def load_json(label):
        return json.loads(label)

    def __call__(self, label):
        return self.load_json(label)
