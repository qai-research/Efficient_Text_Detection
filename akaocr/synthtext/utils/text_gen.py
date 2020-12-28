#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Vu Hoang Viet - VietVH9
Created Date: NOV - 2020
Project : AkaOCR core
_____________________________________________________________________________



The Module Has Been Build To Collect all generator of number and text
_____________________________________________________________________________
"""
import os
import sys
import json
import numpy as np
import random


class NumbericGenerator:
    """
    Text Generator with different options
    """
    @staticmethod
    def randint(low, high=None):
        """
        Return random integers from low (inclusive) to high (exclusive)
        """
        return np.random.randint(low, high)

    @staticmethod
    def randfloat(low, high=None, dec=0):
        """
        Return random float from low (inclusive) to high (exclusive)
        """
        m = 10 ** dec
        if high is not None:
            return self.randint(low * m, high * m) / m
        return self.randint(low * m) / m

    def gen(self, low, high, int_percent, num_samples=10, dec=0):
        """
        Return random list number from low (inclusive) to high (exclusive) with the rate defined
        """

        results = []
        for _ in range(num_samples):
            if np.random.rand() < int_percent:
                results.append(self.randint(low, high=high))
            else:
                if dec == 0:
                    results.append(self.randfloat(low, high=high, dec=dec))
                else:
                    results.append(self.randfloat(low, high=high, dec=np.random.randint(dec)))
        return results


class TextGenerator:
    """
    Text Generator with different options
    """
    def __init__(self, source_text_path, vocab_group_path=None, min_text_length=1, max_text_length=15,
                 replace_percentage=0.5):

        with open(source_text_path, "r", encoding='utf-8-sig') as f:
            self.source_sentences = f.read().strip().split("\n")

        self.vocab_dict = {}
        if vocab_group_path is not None:
            with open(vocab_group_path, "r", encoding='utf-8-sig') as f:
                vocab_dict = json.loads(f.read())
            for group in vocab_dict:
                for word in group:
                    if word != '':
                        self.vocab_dict[word] = group

        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.replace_percentage = replace_percentage
        self.short_percentage = 0.8

    def generate(self, opt_len=None):
        """
        Gen text with len
        """
        if type(opt_len) == int:
            min_text_length = max(1, opt_len - 1)
            max_text_length = max(2, opt_len + 1)
        else:
            if opt_len is None:
                min_text_length = self.min_text_length
                max_text_length = self.max_text_length
            elif type(opt_len) is tuple and len(opt_len) == 2:
                min_text_length, max_text_length = opt_len
            else:
                raise ValueError

        template = random.choice(self.source_sentences)

        if len(template) > max_text_length:
            start_chars = random.randint(0, len(template) - max_text_length)
            if random.random() > self.short_percentage or max_text_length / 2 < min_text_length:
                length_text = random.randint(min_text_length, max_text_length)
            else:
                length_text = random.randint(min_text_length, int(max_text_length / 2))
            template = template[start_chars:start_chars + length_text]
        elif len(template) < min_text_length:
            template = self.generate(min_text_length)

        if not template.replace(" ", ''):
            template = self.generate(min_text_length)

        if random.random() < self.replace_percentage:
            for i in self.vocab_dict:
                if i in template:
                    for _ in range(template.count(i)):
                        template = template.replace(i, random.choice(self.vocab_dict[i]), 1)

        return template
