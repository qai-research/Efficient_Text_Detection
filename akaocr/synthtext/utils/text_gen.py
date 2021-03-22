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
import MeCab


class Numberic:
    """
    Text Generator with different options
    """

    def __init__(self, low, high = None):
        self.low = low
        self.high = high
        self.min_text_length = len(str(self.low**2))
        self.max_text_length = len(str(self.high**2))
    
    def gen(self, opt_len):
        if self.high == None:
            if self.low>=10:
                min_p = 1
                max_p = len(str(int(self.low)))+1
                out = np.random.randint(1,max_p)
                new_M = min(10**out, self.low)
                new_m = max(10**(out-1), self.low)
            else:
                out = 1
                new_M = self.low
                new_m = 0
        else:
            min_p = len(str(int(self.low)))
            max_p = len(str(int(self.high)))
            out = np.random.randint(min_p, max_p)
            new_M = min(10**out, self.high)
            new_m = max(10**(out-1), self.low)
            
        rand = np.random.random_sample()    
        integer = np.random.randint(new_m, new_M)      
        results = integer+rand
        if type(opt_len) == str:
            text_len = len(opt_len)-len(str(integer))
        else:
            text_len = (np.random.randint(self.min_text_length,self.max_text_length)-len(str(integer)))
        txt = "{value:.%sf}"%max(0,text_len)
        txt = txt.format(value = results)
        return txt


class Text:
    """
    Text Generator with different options
    """
    def __init__(self, source_text_path, vocab_group_path=None, min_text_length=1, max_text_length=15,
                 replace_percentage=0.5, text_gen_type = 'randoms'):

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
        self.text_gen_type = text_gen_type

    def gen(self, opt_len = None):
        if type(opt_len) == str:
            text_len = len(opt_len)
        if self.text_gen_type == 'randoms':
            return self.random_generate(text_len)
        elif self.text_gen_type == 'words':
            return self.word_generate(text_len)
        elif self.text_gen_type == 'numberics':
            self.vocab_dict = {str(i):[str(j) for j in range(10)] for i in range(10)}
            return self.random_generate(text_len)
        else:
            raise ValueError("Text gen type must be 'ramdoms', 'words' or 'numberics'.")

    def random_generate(self, opt_len=None):
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
            template = self.random_generate(min_text_length)

        if not template.replace(" ", ''):
            template = self.random_generate(min_text_length)

        if random.random() < self.replace_percentage:
            for i in self.vocab_dict:
                if i in template:
                    for _ in range(template.count(i)):
                        template = template.replace(i, random.choice(self.vocab_dict[i]), 1)

        return template

    def word_generate(self, opt_len=None):
        """
        Gen text as words
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
        wakati = MeCab.Tagger("-Owakati")
        template = [word for word in wakati.parse(template).split()]
        word = ''
        while len(word)<min_text_length or len(word)>max_text_length:
            s_ind = np.random.randint(len(template))
            l_ind = np.random.randint(len(template)-s_ind)
            more = random.choice(template)
            word = "".join(template[s_ind:s_ind+l_ind])           
        return word


class TextGenerator:

    def __init__(self, source_text_path, vocab_group_path=None, min_text_length=1, max_text_length=15,
                 replace_percentage=0.5, text_gen_type = 'words'):
        if not os.path.exists(source_text_path):
            self.generator = Numberic(low = 10**(min_text_length//2),
                                       high = 10**(max_text_length//2))
        else:
            self.generator = Text(source_text_path, 
                                vocab_group_path, 
                                min_text_length, 
                                max_text_length, 
                                replace_percentage,
                                text_gen_type)

    def generate(self, opt_len = None):
        return self.generator.gen(opt_len = opt_len)