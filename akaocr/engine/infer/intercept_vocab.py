#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Ngoc Nghia - Nghiann3
Created Date: Fri March 12 13:00:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

This file contains method for prediction with specific vocab
_____________________________________________________________________________
"""
import torch
from utils.file_utils import read_vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InterceptVocab():
    def __init__(self, vocab, converter):
        """
        vocab: list vocab or path to file vocab.txt
        """
        if isinstance(vocab, str):
            self.vocab = read_vocab(vocab)
        else:
            self.vocab = vocab
        self.vocab.append('[s]')
        self.index = []
        for i in range(len(self.vocab)):
            indx = converter.character.index(self.vocab[i])
            self.index.append(indx)
        
    def intercept_vocab(self, preds):
        tem = torch.zeros(preds.shape).to(device)
        preds = preds - preds.min()
        for i in range(len(self.index)):
            tem[0,:,self.index[i]] = 1
        return torch.mul(tem, preds)