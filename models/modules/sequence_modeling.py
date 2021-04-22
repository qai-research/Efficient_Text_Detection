#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Minh Trang - Trangnm5
Created Date: Mon November 03 10:00:00 VNT 2020
Project : AkaOCR core
_____________________________________________________________________________

This file contain bidirectional lstm for recognition model
_____________________________________________________________________________
"""

import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : Tensor
            visual feature [batch_size x T x input_size]

        Returns
        ------
        Tensor
            contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(inputs)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
