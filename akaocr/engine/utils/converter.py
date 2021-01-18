# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________
Created By  : Nguyen Minh Trang - Trangnm5
Created Date: Fri January 15 14:50:00 VNT 2021
Project : AkaOCR core
_____________________________________________________________________________

Custom converters for recognition model
_____________________________________________________________________________
"""

import torch
import json
import numpy as np
import torch.nn.functional as F


class CTCLabelConverter(object):
    """
    Convert between text-label and text-index

    Attributes
    ----------
    dict : dict
        the character mapping char -> index
    character : list
        the character list

    Methods
    -------
    encode(text)
        Convert text-label into text-index.
    decode(text_index, length)
        Convert text-index into text-label.
    """
    def __init__(self, character):
        # character (list): set of the possible characters.
        dict_character = character

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, max_label_length=25):
        """Convert text-label into text-index.

        Parameters
        ----------
        text : str
            text labels of each image.
        max_label_length : int, optional, default: 25
            dummy argument

        Returns
        ------
        Tensor
            concatenated text index for CTCLoss. [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
        Tensor
            length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, text_index, length):
        """Convert text-index into text-label.

        Parameters
        ----------
        text_index : numpy array
            array of indexes
        length : list
            list of text label lengths

        Returns
        ------
        list
            list of text strings
        """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """
    Convert between text-label and text-index

    Attributes
    ----------
    dict : dict
        the character mapping char -> index
    character : list
        the character list

    Methods
    -------
    encode(text, max_label_length=25)
        perform OCR on the input Zone object
    """
    def __init__(self, character, device):
        self.device = device
        # character (list): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        self.character = list_token + character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, max_label_length=25):
        """Convert text-label into text-index.

        Parameters
        ----------
        text : str
            text labels of each image.
        max_label_length : int, optional, default: 25
            max length of text label in the batch

        Returns
        ------
        Tensor
            the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
            text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
        Tensor
            the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # max_label_length = max(length) # this is not allowed for multi-gpu setting
        max_label_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), max_label_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return batch_text.to(self.device), torch.IntTensor(length).to(self.device)

    def decode(self, text_index, length, **kwargs):
        """Convert text-index into text-label.

        Parameters
        ----------
        text_index : numpy array
            array of indexes
        length : list
            list of text label lengths

        Returns
        ------
        list
            list of text strings
        """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class AttnBeamLabelConverter(AttnLabelConverter):
    """
    Convert text-index into text-label and find the best scored text-label using bigram language model.

    Attributes
    ----------
    dict : dict
        the character mapping char -> index
    character : list
        the character list
    bigram_lm : dict
        pre-calculated bigram probability

    Methods
    -------
    encode(text, max_label_length=25)
        perform OCR on the input Zone object
    """
    def __init__(self, character, device, bigram_lm):
        super().__init__(character, device)
        with open(bigram_lm, 'r', encoding='utf-8-sig') as f:
            self.bigram_lm = json.loads(f.read(), encoding='utf-8-sig')

    def _cal_bigram_prob(self, w1, w2):
        lamb = self.bigram_lm['lambda'].get(w1, 0)
        p_bi = self.bigram_lm['bigram'].get(w1+w2, 0)
        p_uni = self.bigram_lm['unigram'].get(w2, 0)
        prob = lamb * p_bi + (1 - lamb) * p_uni
        return prob

    def _cal_sequence_score(self, seq):
        i = 0
        score = 0
        while i < len(seq) - 1:
            score += np.log(self._cal_bigram_prob(seq[i], seq[i+1]))
            i += 1
        return score

    def decode(self, text_index, length, beam_score=None):
        """Convert text-index into text-label and find the best scored text-label using bigram language model.

        Parameters
        ----------
        beam_score : numpy array, shape = batch_size, beam_size
            array of the sequences' beam scores
        text_index : numpy array, shape = batch_size, beam_size, num_steps
            array of indexes
        length : list
            list of text label lengths

        Returns
        ------
        list
            list of text strings
        """
        best_texts = []
        for idx, l in enumerate(length):
            bi_scores = []
            bm_scores = []
            text_seqs = []
            for k in range(text_index.size()[1]):  # beam_size
                text = ''.join([self.character[i] for i in text_index[idx, k, :]])
                text_seqs.append(text)

                bi_scores.append(self._cal_sequence_score(text))
                bm_scores.append(beam_score[idx, k])

            bi_scores_prob = F.softmax(torch.tensor(bi_scores))
            bm_scores_prob = F.softmax(torch.tensor(bm_scores))
            seq_scores_prob = F.softmax(bi_scores_prob + bm_scores_prob)
            conf, best_idx = seq_scores_prob.max(dim=0)

            pred_EOS = text_seqs[int(best_idx)].find('[s]')
            text = text_seqs[int(best_idx)][:pred_EOS]  # prune after "end of sentence" token ([s])
            best_texts.append([text, float(conf)])
        return best_texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""
    def __init__(self):
        self.n_count = 0
        self.sum = 0

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError  # ValueError that doesn't tell you what the wrong value was
