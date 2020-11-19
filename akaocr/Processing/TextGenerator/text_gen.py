import os
import sys
import json
import numpy as np
import random

class NumbericGenerator():

    def __init__(self):
        """
        A new NumbericGenerator object.
        """
        
    def randint(self,low, high = None):
        """
        Return random integers from low (inclusive) to high (exclusive)
        """
        return np.random.randint(low,high)

    def randfloat(self,low,high = None, dec=0):
        """
        Return random float from low (inclusive) to high (exclusive)
        """
        m = 10**dec
        if high is not None:
            return self.randint(low*m, high*m)/m
        return self.randint(low*m)/m

    def gen(self, low, high, int_percent, num_samples = 10, dec = 0):
        """
        Return random list number from low (inclusive) to high (exclusive) with the rate defined
        """

        results = []
        for _ in range(num_samples):
            if np.random.rand()<int_percent:
                results.append(self.randint(low, high = high))
            else:
                try:
                    results.append(self.randfloat(low, high = high, dec = dec))
                except:
                    results.append(self.randfloat(low, high = high, dec = np.random.randint(dec)))
        return results

class TextGenerator():

    def __init__(self, source_text_path, vocab_group_path=None, min_length=1, max_length=15, replace_percentage=0.5):

        with open(source_text_path, "r", encoding='utf-8-sig') as f:
            self.source_sentences = f.read().strip().split("\n")

        self.vocab_dict = {}
        if vocab_group_path is not None:
            with open(vocab_group_path, "r", encoding='utf-8-sig') as f:
                vocab_dict = json.loads(f.read())
            for l in vocab_dict:
                for j in l:
                    if j!='':
                        self.vocab_dict[j]=l
                
        self.min_length = min_length
        self.max_length = max_length
        self.replace_percentage = replace_percentage
        self.short_percentage = 0.8

    def generate(self, opt_len = None):

        if type(opt_len) == int:
            min_length = max(1,opt_len - 1)
            max_length = max(2,opt_len + 1)
        else:
            try:
                min_length, max_length = opt_len
            except:
                min_length = self.min_length
                max_length = self.max_length

        template = random.choice(self.source_sentences)

        if len(template) > max_length:
            start_chars = random.randint(0, len(template) - max_length)
            if random.random() > self.short_percentage or max_length / 2 < min_length:
                length_text = random.randint(min_length, max_length)
            else:
                length_text = random.randint(min_length, int(max_length / 2))
            template = template[start_chars:start_chars + length_text]
        elif len(template) < min_length:
            template = self.generate(opt_len-1)

        if not template.replace(" ",''):
            template = self.generate(opt_len+1)

        r = random.random()
        if random.random()<self.replace_percentage:
            for i in self.vocab_dict:
                if i in template:
                    for _ in range(template.count(i)):
                        template = template.replace(i,random.choice(self.vocab_dict[i]), 1)

        return template
        