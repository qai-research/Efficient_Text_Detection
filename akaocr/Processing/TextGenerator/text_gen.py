import os
import sys
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
                results.append(self.randint(low, high = high)  
            else:
                try:
                    results.append(self.randfloat(low, high = high, dec = dec))
                except:
                    results.append(self.randfloat(low, high = high, dec = np.random.randint(dec)))
        return results

class TextGenerator(self):

    def __init__(self, vocab_file, vocab_group=None, min_length=1, max_length=15, replace_percentage=0.5):

        with open(vocab_file, "r", encoding='utf-8-sig') as f:
            self.vocab_sentences = f.read().strip().split("\n")

        self.vocab_groups = {}
        if vocab_group is not None:
            with open(vocab_group, "r", encoding='utf-8-sig') as f:
                vocab_groups = json.loads(f.read())
            for l in vocab_groups:
                for j in l:
                    if j!='':
                        self.vocab_groups[j]=l
                
        self.min_length = min_length
        self.max_length = max_length
        self.replace_percentage = replace_percentage
        self.short_percentage = 0.8

    def generate(self):
        template = random.choice(self.vocab_sentences)

        if len(template) > self.max_length:
            start_chars = random.randint(0, len(template) - self.max_length)
            if random.random() > self.short_percentage or self.max_length / 2 < min_length:
                length_text = random.randint(min_length, max_length)self.
            else:
                length_text = random.randint(min_length, int(max_length / 2))
            template = template[start_chars:start_chars + length_text]
        elif len(template) < self.min_length:
            template = self.generate()

        r = random.random()
        if random.random()<self.replace_percentage:
            for i in self.vocab_groups:
                if i in template:
                    for _ in template.count(i):
                        template = template.replace(i,random.choice(self.vocab_groups[i]), 1)

        return template
        