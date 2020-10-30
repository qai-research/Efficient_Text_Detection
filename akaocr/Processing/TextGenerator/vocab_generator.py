from collections import Counter

class Vocabulary(object):
    def __init__(self, **kwargs):
        """
        A new Vocabulary object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.            
        """
        for key, val in kwargs.items():
            setattr(self, key, val)

    def builder(self, source_path, tokenizer):

        counter = Counter()
        with open(source_path,'r', encoding = 'utf8') as reader:
            for line in reader:
                counter.update(countertokenizer.word_tokenize(line))
        