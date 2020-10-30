from collections import Counter

class Tokenizer(object):
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

    def word_tokenize(self, source):
        pass

    def sentence_tokenize(self, source):
        pass

class JapaneseTokenizer(Tokenizer):

    def __init__(self):
        self.wakati = Mecab.Tagger(self.tagger)

    def word_tokenize(self, source):
        list_words = self.wakati.parse(source).split()
        return list_words

    def sentence_tokenize(self, source):
        pass

class AlphabetTokenizer(Tokenizer):

    def __init__(self):
        pass

    def word_tokenize(self, source):
        return nltk.tokenize.word_tokenize(source)

    def sentence_tokenize(self, source):
        return nltk.tokenize.sent_tokenize(source)
        