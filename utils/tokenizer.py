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

        self.PERIOD = "。"
        self.PERIOD_SPECIAL = "__PERIOD__"
        self.PATTERNS = [r"（.*?）", r"「.*?」",]  

    def word_tokenize(self, source):
        list_words = self.wakati.parse(source).split()
        return list_words
        
    def conv_period(self,item):
        return item.group(0).replace(self.PERIOD, self.PERIOD_SPECIAL)

    def sentence_tokenize(self, source):  

        for pattern in self.PATTERNS:
            pattern = re.compile(pattern)  # type: ignore
            document = re.sub(pattern, self.conv_period, document)

        result = []
        for line in source.split("\n"):
            line = line.rstrip()
            line = line.replace("\n", "")
            line = line.replace("\r", "")
            line = line.replace("。", "。\n")
            sentences = line.split("\n")

            for sentence in sentences:
                if not sentence:
                    continue

                period_special = self.PERIOD_SPECIAL
                period = self.PERIOD
                sentence = sentence.replace(period_special, period)
                result.append(sentence)

        return result

class AlphabetTokenizer(Tokenizer):

    def __init__(self):
        pass

    def word_tokenize(self, source):
        return nltk.tokenize.word_tokenize(source)

    def sentence_tokenize(self, source):
        return nltk.tokenize.sent_tokenize(source)
        