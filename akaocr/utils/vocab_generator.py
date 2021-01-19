from collections import Counter

class Vocabulary(object):
    def __init__(self, vocab_path, encoding = None, **kwargs):
        """
        A new Vocabulary object.

        Parameters
        ----------
        **kwargs : any
            Default values for the session state.            
        """
        self.vocab_path = vocab_path       
        self.encoding   = encoding if encoding is not None else "utf-8"
        for key, val in kwargs.items():
            setattr(self, key, val)

    def build(self):
        
        counter = Counter()
        try:
            with open(self.source_path,'r', encoding = self.encoding) as reader:
                for line in reader:
                    counter.update(self.tokenizer.word_tokenize(line))  
            with open(self.vocab_path, 'w', encoding = self.encoding) as writer:
                for word in counter:
                    writer.write(word)
                    writer.write("\n")            
            print("The vocab has been saved in %s"%self.vocab_path)
            
        except AttributeError:
            raise AttributeError("""The attribute 'source_path' is required to build vocab. \
                                Example: Vocabulary(vocab_path = '/path/to/vocab/file', \
                                                    encoding = 'utf-8', \
                                                    source_path = '/path/to/text/source/file' \
                                                    tokenizer = TheTokenizerObject) \
                                """)
        

    def load(self):
        with open(self.vocab_path, 'r', encoding = self.encoding) as reader:
            vocab = reader.reads().split("\n")
        return vocab