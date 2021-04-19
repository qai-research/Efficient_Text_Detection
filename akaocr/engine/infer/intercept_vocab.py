import torch
from utils.file_utils import read_vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InterceptVocab():
    def __init__(self, vocab, converter):
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