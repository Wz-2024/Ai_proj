import torch
import unicodedata
import re

use_cuda=torch.cuda.is_available()
if use_cuda:
    device=torch.device('cuda')
else:
    device=torch.device('cpu')

#start of sentence   end of sentence
SOS_token=0
EES_token=1

class Lang:
    def __init__(self,name):
        self.name=name
        self.word2count={}#统计每个词出现的次数
        self.word2index={}#word->index
        self.index2word={0:'SOS',1:'EOS'}#index->word
        self.n_word=2

    def addSentence(self,sentence):
        for word in sentence.split(''):
            self.addWord(word)



    def addWord(self,word):
        #词是新来的
        if word not in self.word2index:
            self.word2index[word]=self.n_word
            self.word2count[word]=1
            self.index2word[self.n_word]=word
            self.n_word+=1
        #词已经存在过
        else:
            self.word2count+=1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1,lang2,reverse=False):
    print("Reading lines...")
    #