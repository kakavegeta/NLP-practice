import config
import csv
import re
import numpy as np
from stopwords import STOPWORDS
from utils import onehot_encode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def get_phrase(datapath, phrase_col):
    phs = []
    with open(datapath, "rt") as f:
        fcsv = csv.reader(f, delimiter = "\t")
        header = next(f)
        for row in fcsv:
            phs.append(row[phrase_col])
    return phs

def get_label(datapath, label_col=3):
    labels = []
    with open(datapath, "rt") as f:
        fcsv = csv.reader(f, delimiter = "\t")
        header = next(fcsv)
        for row in fcsv:
            labels.append(int(row[label_col]))
    labels = np.array(labels)
    return onehot_encode(labels)

'''
### convert to lowercase, remove punctuation, and remove stopwords
def preprocess(corpus):
    _corpus = []
    for sentence in corpus:
        sentence = re.sub(r'[^\w\s]','',sentence) # remove punctuation
        sentence = sentence.lower() # to lowercase
        words = sentence.split(' ') 
        for word in words:
            if word in STOPWORDS: # remove stop words
                sentence = sentence.replace(word+' ', '')
        if sentence == '':
            continue
        if len(sentence)==1 and sentence in STOPWORDS:
            continue
        _corpus.append(sentence)
    return _corpus
'''

def to_lowercase(corpus):
    _corpus = []
    for s in corpus:
        _corpus.append(s.lower())
    return _corpus

def remove_punctuation(corpus):
    _corpus = []
    for s in corpus:
        _corpus.append(re.sub(r'[^\w\s]','',s))
    return _corpus


def get_vector(pathname):
    corpus = get_phrase(pathname, 2)
    corpus = to_lowercase(corpus)
    corpus = remove_punctuation(corpus)
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,1), dtype=np.float64)
    vec = vectorizer.fit_transform(corpus)
    # print(vec.shape)
    return vec.toarray()

def data_loader(x, y, batch_size = 32):
    size = x.shape[0]
    batch_num = ((size-1)//batch_size)+1
    indices = np.random.permutation(np.arange(size))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(batch_num):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size, size)
        yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id]


def main():
    phs = get_phrase(config.TRAIN, 2)
    #corpus = preprocess(phs)
    vec = get_vector(phs)
    print(vec.shape)
    y = get_label(config.TRAIN, 3)
    print(y.shape)

if __name__ == "__main__":
    main()