from nltk.stem.snowball import PortugueseStemmer
from unidecode import unidecode

import nltk
import numpy as np
import os
import re


STEMMER = PortugueseStemmer()
STOPWORDS = {
    unidecode(word) for word in nltk.corpus.stopwords.words('portuguese')
}


def clean_text(text):
    # remove accents
    text = unidecode(text)
    # remove non-letters from text
    text = re.sub('[^a-z]', ' ', text.lower())
    return text


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    # stemming tokens, if not a stopword
    return [
        STEMMER.stem(token) for token in tokens
        if token not in STOPWORDS
    ]


def get_histograms(texts):
    for text in texts:
        text = clean_text(text)
        tokens = tokenize(text)
        yield nltk.FreqDist(tokens)


def get_text_list_from_disk(basedir):
    for root, __, files in os.walk(basedir):
        for file in files:
            with open(root + '/' + file) as f:
                text = f.read().decode('utf8')
                yield text


class BagOfWords(object):
    def __init__(self, texts, bag_size):
        histogram = reduce(lambda a,b: a+b, get_histograms(texts))
        self.features = {
            word[0]: index for (index, word) in
            enumerate(histogram.most_common(bag_size))
        }

    @classmethod
    def from_data_dir(cls, basedir, bag_size):
        return cls(get_text_list_from_disk(basedir), bag_size)

    def get_features_array(self, texts):
        X_array = np.empty(shape=(0, len(self.features)), dtype=np.integer)

        for text_hist in get_histograms(texts):
            row = np.zeros(shape=len(self.features), dtype=np.integer)
            for token in text_hist:
                if token in self.features:
                    row[self.features[token]] = text_hist[token]
            X_array = np.vstack( (X_array, row) )

        return X_array
