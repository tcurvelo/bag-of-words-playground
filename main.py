from nltk.stem.snowball import PortugueseStemmer
from unidecode import unidecode

import nltk
import numpy as np
import os
import re


class Histogram(nltk.FreqDist):
    stemmer = PortugueseStemmer()
    stops = {unidecode(word) for word in nltk.corpus.stopwords.words('portuguese')}

    def __init__(self, *args, **kw):
        if 'text' in kw and isinstance(kw['text'], unicode):
            # remove accents
            text = unidecode(kw['text'])

            # remove non-letters from text
            text = re.sub('[^a-z]', ' ', text.lower())

            tokens = nltk.word_tokenize(text)

            # stemming tokens, if not a stopword
            tokens = [
                Histogram.stemmer.stem(token)
                for token in tokens
                if token not in Histogram.stops
            ]
            nltk.FreqDist.__init__(self, tokens)
        else:
            nltk.FreqDist.__init__(self, *args, **kw)


    @classmethod
    def from_data_dir(cls, basedir):
        hist = Histogram()

        for root, dir, files in os.walk(basedir):
            for file in files:
                with open(root + '/' + file) as f:
                    text = f.read().decode('utf8')
                    hist += Histogram(text=text)

        return hist


class FeatureSet(dict):
    def __init__(self, histogram_data, size):
        dict.__init__(self, {
            word[0]: index for (index, word) in enumerate(
                histogram_data.most_common(size)
            )
        })

    def get_feature_array(self, text):
        hist = Histogram(text=text)
        array = np.zeros(shape=(len(self),), dtype=np.integer)
        for token in hist:
            if token in self:
                array[self[token]]=hist[token]
        return array
