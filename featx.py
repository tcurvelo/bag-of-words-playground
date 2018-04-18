from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures

import collections
import os
import re


def bag_of_words(words):
    return {word: True for word in words}


def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))


def bag_of_non_stopwords(words, stopfile='portuguese'):
    badwords = stopwords.words(stopfile)
    words = bag_of_words_not_in_set(words, badwords)
    return bag_of_words(words)


def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return {**bag_of_non_stopwords(words),  **bag_of_words(bigrams)}


def clean_up(text):
    text =  re.sub('\W+', ' ', text.lower())
    words = filter(bool, text.split(' '))
    return list(words)


def get_text_list_from_disk(basedir):
    for root, __, textfiles in os.walk(basedir):
        for file in textfiles:
            with open(root + '/' + file) as f:
                text = f.read()
                yield text


def label_feats_from_dir(datadir='data', feat_detector=bag_of_bigrams_words):
    label_feats = collections.defaultdict(list)
    for label in os.listdir(datadir):
        for text in get_text_list_from_disk(f'{datadir}/{label}'):
            feats = feat_detector(clean_up(text))
            label_feats[label].append(feats)
    return label_feats


def label_feats_from_dataframe(data, feat_detector=bag_of_bigrams_words):
    label_feats = collections.defaultdict(list)
    for index, row in data.iterrows():
        label = row['label']
        text = row['text']
        feats = feat_detector(clean_up(text))
        label_feats[label].append(feats)
    return label_feats


def split_label_feats(lfeats, split=0.75):
    train_feats = []
    test_feats = []
    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])
    return train_feats, test_feats


if __name__ == '__main__':
    labeled_feature_set = label_feats_from_dir()
    train_feats, test_feats = split_label_feats(labeled_feature_set, 0.8)
    nb_classifier = NaiveBayesClassifier.train(train_feats)
    print(accuracy(nb_classifier, test_feats))
