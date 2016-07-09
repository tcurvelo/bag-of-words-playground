# -*- encoding: utf8 -*-
import types
import unittest
from bag_of_words import (
    clean_text,
    get_histograms,
    get_text_list_from_disk,
    tokenize,
)
from nltk import FreqDist


class BagOfWordsTestCase(unittest.TestCase):

    def test_clean_test(self):
        text = u'Coração Melão!?'
        expected = 'coracao melao  '
        self.assertEqual(clean_text(text), expected)

    def test_tokenizer(self):
        text = 'quem nasce tatu morre cavando'
        expected = ['nasc', 'tatu', 'morr', 'cav']
        self.assertEqual(tokenize(text), expected)

    def test_get_texts_generator_from_data_dir(self):
        texts = get_text_list_from_disk('tests/fixtures/')
        self.assertIsInstance(texts, types.GeneratorType)
        self.assertEqual(len([t for t in texts]), 2)

    def test_get_histogram(self):
        texts = get_text_list_from_disk('tests/fixtures/')
        histogram = reduce(lambda a, b: a+b, get_histograms(texts))
        expected = FreqDist({
            u'ai': 1,
            u'azu': 1,
            u'bem': 1,
            u'quer': 1,
            u'ros': 2,
            u'vermelh': 2,
            u'violet': 1,
            })
        self.assertIsInstance(histogram, FreqDist)
        self.assertEqual(expected, histogram)

if __name__ == '__main__':
    unittest.main()
