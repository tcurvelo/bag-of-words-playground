# -*- encoding: utf8 -*-
import bag_of_words
import numpy as np
import types
import unittest

from nltk import FreqDist


class BagOfWordsTestCase(unittest.TestCase):

    def test_clean_test(self):
        text = u'Coração Melão!?'
        expected = 'coracao melao  '
        self.assertEqual(bag_of_words.clean_text(text), expected)

    def test_tokenizer(self):
        text = 'quem nasce tatu morre cavando'
        expected = ['nasc', 'tatu', 'morr', 'cav']
        self.assertEqual(bag_of_words.tokenize(text), expected)

    def test_get_texts_generator_from_data_dir(self):
        texts = bag_of_words.get_text_list_from_disk('tests/fixtures/')
        self.assertIsInstance(texts, types.GeneratorType)
        self.assertEqual(len([t for t in texts]), 2)

    def test_get_histogram(self):
        texts = bag_of_words.get_text_list_from_disk('tests/fixtures/')
        hist = reduce(lambda a, b: a+b, bag_of_words.get_histograms(texts))
        expected = FreqDist({
            u'ai': 1,
            u'azu': 1,
            u'bem': 1,
            u'quer': 1,
            u'ros': 2,
            u'vermelh': 2,
            u'violet': 1,
            })
        self.assertIsInstance(hist, FreqDist)
        self.assertEqual(expected, hist)

    def test_instantiate_bag_of_words_from_a_data_dir(self):
        bag = bag_of_words.BagOfWords.from_data_dir('tests/fixtures', 3)
        self.assertEqual(
            bag.features,
            {'ros': 0, 'vermelh': 1, 'quer': 2}
        )
        texts = bag_of_words.get_text_list_from_disk('tests/fixtures')
        arr = bag.get_features_array(texts)
        self.assertTrue(np.array_equal(
            arr,
            np.array([[1, 1, 0], [1, 1, 1]])
        ))


if __name__ == '__main__':
    unittest.main()
