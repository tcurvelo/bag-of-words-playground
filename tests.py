# -*- encoding: utf8 -*-
import unittest
from bag_of_words import clean_text, tokenize

class BagOfWordsTestCase(unittest.TestCase):


    def test_clean_test(self):
        text = u'Coração'
        expected = 'coracao'
        self.assertEqual(clean_text(text), expected)

    def test_tokenizer(self):
        text = 'what is your favorite color'
        expected = ['what', 'is', 'your', 'favorite', 'color']
        self.assertEqual(tokenize(text), expected)


if __name__ == '__main__':
    unittest.main()
