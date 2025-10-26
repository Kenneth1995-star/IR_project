# ======================================================
# tests/test_tokenizer.py
# ======================================================
# Here, we are testing the Tokenizer component to ensure:
#   - it correctly lowercases text
#   - removes stopwords and digits
#   - properly identifies word tokens
#   - supports optional stemming
# ======================================================

import unittest
from src.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):

    def setUp(self):
        # here, we are creating a default tokenizer before each test
        self.tokenizer = Tokenizer()

    def test_basic_tokenization(self):
        # here, we are checking that the tokenizer splits simple text correctly
        text = "Artificial Intelligence is amazing!"
        tokens = self.tokenizer.tokenize(text)
        self.assertIn("artificial", tokens)
        self.assertIn("intelligence", tokens)

    def test_stopword_removal(self):
        # here, we are ensuring that common stopwords like 'the' are removed
        text = "The cat sat on the mat."
        tokens = self.tokenizer.tokenize(text)
        self.assertNotIn("the", tokens)
        self.assertIn("cat", tokens)
        self.assertIn("mat", tokens)

    def test_number_removal(self):
        # here, we are checking that digits are not tokenized as words
        text = "AI has 1234 models"
        tokens = self.tokenizer.tokenize(text)
        self.assertNotIn("1234", tokens)
        self.assertIn("ai", tokens)

    def test_stemming(self):
        # here, we are checking stemming if NLTK’s PorterStemmer is available
        try:
            stem_tokenizer = Tokenizer(use_stemmer=True)
            tokens = stem_tokenizer.tokenize("running runs runner")
            self.assertTrue(all(t.startswith("run") for t in tokens))
        except ImportError:
            self.skipTest("NLTK not installed; skipping stemming test")

if __name__ == "__main__":
    unittest.main()
