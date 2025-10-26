# ======================================================
# tests/test_indexer.py
# ======================================================
# Here, we are testing the Indexer to ensure:
#   - documents are tokenized and indexed correctly
#   - postings lists and positional indexes are built properly
#   - TF-IDF and document statistics are computed
# ======================================================

import unittest
from src.indexer import Indexer
from src.tokenizer import Tokenizer

class TestIndexer(unittest.TestCase):

    def setUp(self):
        # here, we are creating a tokenizer and positional indexer for testing
        self.tokenizer = Tokenizer()
        self.indexer = Indexer(tokenizer=self.tokenizer, positional=True)

        # here, we are adding small test documents
        self.indexer.add_document("doc1", "Artificial intelligence is cool")
        self.indexer.add_document("doc2", "Machine learning is part of artificial intelligence")
        self.indexer.build()

    def test_vocabulary_size(self):
        # here, we are checking that the vocabulary is not empty
        vocab = self.indexer.vocab()
        self.assertTrue(len(vocab) > 0)

    def test_postings_list_content(self):
        # here, we are verifying that the token 'artificial' appears in both documents
        postings = self.indexer.get_postings_list("artificial")
        doc_ids = [d for d, _ in postings]
        self.assertIn("doc1", doc_ids)
        self.assertIn("doc2", doc_ids)

    def test_positions_available(self):
        # here, we are verifying that positional information exists
        pos = self.indexer.get_positions("artificial")
        self.assertIsInstance(pos, dict)
        self.assertIn("doc1", pos)

    def test_tfidf_weights_exist(self):
        # here, we are confirming that TF-IDF weights were computed
        self.assertTrue(len(self.indexer.tfidf_postings) > 0)
        self.assertTrue(all(isinstance(w, dict) for w in self.indexer.tfidf_postings.values()))

    def test_document_norms(self):
        # here, we are checking that each document has a valid norm
        for doc_id, norm in self.indexer.doc_norms.items():
            self.assertGreater(norm, 0.0)

if __name__ == "__main__":
    unittest.main()
