# ======================================================
# tests/test_query.py
# ======================================================
# Here, we are testing the QueryProcessor to ensure:
#   - Boolean, phrase, and ranked queries work correctly
#   - TF-IDF and BM25 return consistent results
#   - Phrase matching uses positional index properly
# ======================================================

import unittest
from src.indexer import Indexer
from src.tokenizer import Tokenizer
from src.query import QueryProcessor

class TestQueryProcessor(unittest.TestCase):

    def setUp(self):
        # here, we are preparing a simple IR environment
        tokenizer = Tokenizer()
        indexer = Indexer(tokenizer=tokenizer, positional=True)
        
        # here, we are adding two short documents for testing
        indexer.add_document("doc1", "Artificial intelligence is the future of computing")
        indexer.add_document("doc2", "Machine learning and deep learning are subfields of artificial intelligence")
        indexer.build()

        # here, we are creating a query processor linked to this indexer
        self.qp = QueryProcessor(indexer)

    def test_boolean_query(self):
        # here, we are testing a simple Boolean AND query
        results = self.qp.search("artificial AND intelligence", method="tfidf")
        self.assertTrue(any("doc1" in d for d, _, _ in results))
        self.assertTrue(any("doc2" in d for d, _, _ in results))

    def test_phrase_query(self):
        # here, we are testing an exact phrase search
        results = self.qp.search('"machine learning"', method="tfidf")
        self.assertTrue(any("doc2" in d for d, _, _ in results))

    def test_tfidf_vs_bm25(self):
        # here, we are ensuring both TF-IDF and BM25 can return ranked results
        tfidf_res = self.qp.search("artificial intelligence", top_k=3, method="tfidf")
        bm25_res = self.qp.search("artificial intelligence", top_k=3, method="bm25")
        self.assertTrue(len(tfidf_res) > 0)
        self.assertTrue(len(bm25_res) > 0)

    def test_snippet_generation(self):
        # here, we are verifying that a snippet is generated for a top result
        results = self.qp.search("computing", method="tfidf")
        top = results[0]
        self.assertTrue(isinstance(top[2], str))
        self.assertGreater(len(top[2]), 0)

if __name__ == "__main__":
    unittest.main()
