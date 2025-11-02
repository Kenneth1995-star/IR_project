import glob
import os
import unittest

from src.Indexer import Indexer
from src.Query import QueryProcessor


class TestQueryProcessor(unittest.TestCase):
    """
    Builds index and tests querying.
    For this test case, call indexer.load() before querying
    """
    def setUp(self):
        sample_dir = os.path.join("data", "sample_docs")
        paths = sorted(glob.glob(os.path.join(sample_dir, "*.txt")))
        indexer = Indexer(path="test_index/")
        indexer.build_index(paths)

        self.qp = QueryProcessor(indexer)

        self.addCleanup(self.qp.indexer.close)

    def test_boolean_query(self):
        self.qp.indexer.load()
        results = self.qp.search("artificial AND intelligence", method="tfidf")
        self.assertTrue(results)

    def test_non_positive_bool(self):
        self.qp.indexer.load()
        self.assertRaises(ValueError, self.qp.search, "NOT ARTIFICIAL")

    def test_phrase_query(self):
        self.qp.indexer.load()
        results = self.qp.search('"machine learning"', method="tfidf")
        self.assertTrue(results)
        results = self.qp.search('"information retrieval"')
        self.assertFalse(results)

    def test_bm25(self):
        self.qp.indexer.load()
        results = self.qp.search("AI", method="bm25")
        self.assertTrue(results)

    def test_smart_schemes(self):
        self.qp.indexer.load()
        schemes = [a + b + c for c in "ncub" for b in "ntp" for a in "nlabL"]
        for ddd in schemes:
            for qqq in schemes:
                results = self.qp.search(
                    "AI", method="tfidf", scheme=f"{ddd}.{qqq}"
                )
                self.assertTrue(results)


if __name__ == "__main__":
    unittest.main()
