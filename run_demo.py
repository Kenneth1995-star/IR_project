# ======================================================
# run_demo.py
# ======================================================
import glob
import os

from src.Indexer import Indexer
from src.Query import QueryProcessor


def build_index_from_sample():
    print("=== Building the inverted index from sample documents ===")

    paths = sorted(glob.glob(os.path.join("data", "**"), recursive=True))

    # If no documents are found, warn the user.
    if not paths:
        print("No sample docs found in data/sample_docs.")
        return None

    indexer = Indexer()

    indexer.build_index(paths)

    terms = indexer.trie.keys()

    print(f"Indexed {indexer.N()} documents.")
    print(f"Vocabulary size: {len(terms)} unique tokens.")

    return indexer


def demo_queries(indexer):
    qp = QueryProcessor(indexer)
    docs = qp.indexer.get_doc_keys()

    # ltc.ltc query
    print("=== ltc.ltc top results for query: 'Charizard Xenophobe' ===")
    results_tfidf = qp.search("Charizard Xenophobe", method="tfidf")
    if not results_tfidf:
        print("No results found for query.")
    else:
        for doc_id, score in results_tfidf:
            print(f"{doc_id}\t{docs[doc_id]}\t{score:.4f}")

    # ntu.Lpc query
    print("=== ntu.Lpc top results for query: 'Charizard Xenophobe' ===")
    results_tfidf = qp.search(
        "Charizard Xenophobe", method="tfidf", scheme="ntu.Lpc"
    )
    if not results_tfidf:
        print("No results found for query.")
    else:
        for doc_id, score in results_tfidf:
            print(f"{doc_id}\t{docs[doc_id]}\t{score:.4f}")

    # BM25 query
    print("=== BM25 top results for query: 'Charizard Xenophobe' ===")
    results_bm25 = qp.search("Charizard Xenophobe", method="bm25")
    if not results_bm25:
        print("No results found for BM25 query.")
    else:
        for doc_id, score in results_bm25:
            print(f"{doc_id}\t{docs[doc_id]}\t{score:.4f}")

    # Phrase query
    print('=== Phrase search for: "Harry Potter" ===')
    results_phrase = qp.search('"Harry Potter"', method="tfidf")
    if not results_phrase:
        print("No documents contain the exact phrase.")
    else:
        for doc_id, score in results_phrase:
            print(f"{doc_id}\t{docs[doc_id]}\t{score:.4f}")

    print("=== Boolean search for: Harry AND (NOT Potter) OR CHARIZARD")
    results_phrase = qp.search("Harry AND Potter", method="tfidf")
    if not results_phrase:
        print("No results found for query.")
    else:
        for doc_id, score in results_phrase:
            print(f"{doc_id}\t{docs[doc_id]}\t{score:.4f}")

    print(
        "=== Boolean and Phrase combination search for: "
        '"Harry Potter" OR CHARIZARD'
    )
    results_phrase = qp.search('"Harry Potter" OR CHARIZARD', method="tfidf")
    if not results_phrase:
        print("No results found for query.")
    else:
        for doc_id, score in results_phrase:
            print(f"{doc_id}\t{docs[doc_id]}\t{score:.4f}")


if __name__ == "__main__":
    indexer = build_index_from_sample()

    # Index files can be loaded in with .load()
    # indexer = Indexer(path="index/")
    # indexer.load()

    if indexer:
        demo_queries(indexer)
