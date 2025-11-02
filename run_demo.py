# ======================================================
# run_demo.py
# ======================================================
import glob
import os

from src.Indexer import Indexer
from src.Query import QueryProcessor


def build_index_from_sample():
    print("=== Building the inverted index from sample documents ===")

    indexer = Indexer()

    sample_dir = os.path.join("data", "sample_docs")
    paths = sorted(glob.glob(os.path.join(sample_dir, "*.txt")))

    # If no documents are found, warn the user.
    if not paths:
        print("No sample docs found in data/sample_docs.")
        return None

    indexer.build_index(paths)

    terms = indexer.trie.keys()

    print(f"Indexed {indexer.N()} documents.")
    print(f"Vocabulary size: {len(terms)} unique tokens.")

    for term in terms[:10]:
        print(f"=== {term} ===")
        print("meta: ", indexer.get_meta(term))
        print("postings: ", indexer.get_postings(term))

    return indexer


def demo_queries(indexer):
    qp = QueryProcessor(indexer)

    # ltc.ltc query
    print("=== ltc.ltc top results for query: 'information retrieval' ===")
    results_tfidf = qp.search("information retrieval", top_k=5, method="tfidf")
    if not results_tfidf:
        print("No results found for query.")
    else:
        for doc_id, score in results_tfidf:
            print(f"{doc_id}\t{score:.4f}")

    # ntu.Lpc query
    print("=== ntu.Lpc top results for query: 'information retrieval' ===")
    results_tfidf = qp.search(
        "information retrieval", top_k=5, method="tfidf", scheme="ntu.Lpc"
    )
    if not results_tfidf:
        print("No results found for query.")
    else:
        for doc_id, score in results_tfidf:
            print(f"{doc_id}\t{score:.4f}")

    # BM25 query
    print("=== BM25 top results for query: 'information retrieval' ===")
    results_bm25 = qp.search("information retrieval", top_k=5, method="bm25")
    if not results_bm25:
        print("No results found for BM25 query.")
    else:
        for doc_id, score in results_bm25:
            print(f"{doc_id}\t{score:.4f}")

    # Phrase query
    print('=== Phrase search for: "machine learning" ===')
    results_phrase = qp.search('"machine learning"', top_k=5, method="tfidf")
    if not results_phrase:
        print("No documents contain the exact phrase.")
    else:
        for doc_id, score in results_phrase:
            print(f"{doc_id}\t{score:.4f}")

    print("=== Boolean search for: Belgium AND Intelligence")
    results_phrase = qp.search(
        "Belgium AND Intelligence", top_k=5, method="tfidf"
    )
    if not results_phrase:
        print("No results found for query.")
    else:
        for doc_id, score in results_phrase:
            print(f"{doc_id}\t{score:.4f}")


if __name__ == "__main__":
    indexer = build_index_from_sample()
    if indexer:
        demo_queries(indexer)
