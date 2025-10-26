# ======================================================
# run_demo.py
# ======================================================
# Here, we are building and demonstrating the complete
# Information Retrieval (IR) system that you implemented
# using your Tokenizer, Indexer, and QueryProcessor classes.
#
# This file:
#   1. Builds an inverted index from sample documents.
#   2. Saves the index to disk (lexicon + postings).
#   3. Demonstrates query processing using:
#        - TF-IDF scoring
#        - BM25 scoring
#        - Phrase search
#   4. Prints intuitive outputs for interpretation.
#
# ======================================================

from src.tokenizer import Tokenizer
from src.indexer import Indexer
from src.query import QueryProcessor
import os, glob

# ======================================================
# Function: build_index_from_sample()
# ======================================================
# Here, we are:
#   - Initializing the tokenizer and indexer.
#   - Loading all text files from data/sample_docs.
#   - Adding each document into the inverted index.
#   - Building and saving the index to disk.
# ======================================================

def build_index_from_sample():
    print("=== Building the inverted index from sample documents ===")
    
    # Here, we are initializing a Tokenizer instance.
    # It performs lowercasing, stopword removal, and tokenization.
    tokenizer = Tokenizer()

    # Here, we are initializing the Indexer.
    # positional=True → stores term positions (for phrase search)
    # updatable=True  → allows new documents to be added later
    indexer = Indexer(tokenizer=tokenizer, positional=True, updatable=True)

    # Here, we are defining the sample document directory.
    sample_dir = os.path.join("data", "sample_docs")
    
    # We retrieve all .txt files in that folder.
    paths = sorted(glob.glob(os.path.join(sample_dir, "*.txt")))

    # If no documents are found, warn the user.
    if not paths:
        print("No sample docs found in data/sample_docs. Add some .txt files to demo.")
        return None

    # Here, we are looping through each document file and adding it to the index.
    for path in paths:
        with open(path, "r", encoding="utf8") as f:
            text = f.read()
        doc_id = os.path.basename(path)
        indexer.add_document(doc_id, text)

    # After adding all documents, we finalize index building.
    indexer.build()

    print(f"Indexed {len(indexer.doc_texts)} documents.")
    print(f"Vocabulary size: {len(indexer.postings_lists)} unique tokens.")
    
    # Save the lexicon and postings list to disk.
    index_prefix = "data/index"
    indexer.save_postings_to_disk(index_prefix)
    
    print(f"Saved postings and lexicon files to: {index_prefix}.postings and {index_prefix}.lexicon\n")
    
    return indexer

# ======================================================
# Function: demo_queries()
# ======================================================
# Here, we are:
#   - Demonstrating how a user query is processed.
#   - Executing it with multiple retrieval models (TF-IDF, BM25).
#   - Showing phrase query behavior (exact match).
# ======================================================

def demo_queries(indexer):
    qp = QueryProcessor(indexer)
    
    # Example 1: TF-IDF query
    print("=== TF-IDF top results for query: 'information retrieval' ===")
    results_tfidf = qp.search("information retrieval", top_k=5, method="tfidf")
    if not results_tfidf:
        print("No results found for TF-IDF query.")
    else:
        for doc_id, score, snippet in results_tfidf:
            print(f"{doc_id}\t{score:.4f}")
            print(f"...{snippet}...\n")

    # Example 2: BM25 query
    print("=== BM25 top results for query: 'information retrieval' ===")
    results_bm25 = qp.search("information retrieval", top_k=5, method="bm25")
    if not results_bm25:
        print("No results found for BM25 query.")
    else:
        for doc_id, score, snippet in results_bm25:
            print(f"{doc_id}\t{score:.4f}")
            print(f"...{snippet}...\n")

    # Example 3: Phrase query (exact sequence search)
    print('=== Phrase search for: "machine learning" ===')
    results_phrase = qp.search('"machine learning"', top_k=5, method="tfidf")
    if not results_phrase:
        print("No documents contain the exact phrase.")
    else:
        for doc_id, score, snippet in results_phrase:
            print(f"{doc_id}\t{score:.4f}")
            print(f"...{snippet}...\n")

# ======================================================
# Main execution section
# ======================================================
# Here, we are:
#   - Building the index only if needed.
#   - Running demo queries.
# ======================================================

if __name__ == "__main__":
    indexer = build_index_from_sample()
    if indexer:
        demo_queries(indexer)

