# Information Retrieval System (IR Project)

### Course Project — University of Antwerp

---

## Project Overview

This project implements a **complete document search and retrieval system** from scratch using only standard Python data structures — fully compliant with the techniques taught in class.  
It **does not** use external search libraries like Lucene, Solr, or ElasticSearch.

It demonstrates:
- **Inverted index** and **positional index**
- **Boolean and phrase queries**
- **TF-IDF vector space model**
- **Okapi BM25 probabilistic model**
- **Evaluation metrics:** Precision, Recall, nDCG, DCG, Average Precision, Mean Average Precision, Reciprocal Rank, Mean Reciprocal Rank

### Dataset  
This project uses the dataset **“Wikipedia Movies”** available on Kaggle:  
https://www.kaggle.com/datasets/exactful/wikipedia-movies  

The dataset is derived from content sourced from the Wikimedia Foundation / Wikipedia community, and is provided under the [Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/) license.  
Attribution: Wikipedia contributors. 

---

## Project Structure
IR_project/
├── README.md
├── requirements.txt
├── run_demo.py
│
├── data/
│ └── sample_docs/
│ ├── doc1.txt
│ ├── doc2.txt
│ ├── doc3.txt
│ ├── doc4.txt
│ ├── doc5.txt
│ ├── doc6.txt
│ └── doc7.txt
│
├── src/
│ ├── init.py
│ ├── tokenizer.py # Tokenization and stopword filtering
│ ├── indexer.py # Inverted + positional index building
│ ├── query.py # Boolean, phrase, TF-IDF, BM25 search
│ └── extras.py # Evaluation metrics (Precision, Recall, nDCG, MAP, MRR)
│
└── tests/
├── test_tokenizer.py
├── test_indexer.py
└── test_query.py

##  Installation

1) **Activate your Conda environment**

```powershell
conda activate environment

2) Navigate to the github repository project folder
https://github.com/Kenneth1995-star/IR_project.git

3) Install dependencies
pip install -r requirements.txt
Note: nltk is optional (only needed if stemming is enabled). If stemming support is needed, install nltk and download the required resources:
pip install nltk
python -c "import nltk; nltk.download('punkt')"

4) Running the Demo

This demo script will:

Read .txt files from data/sample_docs/

Build an inverted index (with positional postings)

Save postings + lexicon to disk (data/index.postings, data/index.lexicon)

Run example queries (TF-IDF, BM25, phrase)

** Expected Output:
Indexed 7 documents, vocab size 247
Saved postings to data/index.postings and data/index.lexicon

TF-IDF top results for query: 'information retrieval'
doc5.txt        0.1575
...ms have transformed communication, allowing people to share information instantly across the world.
However, they also r...


BM25 top results for query: 'information retrieval'
doc5.txt        1.7307
...ms have transformed communication, allowing people to share information instantly across the world.
However, they also r...


Phrase search example for: "information retrieval"
(standard) PS C:\Users\kerne\Desktop\Information_Retrieval code> python run_demo.py
=== Building the inverted index from sample documents ===
Indexed 7 documents.
Vocabulary size: 247 unique tokens.
Saved postings and lexicon files to: data/index.postings and data/index.lexicon

=== TF-IDF top results for query: 'information retrieval' ===
doc5.txt        0.1575
......ms have transformed communication, allowing people to share information instantly across the world.
However, they also r......

=== BM25 top results for query: 'information retrieval' ===
doc5.txt        1.7307
......ms have transformed communication, allowing people to share information instantly across the world.
However, they also r......

=== Phrase search for: "machine learning" ===
doc2.txt        0.2380
...Machine learning is a subset of artificial intelligence focu......

doc1.txt        0.2214
......ntelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn.
AI is a br......

doc4.txt        0.1331
......, and physical activity, improving personalized healthcare.
Machine learning algorithms help discover new drugs by analy......

5) Running Tests

All tests use Python's built-in unittest and are located in the tests/ folder.
Run all tests: 
python -m unittest discover -s tests -p "test_*.py" -v

** Expeceted Output:
test_document_norms (test_indexer.TestIndexer) ... ok
test_positions_available (test_indexer.TestIndexer) ... ok
test_postings_list_content (test_indexer.TestIndexer) ... ok
test_tfidf_weights_exist (test_indexer.TestIndexer) ... ok
test_vocabulary_size (test_indexer.TestIndexer) ... ok
test_boolean_query (test_query.TestQueryProcessor) ... ok
test_phrase_query (test_query.TestQueryProcessor) ... ok
test_snippet_generation (test_query.TestQueryProcessor) ... ok
test_tfidf_vs_bm25 (test_query.TestQueryProcessor) ... ok
test_basic_tokenization (test_tokenizer.TestTokenizer) ... ok
test_number_removal (test_tokenizer.TestTokenizer) ... ok
test_stemming (test_tokenizer.TestTokenizer) ... ok
test_stopword_removal (test_tokenizer.TestTokenizer) ... ok

----------------------------------------------------------------------
Ran 13 tests in 0.012s

OK

** What the tests check:

test_tokenizer.py — tokenization, stopword removal, digit removal, optional stemming

test_indexer.py — document addition, postings list content, positional info, TF-IDF precomputation, docs norms

test_query.py — boolean queries (AND/OR/NOT), phrase queries, TF-IDF and BM25 ranking, snippet generation

6)  Sample Documents are added

 7 .txt files were placed in data/sample_docs/. Example filenames used in this project:

doc1.txt — Artificial Intelligence overview

doc2.txt — Machine Learning & Neural Networks

doc3.txt — Climate Change & Global Warming

doc4.txt — Medical Innovations & Health

doc5.txt — Social Media and Technology

doc6.txt — Renewable Energy

doc7.txt — History of Belgium

Each file contains plain text (UTF-8). The demo script will load every .txt file in that directory automatically.


