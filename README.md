# Document Search and Ranked Retrieval

### Course Project — University of Antwerp

---

## Project Overview

This project implements a document search and retrieval system from scratch and is fully compliant with the techniques taught in class.
It **does not** use external search libraries like Lucene, Solr, or ElasticSearch.

### Features

The following features have been implemented:
- Tokenizer with stemming
- Indexer: Builds index from scratch using SPIMI. Final index is static, stored on disk and memory mapped for lazy loading.
- Query processor that handles normal, boolean, phrase queries and any combination, except the normal and boolean combination. BM25 and all SMART-variations of the Vector Space Model are implemented for ranking.

### Dataset  
This project uses the dataset **“Wikipedia Movies”** available on Kaggle:  
https://www.kaggle.com/datasets/exactful/wikipedia-movies  

The dataset is derived from content sourced from the Wikimedia Foundation / Wikipedia community, and is provided under the [Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/) license.  
Attribution: Wikipedia contributors. 

### NLTK data
This projects also makes use of `punkt` and `corpora` from NLTK, which will be downloaded automatically.

---

##  Instructions

**Install dependencies**
Create virtual environment and run:
```sh
pip install -r requirements.txt
```
A demo script is provided.
```sh
python run_demo.py
```

**Adding documents**
Additional documents can be added by following the existing structure found in the `data` folder.


