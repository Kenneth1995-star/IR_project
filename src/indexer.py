# src/indexer.py
"""
Indexer module for building an inverted index.

Indexing component that supports:
 - basic inverted index building (token → documents)
 - positional indexing (token → document → positions)
 - computation of statistical weights (TF-IDF, IDF, etc.)
 - document norm and average length computations for ranking
 - disk storage and lazy loading for scalability
"""

import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

from .tokenizer import Tokenizer


class Indexer:
    def __init__(self, tokenizer: Optional[Tokenizer] = None, positional: bool = False, updatable: bool = False):
        # - positional: whether we store word positions (needed for phrase queries)
        # - updatable: whether documents can be dynamically added/removed later
        self.tokenizer = tokenizer or Tokenizer()
        self.positional = positional
        self.updatable = updatable

        # Main internal structure for postings:
        # token → {doc_id: count}  or  token → {doc_id: [pos1, pos2, ...]} if positional=True
        self._postings_dict: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Sorted posting list version of the above dictionary for fast merging in boolean queries:
        # token → [(doc_id, freq), ...]
        self.postings_lists: Dict[str, List[Tuple[str, int]]] = {}

        # Raw text and lengths of each document for ranking and snippet generation
        self.doc_texts: Dict[str, str] = {}
        self.doc_lengths: Dict[str, int] = {}

        # Statistical variables for ranking
        self.N: int = 0
        self.df: Dict[str, int] = {}       # document frequency per token
        self.idf: Dict[str, float] = {}    # inverse document frequency
        self.tfidf_postings: Dict[str, Dict[str, float]] = {}  # precomputed TF-IDF weights
        self.doc_norms: Dict[str, float] = {}                  # document vector norms
        self.avg_doc_len: float = 0.0

        # Optional disk-based indexing
        self._lexicon: Optional[Dict[str, int]] = None
        self._postings_file_path: Optional[str] = None

        # Whether the index has been built and is ready for querying
        self._needs_build = True

    # ---------------------------------------------------------
    # DOCUMENT INDEXING
    # ---------------------------------------------------------
    def add_document(self, doc_id: str, text: str) -> None:
        """
        Adding a document to the indexer and update postings accordingly.
        """
        if doc_id in self.doc_texts:
            if not self.updatable:
                raise ValueError(f"Document {doc_id} already exists and indexer not updatable.")
            self.remove_document(doc_id)

        self.doc_texts[doc_id] = text
        tokens = self.tokenizer.tokenize(text)

        # For BM25 normalization later
        self.doc_lengths[doc_id] = len(tokens)

        if self.positional:
            # here, we are storing positions for every token in each document
            for pos, t in enumerate(tokens):
                posting = self._postings_dict.setdefault(t, {})
                posting.setdefault(doc_id, []).append(pos)
        else:
            # here, we are counting term frequencies instead of storing positions
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            # here, we are saving term frequencies for each token-document pair
            for t, cnt in tf.items():
                self._postings_dict.setdefault(t, {})[doc_id] = cnt

        # here, we are marking the index as needing rebuild before querying
        self._needs_build = True

    def remove_document(self, doc_id: str) -> None:
        """
        Remove a document from the index (only if updatable=True).
        """
        if doc_id not in self.doc_texts:
            return

        if not self.updatable:
            raise ValueError("Indexer not updatable.")

        del self.doc_texts[doc_id]
        del self.doc_lengths[doc_id]

        # here, we are iterating through all tokens to remove the document from each posting
        to_del = []
        for t, posting in self._postings_dict.items():
            if doc_id in posting:
                del posting[doc_id]
            if not posting:
                to_del.append(t)
        # here, we are cleaning up empty tokens entirely
        for t in to_del:
            del self._postings_dict[t]

        # here, we are marking index for rebuild
        self._needs_build = True

    # ---------------------------------------------------------
    # BUILD FINAL INDEX
    # ---------------------------------------------------------
    def build(self, idf_smoothing: bool = True) -> None:
        """
        Finalize index: compute all statistics required for ranking and retrieval.
        """
        # here, we are counting the number of documents
        self.N = len(self.doc_texts)

        # here, we are computing document frequency (how many docs contain each token)
        self.df = {t: len(posting) for t, posting in self._postings_dict.items()}

        # here, we are computing inverse document frequency for each token
        self.idf = {}
        for t, df in self.df.items():
            if idf_smoothing:
                # here, we are applying log(1 + N/df) smoothing to avoid zero division
                self.idf[t] = math.log(1.0 + (self.N / df))
            else:
                self.idf[t] = math.log(self.N / df) if df > 0 else 0.0

        # here, we are building sorted posting lists for each token
        self.postings_lists = {}
        for t, posting in self._postings_dict.items():
            lst = []
            for doc_id, val in posting.items():
                # here, we are converting positional lists into frequencies if necessary
                freq = len(val) if isinstance(val, (list, tuple)) else int(val)
                lst.append((doc_id, freq))
            lst.sort(key=lambda x: x[0])
            self.postings_lists[t] = lst

        # here, we are now computing TF-IDF weights and document norms
        self.tfidf_postings = {}
        doc_norm_sqr = {d: 0.0 for d in self.doc_texts}

        for t, lst in self.postings_lists.items():
            idf_t = self.idf.get(t, 0.0)
            wmap = {}
            for doc_id, freq in lst:
                # here, we are calculating term weight = (1 + log(tf)) * idf
                tf_w = 1.0 + math.log(freq) if freq > 0 else 0.0
                w = tf_w * idf_t
                wmap[doc_id] = w
                # here, we are accumulating squared norms for cosine similarity
                doc_norm_sqr[doc_id] += w * w
            self.tfidf_postings[t] = wmap

        # here, we are finalizing document norms (Euclidean length)
        self.doc_norms = {d: (doc_norm_sqr[d] ** 0.5) for d in self.doc_texts}

        # here, we are computing the average document length (used in BM25)
        self.avg_doc_len = sum(self.doc_lengths.values()) / float(self.N) if self.N > 0 else 0.0

        # here, we are marking index as fully built
        self._needs_build = False

    # ---------------------------------------------------------
    # ACCESS HELPERS
    # ---------------------------------------------------------
    def get_postings_dict(self, token: str) -> Dict[str, Any]:
        """
        here, we are retrieving postings for a token as a dictionary.
        It checks in-memory index first, then tries the disk-based one if needed.
        """
        if token in self._postings_dict:
            return self._postings_dict[token]
        if token in self.postings_lists:
            return {doc: freq for (doc, freq) in self.postings_lists[token]}
        if self._lexicon and self._postings_file_path:
            pl = self._read_posting_from_disk(token)
            if pl is None:
                return {}
            return {doc_id: val for doc_id, val in pl}
        return {}

    def get_postings_list(self, token: str) -> List[Tuple[str, int]]:
        """
        here, we are retrieving postings for a token as a sorted list of (doc_id, frequency).
        """
        if token in self.postings_lists:
            return self.postings_lists[token]
        d = self.get_postings_dict(token)
        if not d:
            return []
        lst = [(doc_id, len(v) if isinstance(v, (list, tuple)) else int(v)) for doc_id, v in d.items()]
        lst.sort(key=lambda x: x[0])
        return lst

    def get_positions(self, token: str) -> Optional[Dict[str, List[int]]]:
        """
        here, we are returning the positional postings if available.
        Otherwise, None is returned if positional indexing was not enabled.
        """
        pst = self._postings_dict.get(token)
        if pst:
            for v in pst.values():
                if isinstance(v, (list, tuple)):
                    return pst
            return None
        if self._lexicon and self._postings_file_path:
            pl = self._read_posting_from_disk(token)
            if not pl:
                return None
            out = {doc_id: val for doc_id, val in pl if isinstance(val, list)}
            return out if out else None
        return None

    # ---------------------------------------------------------
    # DISK PERSISTENCE
    # ---------------------------------------------------------
    def save_postings_to_disk(self, prefix: str) -> Tuple[str, str]:
        """
        here, we are saving the entire index to disk using a lexicon + postings file.
        """
        postings_path = prefix + ".postings"
        lexicon_path = prefix + ".lexicon"

        with open(postings_path, "wb") as pf:
            lexicon = {}
            for t, lst in self.postings_lists.items():
                offset = pf.tell()
                source = self._postings_dict.get(t)
                arr = [[doc_id, val] for doc_id, val in (source.items() if source else lst)]
                line = (t + "\t" + json.dumps(arr, ensure_ascii=False) + "\n").encode("utf8")
                pf.write(line)
                lexicon[t] = offset

        with open(lexicon_path, "w", encoding="utf8") as lf:
            json.dump(lexicon, lf, ensure_ascii=False)

        self._lexicon = lexicon
        self._postings_file_path = postings_path
        return postings_path, lexicon_path

    def load_lexicon(self, prefix: str) -> None:
        """
        here, we are loading the lexicon and setting up the postings file for later lazy loading.
        """
        postings_path = prefix + ".postings"
        lexicon_path = prefix + ".lexicon"
        if not os.path.exists(postings_path) or not os.path.exists(lexicon_path):
            raise FileNotFoundError("Postings or lexicon file not found.")
        with open(lexicon_path, "r", encoding="utf8") as lf:
            self._lexicon = json.load(lf)
        self._postings_file_path = postings_path

    def _read_posting_from_disk(self, token: str) -> Optional[List[Any]]:
        """
        here, we are reading a single posting from disk using the byte offset stored in the lexicon.
        """
        if not self._lexicon or not self._postings_file_path:
            return None
        offset = self._lexicon.get(token)
        if offset is None:
            return None
        with open(self._postings_file_path, "rb") as pf:
            pf.seek(offset)
            line = pf.readline().decode("utf8")
            parts = line.split("\t", 1)
            if len(parts) != 2:
                return None
            try:
                return json.loads(parts[1])
            except Exception:
                return None

    # ---------------------------------------------------------
    # SMALL UTILITIES
    # ---------------------------------------------------------
    def vocab(self) -> List[str]:
        """
        here, we are returning the sorted list of all tokens in the index.
        """
        if self.postings_lists:
            return sorted(self.postings_lists.keys())
        return sorted(self._postings_dict.keys())

    def save(self, path: str) -> None:
        """
        here, we are saving both the postings and lexicon to disk with a given prefix.
        """
        self.save_postings_to_disk(path)
