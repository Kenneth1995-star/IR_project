# src/query.py
"""
QueryProcessor: efficient inverted-index based retrieval.

Implements:
 - Boolean queries: AND, OR, NOT, parentheses (shunting-yard + merge ops)
 - Phrase queries: "exact phrase" (requires positional index in Indexer)
 - Ranked retrieval: TF-IDF cosine and BM25
 - Uses Indexer.get_postings_list(token) and Indexer.get_positions(token)
 - Uses Tokenizer.tokenize(query) for identical normalization between index & query
"""

# here we are importing helpful modules and type hints
from typing import List, Tuple, Optional
from collections import defaultdict
import math
import re

# here we are importing the Tokenizer (for splitting queries into words)
# and the Indexer (which stores our inverted index)
from .tokenizer import Tokenizer
from .indexer import Indexer


class QueryProcessor:
    # here we are initializing the query processor
    def __init__(self, indexer: Indexer, tokenizer: Optional[Tokenizer] = None):
        # here we are storing the indexer object that contains postings and stats
        self.indexer = indexer
        # here we are using the given tokenizer, or falling back to the indexer's tokenizer, or creating a new one
        self.tokenizer = tokenizer or indexer.tokenizer or Tokenizer()

    # --- BOOLEAN AND PHRASE PARSING HELPERS ---

    # here we are splitting a Boolean query into meaningful tokens (like AND, OR, NOT, words, and phrases)
    def _tokenize_boolean(self, query: str) -> List[str]:
        # here we are defining a regex to extract phrases, parentheses, and Boolean operators correctly
        token_pattern = r'(\"[^\"]+\"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^()\s]+)'
        parts = re.findall(token_pattern, query, flags=re.I)
        # here we are cleaning whitespace and returning only non-empty parts
        return [p.strip() for p in parts if p.strip()]

    # here we are checking if the query contains Boolean logic or phrase syntax
    def _is_boolean_or_phrase(self, query: str) -> bool:
        # here we are looking for quotes or AND/OR/NOT keywords or parentheses
        return ('"' in query) or bool(re.search(r"\b(AND|OR|NOT)\b", query, flags=re.I)) 

    # --- BASIC BOOLEAN OPERATORS (LIST MERGING) ---

    # here we are implementing the AND operation efficiently (intersection of two sorted lists)
    def _intersect_lists(self, a: List[str], b: List[str]) -> List[str]:
        i, j = 0, 0
        out = []
        # here we are walking through both sorted lists at once
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                # here we are adding matching document IDs to the output
                out.append(a[i]); i += 1; j += 1
            elif a[i] < b[j]:
                i += 1
            else:
                j += 1
        return out

    # here we are implementing OR (union) using a two-pointer merge algorithm
    def _union_lists(self, a: List[str], b: List[str]) -> List[str]:
        i, j = 0, 0
        out = []
        prev = None  # here we are tracking the last added doc ID to avoid duplicates
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                if a[i] != prev:
                    out.append(a[i]); prev = a[i]
                i += 1; j += 1
            elif a[i] < b[j]:
                if a[i] != prev:
                    out.append(a[i]); prev = a[i]
                i += 1
            else:
                if b[j] != prev:
                    out.append(b[j]); prev = b[j]
                j += 1
        # here we are adding any leftover items from list a
        while i < len(a):
            if a[i] != prev:
                out.append(a[i]); prev = a[i]
            i += 1
        # here we are adding any leftover items from list b
        while j < len(b):
            if b[j] != prev:
                out.append(b[j]); prev = b[j]
            j += 1
        return out

    # --- PHRASE SEARCH SUPPORT ---

    # here we are retrieving all documents that contain an exact sequence of tokens (a phrase)
    def _docs_with_phrase(self, phrase_tokens: List[str]) -> List[str]:
        if not phrase_tokens:
            return []

        # here we are retrieving the position dictionary for each token
        # (maps token -> {doc_id: [positions]})
        pos_dicts = []
        for t in phrase_tokens:
            pd = self.indexer.get_positions(t)
            if pd is None:
                return []  # here we are aborting if no positional info is stored
            pos_dicts.append(pd)

        # here we are finding candidate documents that contain all tokens in the phrase
        cand = set(pos_dicts[0].keys())
        for pd in pos_dicts[1:]:
            cand &= set(pd.keys())

        if not cand:
            return []

        out = []
        # here we are checking for consecutive positions in each candidate document
        for doc in sorted(cand):
            pos0 = pos_dicts[0][doc]
            subsequent_sets = [set(pd[doc]) for pd in pos_dicts[1:]]
            for p in pos0:
                ok = True
                for i, sset in enumerate(subsequent_sets, start=1):
                    # here we are verifying that the next token appears exactly after the previous one
                    if (p + i) not in sset:
                        ok = False; break
                if ok:
                    out.append(doc); break
        return out

    # --- BOOLEAN QUERY EVALUATION (SHUNTING-YARD + RPN) ---

    # here we are evaluating a Boolean query using the shunting-yard algorithm
    def _eval_boolean(self, query: str) -> List[str]:
        # here we are tokenizing the Boolean query
        tokens = self._tokenize_boolean(query)
        if not tokens:
            return []

        # here we are defining operator precedence
        prec = {'NOT': 3, 'AND': 2, 'OR': 1}
        output = []  # here we are storing tokens in Reverse Polish Notation
        ops = []     # here we are storing the operator stack

        def is_op(t): return t.upper() in prec

        # here we are converting infix Boolean expressions into RPN
        for tok in tokens:
            if tok == '(':
                ops.append(tok)
            elif tok == ')':
                while ops and ops[-1] != '(':
                    output.append(ops.pop())
                if ops and ops[-1] == '(':
                    ops.pop()
            elif is_op(tok):
                U = tok.upper()
                while ops and ops[-1] != '(' and (
                    (prec.get(ops[-1], 0) > prec[U]) or
                    (prec.get(ops[-1], 0) == prec[U] and U != 'NOT')
                ):
                    output.append(ops.pop())
                ops.append(U)
            else:
                output.append(tok)
        while ops:
            output.append(ops.pop())

        # here we are evaluating the RPN expression
        stack: List[List[str]] = []
        universe = sorted(list(self.indexer.doc_texts.keys()))

        for sym in output:
            if sym == 'NOT':
                a = stack.pop() if stack else []
                aset = set(a)
                res = [d for d in universe if d not in aset]
                stack.append(res)
            elif sym == 'AND':
                b = stack.pop() if stack else []
                a = stack.pop() if stack else []
                stack.append(self._intersect_lists(a, b))
            elif sym == 'OR':
                b = stack.pop() if stack else []
                a = stack.pop() if stack else []
                stack.append(self._union_lists(a, b))
            else:
                # here we are handling regular words or phrases
                if sym.startswith('"') and sym.endswith('"'):
                    phrase = sym[1:-1]
                    pts = self.tokenizer.tokenize(phrase)
                    docs = self._docs_with_phrase(pts)
                    stack.append(sorted(docs))
                else:
                    toks = self.tokenizer.tokenize(sym)
                    if not toks:
                        stack.append([])
                    else:
                        t = toks[0]
                        pl = self.indexer.get_postings_list(t)
                        docs = [d for (d, _) in pl]
                        stack.append(docs)
        return stack.pop() if stack else []

    # --- RANKED RETRIEVAL METHODS ---

    # here we are computing TF-IDF cosine similarity ranking
    def _rank_tfidf(self, q_tokens: List[str], candidate_docs: Optional[List[str]] = None, top_k: int = 10) -> List[Tuple[str, float]]:
        # here we are counting term frequency for query terms
        q_tf = defaultdict(int)
        for t in q_tokens:
            q_tf[t] += 1

        # here we are computing query weights (log-tf * idf)
        idf = dict()
        q_weights = {}
        for t, f in q_tf.items():
            # df of term
            df_t = self.indexer.get_meta(t)
            if df_t is None:
                continue
            df_t = df_t[2]
            idf_t = math.log10(self.indexer.N() / df_t)
            if idf_t <= 0:
                continue
            idf[t] = idf_t
            q_weights[t] = (1.0 + math.log10(f)) * idf_t

        q_norm = math.sqrt(sum(w*w for w in q_weights.values()))
        if q_norm == 0.0:
            return []

        scores = defaultdict(float)
        # here we are computing dot product between query and document vectors
        for t, q_w in q_weights.items():
            postings = {key: value for key, value in self.indexer.get_postings(t)}
            if candidate_docs:
                for d in candidate_docs:
                    dw = len(postings[d]) # tf
                    dw = 1 + math.log10(dw) # Guaranteed > 0
                    dw *= idf[t]
                    if dw:
                        scores[d] += q_w * dw
            else:
                for d, _ in postings.items():
                    dw = len(postings[d]) # tf
                    dw = 1 + math.log10(dw) # Guaranteed > 0
                    dw *= idf[t]
                    scores[d] += q_w * dw

        # here we are normalizing by vector length to compute cosine similarity
        final = []
        for d, dot in scores.items():
            inverse_doc_norm = self.indexer.get_ltc()[d]
            if inverse_doc_norm > 0:
                final.append((d, dot * inverse_doc_norm / q_norm))
        final.sort(key=lambda x: x[1], reverse=True)
        return final[:top_k]

    # here we are implementing BM25 ranking (a more advanced probabilistic scoring method)
<<<<<<< HEAD
    def _rank_bm25(self, q_tokens: List[str], candidate_docs: Optional[List[str]] = None, top_k: int = 10) -> List[Tuple[str, float]]:
        k1 = 1.5
        b = 0.75
        N = self.indexer.N
        avgdl = self.indexer.avg_doc_len if self.indexer.avg_doc_len > 0 else 1.0
=======
    def _rank_bm25(self, q_tokens: List[str], candidate_docs: Optional[List[str]] = None, top_k: int = 10,
                   k1: float = 1.5, k3: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
        N = self.indexer.N()
        # avgdl = self.indexer.avg_doc_len if self.indexer.avg_doc_len > 0 else 1.0
        avgdl = self.indexer.doc_mean()
>>>>>>> ba6744e4
        q_tf = defaultdict(int)
        for t in q_tokens:
            q_tf[t] += 1

        scores = defaultdict(float)
        candset = set(candidate_docs) if candidate_docs is not None else None

        for t, qf in q_tf.items():
            # df = self.indexer.df.get(t, 0)
            df = self.indexer.get_meta(t)
            if df is None:
                continue
            df = df[2]
            # idf_t = math.log(1 + (N - df + 0.5) / (df + 0.5))
            idf_t = math.log(N / df) # follow slides -> no smoothing
            # postings = self.indexer.get_postings_list(t)
            for d, tf in self.indexer.get_tfs(t):
                # Skip docs that are filtered out
                if candset is not None and d not in candset:
                    continue
                dl = self.indexer.get_doc_length(d)
                denom = tf + k1 * (1 - b + b * (dl / avgdl))
                score_t = idf_t * ((tf * (k1 + 1)) / denom)
                score_t *= (k3 + 1) * q_tf[t] / (k3 + q_tf[t])
                scores[d] += score_t
        final = [(d, s) for d, s in scores.items()]
        final.sort(key=lambda x: x[1], reverse=True)
        return final[:top_k]

    # --- PUBLIC SEARCH FUNCTION ---

    # here we are defining the main search() function that users will call
    def search(self, query: str, top_k: int = 10, method: str = "tfidf"):
        method = method.lower()
        if method not in ("tfidf", "bm25"):
            raise ValueError("method must be 'tfidf' or 'bm25'")

        # here we are checking if the query is Boolean/phrase or normal free-text
        if self._is_boolean_or_phrase(query):
            candidates = self._eval_boolean(query)
            if not candidates:
                return []
            # here we are flattening all Boolean tokens into a list of words for ranking
            parts = re.findall(r'\"[^\"]+\"|[^()\s]+', query)
            q_tokens = []
            for p in parts:
                if p.startswith('"') and p.endswith('"'):
                    q_tokens.extend(self.tokenizer.tokenize(p[1:-1]))
                else:
                    up = p.upper()
                    if up in ("AND", "OR", "NOT"):
                        continue
                    q_tokens.extend(self.tokenizer.tokenize(p))
            if method == "tfidf":
                ranked = self._rank_tfidf(q_tokens, candidates, top_k)
            else:
                ranked = self._rank_bm25(q_tokens, candidates, top_k)
            return [(d, s, self._snippet(d, q_tokens)) for d, s in ranked]
        else:
            # here we are tokenizing and ranking a normal free-text query
            q_tokens = self.tokenizer.tokenize(query)
            if not q_tokens:
                return []
            if method == "tfidf":
                ranked = self._rank_tfidf(q_tokens, None, top_k)
            else:
                ranked = self._rank_bm25(q_tokens, None, top_k)
            return [(d, s, self._snippet(d, q_tokens)) for d, s in ranked]

    # --- SNIPPET GENERATION (for display) ---

    # here we are extracting a short text snippet showing where query terms occur
    def _snippet(self, doc_id: str, q_tokens: List[str], window_chars: int = 60) -> str:
        """
        Doesn't work with stemming, needs to stem whole doc, very expensive
        """
        return
        text = self.indexer.doc_texts.get(doc_id, "")
        lt = text.lower()
        first = None
        for t in q_tokens:
            pos = lt.find(t)
            if pos >= 0 and (first is None or pos < first):
                first = pos
        if first is None:
            return text[:200] + ("..." if len(text) > 200 else "")
        s = max(0, first - window_chars)
        e = min(len(text), first + window_chars)
        snippet = text[s:e].strip()
        if s > 0:
            snippet = "..." + snippet
        if e < len(text):
            snippet = snippet + "..."
        return snippet


<<<<<<< HEAD
=======
if __name__ == "__main__":
    indexer = Indexer(path="index2/")
    indexer.load()
    query = QueryProcessor(indexer)

    l = [
        # "3million",
        # "pokémon",
        # "\"security footage\"",
        # "\"shadowless charizard\"",
        # "charizard AND shadowless",
        # "information retrieval",
        "\"machine learning\""
    ]

    for x in l:
        print(query.search(x, method="bm25"))
>>>>>>> ba6744e4
