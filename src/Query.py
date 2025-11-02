# src/Query.py
"""
QueryProcessor: efficient inverted-index based retrieval.

Implements:
 - Boolean queries: AND, OR, NOT, parentheses (shunting-yard + merge ops)
 - Phrase queries: "exact phrase" (requires positional index in Indexer)
 - Ranked retrieval: TF-IDF cosine and BM25
"""
from typing import List, Tuple, Optional, Any, Dict, Literal
from collections import defaultdict
import math
import re

from tokenizer import Tokenizer
# from indexer import Indexer
from Indexer import Indexer


class QueryProcessor:
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.tokenizer = indexer.tokenizer

    # --- BOOLEAN AND PHRASE PARSING HELPERS ---

    # Split query into tokens, operators and phrases
    def _split_query(self, query: str) -> List[str]:
        token_pattern = r'(\"[^\"]+\"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^()\s]+)'
        parts = re.findall(token_pattern, query, flags=re.I)
        # here we are cleaning whitespace and returning only non-empty parts
        return [p.strip() for p in parts if p.strip()]

    # See if it contains any boolean operators
    def _is_boolean(self, parts: List[str]) -> bool:
        return any(op in parts for op in ["AND", "OR", "NOT"])

    # --- BASIC BOOLEAN OPERATORS (LIST MERGING) ---

    # AND operator
    def _intersect_lists(self, a: List[str], b: List[str]) -> List[str]:
        """
        Walk through both lists and match document IDs
        """
        i, j = 0, 0
        out = []
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                out.append(a[i]); i += 1; j += 1
            elif a[i] < b[j]:
                i += 1
            else:
                j += 1
        return out

    # OR operator
    def _union_lists(self, a: List[str], b: List[str]) -> List[str]:
        """
        2 pointer implementation for union
        """
        i, j = 0, 0
        out = []
        prev = None  # avoid duplicates
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

    def _phrase_search(self, phrase_tokens: List[str]) -> Tuple[List[int], Tuple[List[Tuple[int, int]], int]]:
        """
        Returns candidate documents and tf, df
        """
        if not phrase_tokens:
            return [], ([], 0)
        
        maps = []
        for t in phrase_tokens:
            p_list = self.indexer.get_postings(t)
            if p_list is None:
                return [], ([], 0)

            # Convert to dict for Doc lookup
            maps.append({doc: positions for doc, positions in p_list})


        # Docs contained in all terms
        docs = set(maps[0].keys())
        for pd in maps[1:]:
            docs &= set(pd.keys())
            if not docs:
                return [], ([], 0)

        out = []
        tfs = []
        # Consecutive positions check
        for doc in sorted(docs):
            candidates = maps[0][doc]

            for i in range(1, len(maps)):
                next_pos_set = set(maps[i][doc])
                candidates = [p for p in candidates if (p + i) in next_pos_set]
                if not candidates:
                    break
            if candidates:
                out.append(doc)
                tfs.append((doc, len(candidates)))
        df = len(tfs)
        return out, (tfs, df)

    # --- BOOLEAN QUERY EVALUATION (SHUNTING-YARD + RPN) ---

    # here we are evaluating a Boolean query using the shunting-yard algorithm
    def _shunting_yard(self, tokens: List[str]) -> List[str]:
        """
        Boolean query evaluation using shunting yard, converting to RPN
        """
        if not tokens:
            return []

        # Insert implicit AND
        result = []
        for i, tok in enumerate(tokens):
            result.append(tok)
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if any(x in ["AND", "OR"] for x in [tok, nxt]):
                    continue
                if tok not in  ["(", "NOT"] and nxt != ")":
                    result.append('AND')

        tokens = result

        precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
        op_stack = [] # operator stack
        output = [] # RPN
        pol_stack = [] # positivity check

        def is_op(t): return t.upper() in precedence

        def emit(tok):
            # push to RPN
            output.append(tok)

            # update polarity stack
            if tok == 'NOT':
                pos, neg = pol_stack.pop() if pol_stack else (False, False)
                pol_stack.append((neg, pos))  # NOT swaps polarity
            elif tok in ('AND', 'OR'):
                b_pos, b_neg = pol_stack.pop() if pol_stack else (False, False)
                a_pos, a_neg = pol_stack.pop() if pol_stack else (False, False)
                pol_stack.append((a_pos or b_pos, a_neg or b_neg))
            else:
                # term/phrase -> contributes a positive literal
                pol_stack.append((True, False))

        for tok in tokens:
            if is_op(tok):
                while (op_stack and op_stack[-1] != '(' and
                    (precedence[op_stack[-1]] > precedence[tok] or
                        (precedence[op_stack[-1]] == precedence[tok] and tok != "NOT"))):
                    emit(op_stack.pop())
                op_stack.append(tok)

            elif tok == '(':
                op_stack.append(tok)

            elif tok == ')':
                while op_stack and op_stack[-1] != '(':
                    emit(op_stack.pop())
                if op_stack and op_stack[-1] == '(':
                    op_stack.pop()  # discard '('
            else:
                # term or phrase token
                emit(tok)

        while op_stack:
            emit(op_stack.pop())

        has_pos = pol_stack[-1][0] if pol_stack else False
        if not has_pos:
            return # Not positive, bad
        return output

    def _rpn(self, tokens: List[str]):
        """
        Evaluate RPN expression, while filtering out non related queries for ranking
        """

        stack: List[List[int]] = []

        # Used to filter out tokens unrelated to the extracted documents
        stack_2: List[List[str]] = []
        universe = self.indexer.get_doc_ids()

        phrase_info = dict()

        # Merge tokens list, removing tokens that don't appear in the list of documents
        def token_hit(tokens: List[str], docs: List[int]):
            output = []
            for token in tokens:
                if token in phrase_info:
                    tfs = phrase_info[token][0]
                else:
                    tfs = self.indexer.get_tfs(token)
                if any(doc in docs for doc, _ in tfs):
                    output.append(token)
            return output

        for sym in tokens:
            if sym == 'NOT':
                a = stack.pop() if stack else []
                res = [d for d in universe if d not in set(a)]
                stack.append(res)

                stack_2.pop()
                stack_2.append([])
            elif sym == 'AND':
                b = stack.pop() if stack else []
                a = stack.pop() if stack else []
                stack.append(self._intersect_lists(a, b))

                b2 = stack_2.pop() if stack else []
                a2 = stack_2.pop() if stack else []
                stack_2.append(token_hit(a2 + b2, stack[-1]))
            elif sym == 'OR':
                b = stack.pop() if stack else []
                a = stack.pop() if stack else []
                stack.append(self._union_lists(a, b))

                b2 = stack_2.pop() if stack else []
                a2 = stack_2.pop() if stack else []
                stack_2.append(token_hit(a2 + b2, stack[-1]))
            else:
                if sym.startswith('"') and sym.endswith('"'):
                    phrase_tokens = self.tokenizer.tokenize(sym)
                    docs, stats = self._phrase_search(phrase_tokens)
                    phrase_token = " ".join(phrase_tokens)
                    phrase_info[phrase_token] = stats
                    stack.append(sorted(docs))
                    stack_2.append([phrase_token])
                else:
                    toks = self.tokenizer.tokenize(sym)
                    if not toks:
                        stack.append([])
                        stack_2.append([])
                    else:
                        t = toks[0]
                        pl = self.indexer.get_tfs(t) # so we don't read the whole block
                        docs = [d for (d, _) in pl]
                        stack.append(docs)
                        stack_2.append([t])
        if stack and stack_2:
            return stack.pop(), stack_2.pop(), phrase_info
        return [], [], {}

    # --- RANKED RETRIEVAL METHODS ---

    # here we are computing TF-IDF cosine similarity ranking
    def _rank_tfidf(self, q_tokens: List[str], candidate_docs: Optional[List[str]] = None, top_k: int = 10, 
                    phrase_info: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        """
        Rank documents by tf idf. Phrase info contains tf and df of phrase tokens,
        candidate docs for boolean search
        """
        # Query tf
        q_tf = defaultdict(int)
        for t in q_tokens:
            q_tf[t] += 1
            
        # Query weights and idf
        idf = dict()
        q_weights = {}
        for t, f in q_tf.items():
            # df of term
            if t in phrase_info:
                # previously calculated df
                df_t = phrase_info[t][1]
                if df_t <= 0:
                    continue
            else:
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
        # Dot product between query and document vectors
        for t, q_w in q_weights.items():
            if t in phrase_info:
                tfs = phrase_info[t][0]
            else:
                tfs = self.indexer.get_tfs(t)
            if candidate_docs:
                tfs = {key: value for key, value in tfs}
                for d in candidate_docs:
                    # d not guaranteed in tfs with OR operator
                    if d in tfs:
                        dw = tfs[d] # tf
                        dw = 1 + math.log10(dw)
                        dw *= idf[t]
                    else:
                        dw = 0
                    if dw:
                        scores[d] += q_w * dw
            else:
                for d, tf in tfs:
                    dw = 1 + math.log10(tf)
                    dw *= idf[t]
                    if dw:
                        scores[d] += q_w * dw

        # Cosine
        final = []
        for d, dot in scores.items():
            inverse_doc_norm = self.indexer.get_ltc()[d]
            if inverse_doc_norm > 0:
                final.append((d, dot * inverse_doc_norm / q_norm))
        final.sort(key=lambda x: x[1], reverse=True)
        return final[:top_k]

    # here we are implementing BM25 ranking (a more advanced probabilistic scoring method)
    def _rank_bm25(self, q_tokens: List[str], candidate_docs: Optional[List[str]] = None, 
                   top_k: int = 10, phrase_info: Dict[str, Any] = None, k1: float = 1.5, 
                   k3: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        bm25 okapi ranking, candidate docs for boolean search
        """
        N = self.indexer.N()
        # avgdl = self.indexer.avg_doc_len if self.indexer.avg_doc_len > 0 else 1.0
        avgdl = self.indexer.doc_mean()
        q_tf = defaultdict(int)
        for t in q_tokens:
            q_tf[t] += 1

        scores = defaultdict(float)
        candidate_docs = set(candidate_docs) if candidate_docs is not None else None

        for t, qf in q_tf.items():
            if t in phrase_info:
                tfs, df = phrase_info[t]
                if df <= 0 >= len(tfs):
                    continue
            else:
                df = self.indexer.get_meta(t)
                if df is None:
                    continue
                df = df[2]
                tfs = self.indexer.get_tfs(t)
            # idf_t = math.log(1 + (N - df + 0.5) / (df + 0.5))
            idf_t = math.log(N / df) # follow slides -> no smoothing

            for d, tf in tfs:
                # Skip docs that are filtered out
                if candidate_docs is not None and d not in candidate_docs:
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
    def search(self, query: str, top_k: int = 10, method: Literal["tfidf", "bm25"] = "tfidf"):
        """
        Main search function. Accepts boolean / phrase / normal queries. 
        Methods implemented: TF-IDF, BM25. 
        """
        method = method.lower()
        if method not in ("tfidf", "bm25"):
            raise ValueError("method must be 'tfidf' or 'bm25'")

        parts = self._split_query(query)
        q_tokens = []
        phrase_info = dict()
        candidates = None

        if self._is_boolean(parts):
            rpn = self._shunting_yard(parts)
            if rpn is None:
                raise ValueError("Query must include at least one positive clause")
            candidates, q_tokens, phrase_info = self._rpn(rpn)
            if not candidates:
                return []
        else:
            for part in parts:
                if part.startswith('"') and part.endswith('"'):
                    phrase_tokens = self.tokenizer.tokenize(part)
                    _, stats = self._phrase_search(phrase_tokens)
                    phrase_token = " ".join(phrase_tokens)
                    phrase_info[phrase_token] = stats
                    q_tokens.append(phrase_token)
                else:
                    q_tokens.extend(self.tokenizer.tokenize(part))
            if not q_tokens:
                return []
        
        if method == "tfidf":
            ranked = self._rank_tfidf(q_tokens, candidates, top_k, phrase_info)
        else:
            ranked = self._rank_bm25(q_tokens, candidates, top_k, phrase_info)
        return [(d, s) for d, s in ranked]

    # --- SNIPPET GENERATION (for display) ---

    # here we are extracting a short text snippet showing where query terms occur
    def _snippet(self, doc_id: str, q_tokens: List[str], window_chars: int = 60) -> str:
        """
        Doesn't work with stemming, needs to stem whole doc, very expensive
        """
        return
        # text = self.indexer.doc_texts.get(doc_id, "")
        # lt = text.lower()
        # first = None
        # for t in q_tokens:
        #     pos = lt.find(t)
        #     if pos >= 0 and (first is None or pos < first):
        #         first = pos
        # if first is None:
        #     return text[:200] + ("..." if len(text) > 200 else "")
        # s = max(0, first - window_chars)
        # e = min(len(text), first + window_chars)
        # snippet = text[s:e].strip()
        # if s > 0:
        #     snippet = "..." + snippet
        # if e < len(text):
        #     snippet = snippet + "..."
        # return snippet


if __name__ == "__main__":
    indexer = Indexer(path="index/")
    indexer.load()
    query = QueryProcessor(indexer)

    # l = [
    #     "INFORMATION RETRIEVAL",
    #     "INFORMATION AND RETRIEVAL",
    #     "INFORMATION OR RETRIEVAL",
    #     "\"MONKEY BALLS\"",
    #     "\"NATURAL LANGUAGE PROCESSING\" AND \"ARTIFICIAL INTELLIGENCE AI SIMULATION\" OR BELGIUM",
    #     "LATTE AND MACHA OR BELGIUM"
    # ]

    l = [
        "\"HARRY POTTER\"",
        "\"HARRY POTTER\" AND \"CHAMBER OF SECRETS\"",
        "SHADOWLESS AND CHARIZARD",
        "100-meters pokémon"
    ]

    for x in l:
        print(query.search(x, method="bm25"))