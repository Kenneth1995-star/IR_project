# src/extras.py
# here, we are adding standard IR evaluation helpers and keeping them small and correct.

import math
from typing import Iterable, List


def precision_at_k(
    relevant: Iterable[str], ranked: List[str], k: int
) -> float:
    # here, we are computing precision@k
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    topk = ranked[:k]
    hits = sum(1 for d in topk if d in relevant_set)
    return hits / float(k)


def recall_at_k(relevant: Iterable[str], ranked: List[str], k: int) -> float:
    # here, we are computing recall@k
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    topk = ranked[:k]
    hits = sum(1 for d in topk if d in relevant_set)
    return hits / float(len(relevant_set))


def dcg_at_k(relevant_set: Iterable[str], ranked: List[str], k: int) -> float:
    # here, we are computing DCG with binary relevance by default
    rset = set(relevant_set)
    dcg = 0.0
    for i in range(min(k, len(ranked))):
        rel = 1.0 if ranked[i] in rset else 0.0
        denom = math.log2(i + 2)
        dcg += (2**rel - 1) / denom
    return dcg


def ndcg_at_k(relevant_set: Iterable[str], ranked: List[str], k: int) -> float:
    # here, we are computing normalized DCG by dividing by IDCG
    rset = set(relevant_set)
    # here, IDCG for binary relevance with R relevant docs is sum_{i=0..R-1} 1/log2(i+2)
    R = min(len(rset), k)
    if R == 0:
        return 0.0
    idcg = sum((1.0 / math.log2(i + 2)) for i in range(R))
    if idcg == 0:
        return 0.0
    return dcg_at_k(rset, ranked, k) / idcg


# here, we add Average Precision and MAP for stronger evaluation evidence
def average_precision(
    relevant: Iterable[str], ranked: List[str], k: int = None
) -> float:
    rset = set(relevant)
    if k is None:
        k = len(ranked)
    hits = 0
    sum_prec = 0.0
    for i in range(min(k, len(ranked))):
        if ranked[i] in rset:
            hits += 1
            sum_prec += hits / float(i + 1)
    if hits == 0:
        return 0.0
    return sum_prec / float(len(rset))


def mean_average_precision(
    qrels: List[Iterable[str]], ranked_lists: List[List[str]], k: int = None
) -> float:
    aps = []
    for rel, ranked in zip(qrels, ranked_lists):
        aps.append(average_precision(rel, ranked, k))
    return sum(aps) / float(len(aps)) if aps else 0.0


def reciprocal_rank(relevant: Iterable[str], ranked: List[str]) -> float:
    rset = set(relevant)
    for i, d in enumerate(ranked):
        if d in rset:
            return 1.0 / float(i + 1)
    return 0.0


def mean_reciprocal_rank(
    qrels: List[Iterable[str]], ranked_lists: List[List[str]]
) -> float:
    rr = [reciprocal_rank(r, rl) for r, rl in zip(qrels, ranked_lists)]
    return sum(rr) / float(len(rr)) if rr else 0.0


def vbyte_encode(nums):
    """
    Encodes list of nums
    """
    out = bytearray()
    for n in nums:
        while True:
            b = n & 0x7F
            n >>= 7
            if n:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def vbyte_decode(buf):
    """
    Decodes encoded list of nums
    """
    n = 0
    shift = 0
    for b in buf:
        n |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            yield n
            n = 0
            shift = 0
        else:
            shift += 7


def delta_encode(sorted_nums):
    """
    Assumes sorted
    """
    prev = 0
    for x in sorted_nums:
        yield x - prev
        prev = x


def delta_decode(deltas):
    s = 0
    for d in deltas:
        s += d
        yield s
