"""Ranked-retrieval metrics: Success@k, MRR@k, and nDCG@k (binary and graded).

Why this module exists
----------------------
Earlier scripts computed nDCG@10 with the ideal DCG fixed at 1.0 and the result
clipped to [0, 1]. That is correct only when each query has exactly ONE relevant
document. Under condition-group relevance every query has hundreds (each of the
18 groups holds ~555 of the 10,000 cases), so the true ideal DCG@10 is
sum_{i=1..10} 1/log2(i+1) ~= 4.54, and the old number was a position-weighted hit
indicator, not nDCG. This module computes the ideal per query from the number of
relevant documents in the whole corpus.

Naming
------
"Any relevant document in the top k" is Success@k (a hit rate), not recall: with
hundreds of relevant documents per query, recall@10 in the IR sense could never
exceed 10/555. Earlier result files call it ``recall@k``; new files use
``success@k``.
"""

from __future__ import annotations

import numpy as np


def discounts(k: int) -> np.ndarray:
    """Positional discount 1/log2(rank + 1) for ranks 1..k."""
    return 1.0 / np.log2(np.arange(2, k + 2))


def success_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    """Per-query 1.0 if any of the top-k results is relevant."""
    return hits[:, :k].any(axis=1).astype(float)


def reciprocal_rank(hits: np.ndarray, k: int | None = None) -> np.ndarray:
    """Per-query 1/rank of the first relevant result within the top k (0 if none)."""
    h = hits if k is None else hits[:, :k]
    return np.where(h.any(axis=1), 1.0 / (h.argmax(axis=1) + 1), 0.0)


def ndcg_binary(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """Per-query binary nDCG@k.

    ``n_relevant[q]`` is the number of relevant documents for query ``q`` in the
    WHOLE corpus, which fixes the ideal ranking: min(n_relevant, k) hits at the top.
    Queries with no relevant document score 0.
    """
    d = discounts(k)
    dcg = (hits[:, :k].astype(float) * d).sum(axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    idcg = cum[np.minimum(np.asarray(n_relevant), k)]
    return np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)


def ndcg_graded(grades: np.ndarray, grade_counts: np.ndarray, k: int) -> np.ndarray:
    """Per-query graded nDCG@k with exponential gain 2^grade - 1.

    grades:        (n_queries, >=k) integer grade of each ranked result (0 = irrelevant).
    grade_counts:  (n_queries, max_grade + 1) number of corpus documents at each grade
                   for that query; column 0 is ignored.
    """
    d = discounts(k)
    gains = (2.0 ** grades[:, :k].astype(float)) - 1.0
    dcg = (gains * d).sum(axis=1)

    max_grade = grade_counts.shape[1] - 1
    idcg = np.zeros(len(grades))
    for q in range(len(grades)):
        ideal: list[float] = []
        for g in range(max_grade, 0, -1):
            take = min(int(grade_counts[q, g]), k - len(ideal))
            ideal.extend([2.0 ** g - 1.0] * take)
            if len(ideal) == k:
                break
        if ideal:
            idcg[q] = float((np.array(ideal) * d[: len(ideal)]).sum())
    return np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)


def rank_metrics(hits: np.ndarray, n_relevant: np.ndarray,
                 k_values: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    """Mean Success@k for each k, MRR@K and binary nDCG@K, with K = max(k_values)."""
    K = max(k_values)
    out = {f"success@{k}": float(success_at_k(hits, k).mean()) for k in k_values}
    out[f"MRR@{K}"] = float(reciprocal_rank(hits, K).mean())
    out[f"nDCG@{K}"] = float(ndcg_binary(hits, n_relevant, K).mean())
    return out
