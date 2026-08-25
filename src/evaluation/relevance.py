"""Relevance criterion for retrieval evaluation, with the multi-label defect fixed.

THE DEFECT. `evidence_metadata.csv` has 10,000 rows but only 9,048 unique `case_id`s,
and **871 case_ids carry more than one `condition_group`** -- the same case is indexed
several times under different labels. For example `PMC10013424` appears as
`dry_scalp`, `hand_lump` AND `swollen_tonsils`.

The shipped criterion compares a query's condition against the ONE label on whichever
duplicate row FAISS happened to return:

    hits = meta.condition_group[retrieved] == query_condition

So a retriever can return exactly the right case and be scored WRONG, purely because
the row it hit carries a different one of that case's labels. That is indefensible
regardless of how much it moves the number.

HOW MUCH IT MOVES THE NUMBER: not much. Measured, LaBSE @128 on 395 stratified queries:

    recorded-row label (shipped)   R@1 0.1316   R@10 0.6329
    full case label set (fixed)    R@1 0.1342   R@10 0.6380

**+0.0026.** The ceiling is structural -- only 9.6% of cases are multi-label, so the
fix can only touch about one retrieval in ten. It was suggested that repairing this
would lift LaBSE to 0.3476 and reverse the BM25 comparison; that does not reproduce,
and this module exists for correctness, NOT to rescue that result. Do not cite it as
a reason the encoder comparison changed.

REPORTING NOTE. Because this loosens the criterion, report both numbers. A reviewer
is entitled to ask whether the metric was relaxed until the result improved, and the
honest answer is that it was relaxed by 0.0026 and the ordering did not change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_label_sets(meta: pd.DataFrame) -> dict[str, set[str]]:
    """Map each `case_id` to the FULL set of condition groups it is filed under."""
    return meta.groupby("case_id")["condition_group"].apply(set).to_dict()


def relevance_matrix(
    retrieved_rows: np.ndarray, gold: np.ndarray, meta: pd.DataFrame,
    strict: bool = False,
) -> np.ndarray:
    """Boolean hit matrix for a `(n_queries, k)` array of retrieved METADATA ROW indices.

    `strict=True` reproduces the shipped single-label criterion, so both can be
    reported side by side from one run.
    """
    if strict:
        cond = meta["condition_group"].to_numpy()
        return np.where(retrieved_rows >= 0,
                        cond[np.clip(retrieved_rows, 0, len(cond) - 1)] == gold[:, None],
                        False)

    label_sets = build_label_sets(meta)
    case_ids = meta["case_id"].to_numpy()
    out = np.zeros(retrieved_rows.shape, dtype=bool)
    for i in range(retrieved_rows.shape[0]):
        g = gold[i]
        for j, row in enumerate(retrieved_rows[i]):
            if row < 0:
                continue
            out[i, j] = g in label_sets.get(case_ids[row], ())
    return out


def dedup_case_rows(retrieved_rows: np.ndarray, meta: pd.DataFrame, top_k: int) -> np.ndarray:
    """Collapse duplicate case_ids in a ranked list, keeping the best rank.

    Without this a top-10 can contain the same case several times under different
    labels, so the list holds fewer than 10 distinct cases and recall@k is
    understated. Measured: mean distinct cases in a top-10 is 9.105, not 10.
    """
    case_ids = meta["case_id"].to_numpy()
    out = np.full((retrieved_rows.shape[0], top_k), -1, dtype=np.int64)
    for i in range(retrieved_rows.shape[0]):
        seen: set[str] = set()
        slot = 0
        for row in retrieved_rows[i]:
            if row < 0 or slot >= top_k:
                continue
            cid = case_ids[row]
            if cid in seen:
                continue
            seen.add(cid)
            out[i, slot] = row
            slot += 1
    return out


__all__ = ["build_label_sets", "relevance_matrix", "dedup_case_rows"]
