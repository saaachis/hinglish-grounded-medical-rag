"""Passage-level index: encode 100% of every case instead of the first 15%.

THE PROBLEM THIS SOLVES. `build_index.py` encoded one vector per case, built from
the first 200 words, then truncated again to 128 tokens. Measured against the
corpus (median case = 554 words):

    original index (128 tok)   ~85 words encoded    15% of the median case
    rebuilt at 256 tok        ~170 words encoded    31%
    passage chunking          all 554 words        100%

Worse, `h4_baselines.py` handed BM25 and TF-IDF the full 200 words while LaBSE saw
~85, so "BM25 beats LaBSE" was partly just BM25 reading 2.4x more text.

THE FIX. Split each case into overlapping token windows that fit the encoder, index
every window, and score a case by its BEST window (max-pooling). A long case is then
represented by all of its content rather than by whichever findings happened to fall
in the opening sentences -- and a short query can match a focused passage instead of
being diluted by a whole narrative, which is why untruncating alone helped Hinglish
queries but HURT short English ones.

MAX_CHUNKS caps the tail: the longest case is 12,522 words, and encoding all of it
would cost more than it can possibly add. The cap is reported in the output so the
coverage claim stays honest.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: LaBSE's real limit. The encoder silently truncates beyond this.
CHUNK_TOKENS = 256

#: Overlap so a finding spanning a boundary is not lost from both windows.
CHUNK_OVERLAP = 32

#: Cap per case. At stride 224 this covers ~1,300 words, well past the 554-word
#: median; only the long tail is clipped.
MAX_CHUNKS = 6


def chunk_tokens(
    text: str, tokenizer, chunk_tokens: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP, max_chunks: int = MAX_CHUNKS,
) -> list[str]:
    """Split `text` into overlapping windows that each fit the encoder."""
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    if not ids:
        return []
    stride = max(1, chunk_tokens - overlap)
    out: list[str] = []
    for start in range(0, len(ids), stride):
        window = ids[start:start + chunk_tokens]
        if not window:
            break
        out.append(tokenizer.decode(window, skip_special_tokens=True))
        if len(out) >= max_chunks or start + chunk_tokens >= len(ids):
            break
    return out


def build_passage_frame(meta: pd.DataFrame, tokenizer, **kw) -> pd.DataFrame:
    """Expand a case-level metadata frame into a passage-level one."""
    rows = []
    for i, r in enumerate(meta.itertuples(index=False)):
        for j, passage in enumerate(chunk_tokens(r.case_text, tokenizer, **kw)):
            rows.append({"case_row": i, "case_id": r.case_id,
                         "condition_group": r.condition_group,
                         "chunk_ix": j, "passage": passage})
        if (i + 1) % 2000 == 0:
            logger.info("chunked %d/%d cases -> %d passages", i + 1, len(meta), len(rows))
    df = pd.DataFrame(rows)
    logger.info("%d cases -> %d passages (%.2f per case)",
                len(meta), len(df), len(df) / max(1, len(meta)))
    return df


def pool_passages_to_cases(
    scores: np.ndarray, passage_idx: np.ndarray, case_row: np.ndarray, top_k: int,
) -> np.ndarray:
    """Max-pool passage hits into a ranked list of unique case rows.

    FAISS returns passages; the evaluation is over cases. Taking each case's best
    passage (rather than summing) keeps a case with one strongly-matching finding
    from being outranked by one that is merely long.
    """
    out = np.full((scores.shape[0], top_k), -1, dtype=np.int64)
    for q in range(scores.shape[0]):
        seen: dict[int, float] = {}
        for s, p in zip(scores[q], passage_idx[q]):
            if p < 0:
                continue
            c = int(case_row[p])
            if c not in seen or s > seen[c]:
                seen[c] = float(s)
        ranked = sorted(seen.items(), key=lambda kv: -kv[1])[:top_k]
        for slot, (c, _) in enumerate(ranked):
            out[q, slot] = c
    return out


__all__ = ["CHUNK_TOKENS", "CHUNK_OVERLAP", "MAX_CHUNKS",
           "chunk_tokens", "build_passage_frame", "pool_passages_to_cases"]
