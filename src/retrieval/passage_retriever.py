"""Retriever backed by the passage index, for the downstream grounding experiments.

WHY. Every grounding result so far used the flat index, which encoded ~15% of each
case (first 128 tokens of a 554-word median narrative). Retrieval quality caps what
grounding can achieve -- if the retriever hands the generator the wrong patient's
case, no prompting strategy rescues it -- so the grounding numbers were measured
against a crippled retriever.

Measured on 3,015 queries, Hinglish R@1:

    flat index, 128 tokens (what H1 used)    0.1144
    flat index, 256 tokens                   0.1310
    passage index, full coverage             0.1280   (and BM25-full 0.0935)

This exposes the passage index behind the same interface the generation runners
already use, so `h1_real_retrieval.py` can swap retrievers with a flag and the
oracle-vs-real-vs-improved contrast is measurable on identical queries and prompts.

Evidence text returned is the FULL case narrative, not the matching passage: the
passage is how the case was FOUND, but the generator should reason over the whole
case, exactly as the flat-index path did. Keeping that constant means a change in
grounding quality is attributable to which case was retrieved, not to how much of
it the generator got to read.
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.encoding.text_encoder import TextEncoder
from src.retrieval.passage_index import pool_passages_to_cases

logger = logging.getLogger(__name__)

CACHE = Path("data/passage_index")
META = Path("data/faiss_index/evidence_metadata.csv")

#: Passages fetched before max-pooling to distinct cases. Must exceed top_k by a
#: wide margin because ~4.17 passages map to each case.
PASSAGE_POOL = 200


class PassageRetriever:
    """Case-level retrieval over a passage index, max-pooled per case."""

    def __init__(self, max_seq_length: int = 256):
        if not (CACHE / "passage_emb.npy").exists():
            raise SystemExit(
                "No passage index. Build it first:  python -m src.analysis.retrieval_v2")
        self.pf = pd.read_parquet(CACHE / "passages.parquet")
        emb = np.load(CACHE / "passage_emb.npy").astype(np.float32)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.meta = pd.read_csv(META)
        self.case_row = self.pf.case_row.to_numpy()
        self.encoder = TextEncoder(device="cpu")
        self.encoder.load_model()
        self.encoder.model.max_seq_length = max_seq_length
        logger.info("PassageRetriever: %d passages over %d cases (msl=%d)",
                    len(self.pf), len(self.meta), max_seq_length)

    def encode(self, queries: list[str]) -> np.ndarray:
        return self.encoder.encode(queries, batch_size=32, show_progress=False)

    def retrieve_rows(self, queries: list[str], top_k: int = 1) -> np.ndarray:
        """Return `(n_queries, top_k)` metadata ROW indices, best case first."""
        q = self.encode(queries)
        scores, pidx = self.index.search(q.astype(np.float32), PASSAGE_POOL)
        return pool_passages_to_cases(scores, pidx, self.case_row, top_k)

    def evidence_for(self, rows: np.ndarray) -> list[list[dict]]:
        """Materialise retrieved rows into evidence dicts with the FULL case text."""
        out = []
        for r in rows:
            items = []
            for row in r:
                if row < 0:
                    continue
                m = self.meta.iloc[int(row)]
                items.append({"case_id": str(m.case_id),
                              "case_text": str(m.case_text),
                              "condition_group": str(m.condition_group)})
            out.append(items)
        return out


__all__ = ["PassageRetriever", "PASSAGE_POOL"]
