"""Rebuild the FAISS index without the truncation defect, and re-measure H04.

THE DEFECT. `build_index.py:105` pins `max_seq_length = 128` while LaBSE supports
256. Measured against the actual corpus:

    indexed case narratives   median 307 tokens -> 100.0% truncated at 128
    Hinglish queries          median 132 tokens ->  52.5% truncated at 128

So every indexed document lost roughly 60% of its content before it was ever
encoded, and half the queries lost their tail. Meanwhile BM25 -- which beat LaBSE
0.1343 to 0.1144 on Q1 -- reads the FULL 200-word document. The comparison was
never fair: the lexical baseline had ~2.4x more information than the dense one.

This rebuilds at the encoder's real limit and re-runs the same evaluation, so the
BM25-beats-LaBSE result can be attributed to the method rather than to a
configuration mistake.

Writes data/faiss_index_256/ and results/index_truncation/.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.encoding.text_encoder import TextEncoder
from src.retrieval.indexer import FAISSIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SRC_META = Path("data/faiss_index/evidence_metadata.csv")
PAIRS = Path("data/processed/mmcqsd_multicare_paired.csv")
OUT_DIR = Path("data/faiss_index_256")
RESULTS = Path("results/index_truncation")
K_VALUES = (1, 3, 5, 10)


def rank_metrics(hits: np.ndarray) -> dict:
    out = {f"recall@{k}": float(hits[:, :k].any(axis=1).mean()) for k in K_VALUES}
    rr = np.where(hits.any(axis=1), 1.0 / (hits.argmax(axis=1) + 1), 0.0)
    out["MRR@10"] = float(rr.mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seq-length", type=int, default=256)
    args = ap.parse_args()

    meta = pd.read_csv(SRC_META)
    pairs = pd.read_csv(PAIRS)
    logger.info("Re-encoding %d cases at max_seq_length=%d", len(meta), args.max_seq_length)

    enc = TextEncoder(device="cpu")
    enc.load_model()
    enc.model.max_seq_length = args.max_seq_length

    docs = meta["case_text"].astype(str).tolist()
    demb = enc.encode(docs, batch_size=16, show_progress=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "evidence_embeddings.npy", demb)
    ix = FAISSIndexer(embedding_dim=demb.shape[1])
    ix.build_index(demb)
    ix.save_index(OUT_DIR / "evidence.index")
    meta.to_csv(OUT_DIR / "evidence_metadata.csv", index=False, encoding="utf-8")
    logger.info("Index built: %d vectors", demb.shape[0])

    # Same three query conditions as h4_retrieval, same relevance criterion.
    from src.analysis.h4_retrieval import assert_no_leakage, strip_caption
    q3 = pairs["english_summary"].astype(str)
    q2 = q3.apply(strip_caption)
    assert_no_leakage(q2, pairs["condition_query"])
    variants = {"Q1_hinglish": pairs["hinglish_query"].astype(str), "Q2_english_question": q2}

    gold = pairs["condition_query"].to_numpy()
    cond = meta["condition_group"].to_numpy()

    rows = []
    for name, texts in variants.items():
        qemb = enc.encode(texts.tolist(), batch_size=16, show_progress=False)
        _, idx = ix.index.search(qemb.astype(np.float32), 10)
        hits = cond[idx] == gold[:, None]
        rows.append({"variant": name, "max_seq_length": args.max_seq_length, **rank_metrics(hits)})
        logger.info("%s R@1=%.4f MRR=%.4f", name, rows[-1]["recall@1"], rows[-1]["MRR@10"])

    new = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    new.to_csv(RESULTS / f"h4_seqlen{args.max_seq_length}.csv", index=False)

    old = pd.read_csv("results/h4_retrieval/h4_metrics.csv")
    old = old[old.variant.isin(variants)][["variant", "recall@1", "recall@10", "MRR@10"]]
    old["max_seq_length"] = 128

    print("\n=== TRUNCATION FIX: 128 -> %d tokens ===" % args.max_seq_length)
    cmp = old.merge(new[["variant", "recall@1", "recall@10", "MRR@10"]],
                    on="variant", suffixes=("_128", f"_{args.max_seq_length}"))
    for c in ("recall@1", "recall@10", "MRR@10"):
        cmp[f"{c}_delta"] = cmp[f"{c}_{args.max_seq_length}"] - cmp[f"{c}_128"]
    print(cmp.to_string(index=False))
    cmp.to_csv(RESULTS / "comparison.csv", index=False)
    print("\nBM25 reference on Q1: R@1 = 0.1343 (reads the FULL document)")


if __name__ == "__main__":
    main()
