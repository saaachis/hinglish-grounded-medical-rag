"""Retrieval baselines for H04 -- Table 1 of the paper.

Runs the same Q1/Q2/Q3 query conditions and the same relevance criterion as
`h4_retrieval.py`, but swaps the retrieval system:

    BM25                  lexical; expected to fail on Hinglish -- that failure,
                          quantified, is the motivation for cross-lingual embedding
    TF-IDF                the project's original approach (the 11-pair era)
    multilingual-e5-base  strong modern multilingual dense retriever
    MuRIL                 Indian-language-specialised encoder
    LaBSE                 the deployed system (read from h4_metrics.csv)

CPU only, no API calls. Writes results/h4_retrieval/h4_baselines.csv
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.h4_retrieval import (
    META_PATH, PAIRS_PATH, TOP_K, OUTPUT_DIR,
    assert_no_leakage, rank_metrics, strip_caption,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_DOC_WORDS = 200
ENCODERS = {
    "multilingual-e5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "MuRIL": ("google/muril-base-cased", "", ""),
}


def main() -> None:
    pairs = pd.read_csv(PAIRS_PATH)
    meta = pd.read_csv(META_PATH)
    logger.info("Loaded %d pairs, %d indexed cases", len(pairs), len(meta))

    q3 = pairs["english_summary"].astype(str)
    q2 = q3.apply(strip_caption)
    assert_no_leakage(q2, pairs["condition_query"])
    variants = {
        "Q1_hinglish": pairs["hinglish_query"].astype(str),
        "Q2_english_question": q2,
        "Q3_english_plus_caption": q3,
    }

    gold = pairs["condition_query"].to_numpy()
    meta_cond = meta["condition_group"].to_numpy()
    docs = [" ".join(str(t).split()[:MAX_DOC_WORDS]) for t in meta["case_text"]]
    rows: list[dict] = []

    # ---------- lexical ----------
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    tok = lambda s: re.findall(r"[a-z0-9]+", str(s).lower())
    logger.info("Building BM25 over %d docs ...", len(docs))
    bm25 = BM25Okapi([tok(d) for d in docs])
    for name, texts in variants.items():
        top = np.vstack([np.argsort(bm25.get_scores(tok(t)))[::-1][:TOP_K] for t in texts])
        h = meta_cond[top] == gold[:, None]
        rows.append({"system": "BM25", "variant": name, **rank_metrics(h)})
        logger.info("BM25 %s R@1=%.4f", name, h[:, 0].mean())

    logger.info("Fitting TF-IDF ...")
    vec = TfidfVectorizer(min_df=2, max_features=200_000, ngram_range=(1, 2))
    D = vec.fit_transform(docs)
    for name, texts in variants.items():
        S = linear_kernel(vec.transform(texts.tolist()), D)
        top = np.argsort(-S, axis=1)[:, :TOP_K]
        h = meta_cond[top] == gold[:, None]
        rows.append({"system": "TF-IDF", "variant": name, **rank_metrics(h)})
        logger.info("TF-IDF %s R@1=%.4f", name, h[:, 0].mean())

    # ---------- dense ----------
    import faiss
    from sentence_transformers import SentenceTransformer

    def enc(model, texts, prefix):
        return np.asarray(model.encode([prefix + t for t in texts], batch_size=32,
                                       show_progress_bar=False, normalize_embeddings=True),
                          dtype=np.float32)

    for label, (model_id, qpre, dpre) in ENCODERS.items():
        logger.info("=== %s ===", label)
        try:
            model = SentenceTransformer(model_id, device="cpu")
            model.max_seq_length = 128
            demb = enc(model, docs, dpre)
            ix = faiss.IndexFlatIP(demb.shape[1]); ix.add(demb)
            for name, texts in variants.items():
                _, idx = ix.search(enc(model, texts.tolist(), qpre), TOP_K)
                h = meta_cond[idx] == gold[:, None]
                rows.append({"system": label, "variant": name, **rank_metrics(h)})
                logger.info("%s %s R@1=%.4f", label, name, h[:, 0].mean())
            del model, ix, demb
        except Exception as e:
            logger.warning("!! %s failed: %s: %s", label, type(e).__name__, e)

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "h4_baselines.csv", index=False)
    logger.info("Wrote %s (%d rows)", OUTPUT_DIR / "h4_baselines.csv", len(out))
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
