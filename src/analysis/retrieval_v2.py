"""Retrieval v2: passage chunking, matched-content baselines, and hybrid fusion.

Replaces the Table 1 comparison, which was not a fair fight. Three changes:

1. PASSAGE CHUNKING. Every case is split into overlapping 256-token windows, all
   windows are indexed, and a case scores as its best window. The old index encoded
   ~85 words of a 554-word median case (15%); this encodes all of it.

2. MATCHED CONTENT. `h4_baselines.py` gave BM25/TF-IDF the full 200 words while
   LaBSE saw ~85 tokens, so the lexical baselines read ~2.4x more text than the
   dense one. Here every system sees the SAME full case text -- lexical natively,
   dense via its passages -- so a difference between them is a method difference.

3. HYBRID FUSION. Reciprocal-rank fusion of the lexical and dense rankings.
   RRF(case) = sum_i 1 / (k + rank_i), k=60. Standard practice, and near-always
   beats either component, because lexical and dense fail on different queries.

Evaluated on the same three query conditions and the same relevance criterion as
H04, so the numbers drop straight into the paper's Table 1.

CPU only, no API calls. Writes results/retrieval_v2/.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.analysis.h4_retrieval import bootstrap_delta, mcnemar, strip_caption
from src.encoding.text_encoder import TextEncoder
from src.retrieval.passage_index import build_passage_frame, pool_passages_to_cases

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PAIRS = Path("data/processed/mmcqsd_multicare_paired.csv")
META = Path("data/faiss_index/evidence_metadata.csv")
OUT = Path("results/retrieval_v2")
CACHE = Path("data/passage_index")

TOP_K = 10

#: Depth of the CANDIDATE lists fed to fusion, before truncating to TOP_K.
#: This is not a tuning knob -- it is a correctness requirement. Reciprocal-rank
#: fusion can only ever surface a document that appears in at least one input
#: list, so fusing two depth-10 lists cannot promote anything outside a union of
#: 20 documents and the "hybrid" degenerates into a re-ordering of what the
#: components already agreed on. Fusion has to see deep candidate lists and be
#: truncated to TOP_K afterwards.
FUSION_DEPTH = 100

PASSAGE_POOL = 400  # passages fetched before pooling to FUSION_DEPTH cases
K_VALUES = (1, 3, 5, 10)
RRF_K = 60
SEED = 42


def rank_metrics(hits: np.ndarray) -> dict:
    out = {f"recall@{k}": float(hits[:, :k].any(axis=1).mean()) for k in K_VALUES}
    rr = np.where(hits.any(axis=1), 1.0 / (hits.argmax(axis=1) + 1), 0.0)
    out["MRR@10"] = float(rr.mean())
    gain = hits / np.log2(np.arange(2, hits.shape[1] + 2))
    ideal = 1.0 / np.log2(2)
    out["nDCG@10"] = float((gain.sum(axis=1) / ideal).clip(0, 1).mean())
    return out


def rrf(*rankings: np.ndarray, top_k: int = TOP_K) -> np.ndarray:
    """Reciprocal-rank fusion over several ranked case-row arrays."""
    n = rankings[0].shape[0]
    fused = np.full((n, top_k), -1, dtype=np.int64)
    for q in range(n):
        score: dict[int, float] = {}
        for r in rankings:
            for rank, c in enumerate(r[q]):
                if c < 0:
                    continue
                score[int(c)] = score.get(int(c), 0.0) + 1.0 / (RRF_K + rank + 1)
        for slot, (c, _) in enumerate(sorted(score.items(), key=lambda kv: -kv[1])[:top_k]):
            fused[q, slot] = c
    return fused


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval v2: chunking + matched baselines + hybrid")
    ap.add_argument("--limit-cases", type=int, default=0, help="0 = all 10,000")
    args = ap.parse_args()

    pairs = pd.read_csv(PAIRS)
    meta = pd.read_csv(META)
    if args.limit_cases:
        meta = meta.head(args.limit_cases)
    logger.info("cases=%d queries=%d", len(meta), len(pairs))

    enc = TextEncoder(device="cpu")
    enc.load_model()
    enc.model.max_seq_length = 256
    tok = enc.model.tokenizer

    # ---------- passages ----------
    CACHE.mkdir(parents=True, exist_ok=True)
    pf_path = CACHE / "passages.parquet"
    emb_path = CACHE / "passage_emb.npy"
    if pf_path.exists() and emb_path.exists():
        pf = pd.read_parquet(pf_path)
        pemb = np.load(emb_path)
        logger.info("cache hit: %d passages", len(pf))
    else:
        pf = build_passage_frame(meta, tok)
        pf.to_parquet(pf_path, index=False)
        logger.info("encoding %d passages ...", len(pf))
        pemb = enc.encode(pf.passage.tolist(), batch_size=32, show_progress=False)
        np.save(emb_path, pemb)

    pix = faiss.IndexFlatIP(pemb.shape[1])
    pix.add(pemb.astype(np.float32))
    case_row = pf.case_row.to_numpy()

    # ---------- queries ----------
    q3 = pairs.english_summary.astype(str)
    variants = {
        "Q1_hinglish": pairs.hinglish_query.astype(str).tolist(),
        "Q2_english_question": q3.apply(strip_caption).tolist(),
    }
    gold = pairs.condition_query.to_numpy()
    cond = meta.condition_group.to_numpy()

    # ---------- lexical over FULL case text (matched content) ----------
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    tk = lambda s: re.findall(r"[a-z0-9]+", str(s).lower())
    docs_full = meta.case_text.astype(str).tolist()
    logger.info("building BM25 over FULL case text ...")
    bm25 = BM25Okapi([tk(d) for d in docs_full])
    vec = TfidfVectorizer(min_df=2, max_features=200_000, ngram_range=(1, 2))
    D = vec.fit_transform(docs_full)

    rows, hit_store = [], {}
    for name, texts in variants.items():
        logger.info("=== %s ===", name)

        # Query embeddings are independent of the case corpus, so they survive
        # index changes and are worth caching -- encoding 6,030 queries at 256
        # tokens dominates the runtime of every experiment in this file.
        qcache = CACHE / f"q_{name}_256.npy"
        if qcache.exists():
            qemb = np.load(qcache)
            if qemb.shape[0] != len(texts):
                qemb = None
        else:
            qemb = None
        if qemb is None:
            qemb = enc.encode(texts, batch_size=32, show_progress=False)
            np.save(qcache, qemb)
        s, p = pix.search(qemb.astype(np.float32), PASSAGE_POOL)
        # Deep candidate lists for fusion; each system is TRUNCATED to TOP_K
        # only when it is scored on its own.
        dense_deep = pool_passages_to_cases(s, p, case_row, FUSION_DEPTH)
        lex_deep = np.vstack([np.argsort(bm25.get_scores(tk(t)))[::-1][:FUSION_DEPTH]
                              for t in texts])
        S = linear_kernel(vec.transform(texts), D)
        tfidf_deep = np.argsort(-S, axis=1)[:, :FUSION_DEPTH]

        hybrid = rrf(dense_deep, lex_deep, top_k=TOP_K)

        for sysname, ranking in [("LaBSE-passages", dense_deep[:, :TOP_K]),
                                 ("BM25-full", lex_deep[:, :TOP_K]),
                                 ("TFIDF-full", tfidf_deep[:, :TOP_K]),
                                 ("Hybrid-RRF", hybrid)]:
            h = np.where(ranking >= 0, cond[np.clip(ranking, 0, len(cond) - 1)] == gold[:, None], False)
            hit_store[(sysname, name)] = h
            rows.append({"system": sysname, "variant": name, **rank_metrics(h)})
            logger.info("  %-16s R@1=%.4f MRR=%.4f", sysname, rows[-1]["recall@1"], rows[-1]["MRR@10"])

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "retrieval_v2_metrics.csv", index=False)

    # ---------- H04 re-tested on the best system ----------
    tests = []
    for sysname in ("LaBSE-passages", "BM25-full", "Hybrid-RRF"):
        a = hit_store[(sysname, "Q1_hinglish")][:, 0]
        b = hit_store[(sysname, "Q2_english_question")][:, 0]
        n01, n10, p = mcnemar(a, b)
        d, lo, hi = bootstrap_delta(a.astype(float), b.astype(float))
        tests.append({"system": sysname, "Q1_R@1": a.mean(), "Q2_R@1": b.mean(),
                      "delta": b.mean() - a.mean(), "ci_lo": lo, "ci_hi": hi,
                      "mcnemar_p": p, "n": len(pairs)})
    tdf = pd.DataFrame(tests)
    tdf.to_csv(OUT / "h4_v2_tests.csv", index=False)

    piv = res.pivot(index="system", columns="variant", values="recall@1")
    L = ["# Retrieval v2 -- passage chunking, matched content, hybrid fusion", "",
         f"n = {len(pairs)} queries over {len(meta)} cases ({len(pf)} passages, "
         f"{len(pf)/len(meta):.2f} per case). CPU only.", "",
         "Every system now reads the SAME full case text: lexical natively, dense via its",
         "passages. The previous Table 1 gave BM25 the full 200 words while LaBSE saw ~85",
         "tokens, so that comparison measured the configuration, not the method.", "",
         "## Recall@1", "", "| System | Q1 Hinglish | Q2 English |", "|---|---:|---:|"]
    for s in piv.index:
        L.append(f"| `{s}` | {piv.loc[s,'Q1_hinglish']:.4f} | {piv.loc[s,'Q2_english_question']:.4f} |")
    L += ["| *random floor* | *0.0626* | *0.0626* |", "",
          "### For reference, the OLD (truncated, unmatched) numbers", "",
          "| System | Q1 Hinglish |", "|---|---:|",
          "| LaBSE @128 tok, ~85 words | 0.1144 |",
          "| LaBSE @256 tok, ~170 words | 0.1310 |",
          "| BM25 @200 words | 0.1343 |", "",
          "## H04 re-tested per system", "",
          "| System | Q1 | Q2 | Q2-Q1 | 95% CI | McNemar p |", "|---|---:|---:|---:|---|---:|"]
    for _, r in tdf.iterrows():
        L.append(f"| `{r.system}` | {r['Q1_R@1']:.4f} | {r['Q2_R@1']:.4f} | "
                 f"**{r.delta:+.4f}** | [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {r.mcnemar_p:.4g} |")
    L += ["", "> H04 is re-tested on EVERY system because the code-mixing penalty is a",
          "> difference between two arms, and fixing truncation moved those arms in opposite",
          "> directions. A penalty that holds across retrieval methods is a property of",
          "> code-mixing; one that appears only under a particular configuration is not.", ""]

    (OUT / "retrieval_v2_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
