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
   RRF(case) = sum_i 1 / (k + rank_i), k=60.

Metrics (revised for ICCSDI review, R3.2)
-----------------------------------------
* Success@k (k = 1, 3, 5, 10) -- "any relevant case in the top k". Earlier files
  called this recall@k; with ~555 relevant cases per query it is a hit rate.
* MRR@10.
* nDCG@10, binary, with the ideal DCG computed from the number of relevant cases
  in the whole corpus. The previous nDCG fixed the ideal at 1.0 and clipped, which
  is only valid for a single relevant document (see src/evaluation/retrieval_metrics.py).
* Graded nDCG@10 against an AUTOMATIC graded-relevance proxy:
      grade 2  same condition group AND shares >= 1 lexicon concept with the
               query's gold English question (caption-stripped)
      grade 1  same condition group only
      grade 0  otherwise
  This is finer than group-level relevance but is NOT a clinical judgment.

The code-mixing penalty (English - Hinglish) is tested for EVERY system at EVERY
depth: McNemar for Success@k, and a paired bootstrap CI with a Wilcoxon
signed-rank test for MRR@10 and both nDCGs.

Rankings are cached to data/passage_index/rankings_<variant>.npz so later
baselines (transliteration, translation) reuse the same indexes. CPU only, no API
calls. Writes results/retrieval_v2/.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.h4_retrieval import bootstrap_delta, mcnemar, strip_caption
from src.evaluation.concept_lexicon import POSITIVE_CONCEPTS, extract_concepts
from src.evaluation.retrieval_metrics import (
    ndcg_binary, ndcg_graded, reciprocal_rank, success_at_k,
)
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

VARIANTS = ("Q1_hinglish", "Q2_english_question")
SYSTEMS = ("Hybrid-RRF", "LaBSE-passages", "BM25-full", "TFIDF-full")


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(s).lower())


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


# --------------------------------------------------------------------------- #
# Relevance
# --------------------------------------------------------------------------- #
def concept_matrix(texts: list[str], concepts: list[str]) -> np.ndarray:
    col = {c: j for j, c in enumerate(concepts)}
    M = np.zeros((len(texts), len(concepts)), dtype=bool)
    for i, t in enumerate(texts):
        for c in extract_concepts(t):
            if c in col:
                M[i, col[c]] = True
    return M


def build_relevance(meta: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    """Group-level relevance counts and the graded proxy, for every query x case."""
    concepts = sorted(POSITIVE_CONCEPTS)
    cc_path = CACHE / "case_concepts.npy"
    if cc_path.exists() and np.load(cc_path).shape == (len(meta), len(concepts)):
        C = np.load(cc_path)
    else:
        logger.info("extracting lexicon concepts from %d cases ...", len(meta))
        C = concept_matrix(meta.case_text.astype(str).tolist(), concepts)
        np.save(cc_path, C)

    q_text = pairs.english_summary.astype(str).apply(strip_caption).tolist()
    Q = concept_matrix(q_text, concepts)

    cond = meta.condition_group.to_numpy()
    gold = pairs.condition_query.to_numpy()
    same = gold[:, None] == cond[None, :]                              # (nq, ncase)
    overlap = (Q.astype(np.int32) @ C.T.astype(np.int32)) > 0            # (nq, ncase)
    grade = np.where(same & overlap, 2, np.where(same, 1, 0)).astype(np.int8)
    counts = np.stack([(grade == g).sum(axis=1) for g in (0, 1, 2)], axis=1)
    return {"grade": grade, "n_relevant": same.sum(axis=1), "grade_counts": counts,
            "query_has_concept": Q.any(axis=1)}


def per_query_scores(ranking: np.ndarray, rel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    safe = np.clip(ranking, 0, rel["grade"].shape[1] - 1)
    grades = np.take_along_axis(rel["grade"], safe, axis=1)
    grades = np.where(ranking >= 0, grades, 0)
    hits = grades > 0
    out = {f"success@{k}": success_at_k(hits, k) for k in K_VALUES}
    out["MRR@10"] = reciprocal_rank(hits, TOP_K)
    out["nDCG@10"] = ndcg_binary(hits, rel["n_relevant"], TOP_K)
    out["graded_nDCG@10"] = ndcg_graded(grades, rel["grade_counts"], TOP_K)
    return out


# --------------------------------------------------------------------------- #
# Retrieval (cached)
# --------------------------------------------------------------------------- #
def run_retrieval(meta: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    from src.encoding.text_encoder import TextEncoder

    enc = TextEncoder(device="cpu")
    enc.load_model()
    enc.model.max_seq_length = 256
    tok = enc.model.tokenizer

    CACHE.mkdir(parents=True, exist_ok=True)
    pf_path, emb_path = CACHE / "passages.parquet", CACHE / "passage_emb.npy"
    if pf_path.exists() and emb_path.exists():
        pf, pemb = pd.read_parquet(pf_path), np.load(emb_path)
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

    docs_full = meta.case_text.astype(str).tolist()
    logger.info("building BM25 and TF-IDF over FULL case text ...")
    bm25 = BM25Okapi([tokenize(d) for d in docs_full])
    vec = TfidfVectorizer(min_df=2, max_features=200_000, ngram_range=(1, 2))
    D = vec.fit_transform(docs_full)

    texts_by_variant = {
        "Q1_hinglish": pairs.hinglish_query.astype(str).tolist(),
        "Q2_english_question": pairs.english_summary.astype(str).apply(strip_caption).tolist(),
    }
    out = {}
    for name, texts in texts_by_variant.items():
        cache = CACHE / f"rankings_{name}.npz"
        if cache.exists():
            z = np.load(cache)
            if z["LaBSE-passages"].shape[0] == len(texts):
                out[name] = {s: z[s] for s in SYSTEMS}
                logger.info("rankings cache hit: %s", name)
                continue
        logger.info("=== retrieving %s ===", name)
        qcache = CACHE / f"q_{name}_256.npy"
        qemb = np.load(qcache) if qcache.exists() else None
        if qemb is None or qemb.shape[0] != len(texts):
            qemb = enc.encode(texts, batch_size=32, show_progress=False)
            np.save(qcache, qemb)
        s, p = pix.search(qemb.astype(np.float32), PASSAGE_POOL)
        dense_deep = pool_passages_to_cases(s, p, case_row, FUSION_DEPTH)
        lex_deep = np.vstack([np.argsort(bm25.get_scores(tokenize(t)))[::-1][:FUSION_DEPTH]
                              for t in texts])
        tfidf_deep = np.argsort(-linear_kernel(vec.transform(texts), D), axis=1)[:, :FUSION_DEPTH]
        ranks = {"LaBSE-passages": dense_deep[:, :TOP_K], "BM25-full": lex_deep[:, :TOP_K],
                 "TFIDF-full": tfidf_deep[:, :TOP_K],
                 "Hybrid-RRF": rrf(dense_deep, lex_deep, top_k=TOP_K)}
        np.savez_compressed(cache, **ranks, dense_deep=dense_deep, lex_deep=lex_deep)
        out[name] = ranks
    return out


# --------------------------------------------------------------------------- #
# Penalty tests
# --------------------------------------------------------------------------- #
def penalty_tests(scores: dict, n: int) -> pd.DataFrame:
    rows = []
    for sysname in SYSTEMS:
        hi, en = scores[(sysname, "Q1_hinglish")], scores[(sysname, "Q2_english_question")]
        for metric in [f"success@{k}" for k in K_VALUES] + ["MRR@10", "nDCG@10", "graded_nDCG@10"]:
            a, b = hi[metric], en[metric]
            lo, hi_ci = bootstrap_delta(b, a)            # CI on English - Hinglish
            if metric.startswith("success@"):
                _, _, p = mcnemar(a, b)
                test = "McNemar (exact)"
            else:
                diff = b - a
                p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                test = "Wilcoxon signed-rank"
            rows.append({"system": sysname, "metric": metric, "hinglish": a.mean(),
                         "english": b.mean(), "penalty": b.mean() - a.mean(),
                         "ci_lo": lo, "ci_hi": hi_ci, "p": p, "test": test, "n": n})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval v2: chunking + matched baselines + hybrid")
    ap.add_argument("--limit-cases", type=int, default=0, help="0 = all 10,000")
    args = ap.parse_args()

    pairs = pd.read_csv(PAIRS)
    meta = pd.read_csv(META)
    if args.limit_cases:
        meta = meta.head(args.limit_cases)
    logger.info("cases=%d queries=%d", len(meta), len(pairs))

    rankings = run_retrieval(meta, pairs)
    rel = build_relevance(meta, pairs)

    rows, scores = [], {}
    for name in VARIANTS:
        for sysname in SYSTEMS:
            sc = per_query_scores(rankings[name][sysname], rel)
            scores[(sysname, name)] = sc
            rows.append({"system": sysname, "variant": name, **{m: float(v.mean()) for m, v in sc.items()}})
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "retrieval_v2_metrics.csv", index=False)

    tests = penalty_tests(scores, len(pairs))
    tests.to_csv(OUT / "h4_v2_tests_by_depth.csv", index=False)
    # Backward-compatible top-1 table consumed by retrieval_figures.py.
    t1 = tests[tests.metric == "success@1"].rename(
        columns={"hinglish": "Q1_R@1", "english": "Q2_R@1", "penalty": "delta", "p": "mcnemar_p"})
    t1[["system", "Q1_R@1", "Q2_R@1", "delta", "ci_lo", "ci_hi", "mcnemar_p", "n"]].to_csv(
        OUT / "h4_v2_tests.csv", index=False)

    write_report(res, tests, rel, len(pairs), len(meta))


def write_report(res: pd.DataFrame, tests: pd.DataFrame, rel: dict, nq: int, ncase: int) -> None:
    f = lambda v: f"{v:.4f}"
    L = ["# Retrieval v2 -- passage chunking, matched content, hybrid fusion", "",
         f"n = {nq} query pairs over {ncase} cases. CPU only, no API calls.", "",
         "Every system reads the SAME full case text: lexical natively, dense via passages.", "",
         "## Metrics by system and query language", "",
         "Success@k = any relevant case in the top k (a hit rate; earlier files called it recall@k).",
         "nDCG@10 uses the corpus-level ideal DCG. Graded nDCG uses the automatic proxy:",
         "grade 2 = same condition group and >= 1 shared lexicon concept with the gold English",
         "question; grade 1 = same group only. It is not a clinical relevance judgment.", "",
         "| System | Query | S@1 | S@3 | S@5 | S@10 | MRR@10 | nDCG@10 | graded nDCG@10 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        L.append(f"| `{r.system}` | {r.variant} | {f(r['success@1'])} | {f(r['success@3'])} | "
                 f"{f(r['success@5'])} | {f(r['success@10'])} | {f(r['MRR@10'])} | "
                 f"{f(r['nDCG@10'])} | {f(r['graded_nDCG@10'])} |")
    L += ["| *random floor* | both | *0.0626* | | | | | | |", "",
          "## Code-mixing penalty (English - Hinglish) at every depth", "",
          "Success@k: exact McNemar. MRR / nDCG: Wilcoxon signed-rank. CIs: paired bootstrap, 10,000 resamples.", "",
          "| System | Metric | Hinglish | English | Penalty | 95% CI | p | Test |",
          "|---|---|---:|---:|---:|---|---:|---|"]
    for _, r in tests.iterrows():
        L.append(f"| `{r.system}` | {r.metric} | {f(r.hinglish)} | {f(r.english)} | "
                 f"**{r.penalty:+.4f}** | [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {r.p:.3g} | {r.test} |")
    sig = tests.p < 0.05
    L += ["", f"Penalty tests significant at p < 0.05: **{int(sig.sum())} of {len(tests)}**; "
          f"penalty positive (English better): **{int((tests.penalty > 0).sum())} of {len(tests)}**.", "",
          "## Relevance diagnostics", "",
          f"- Relevant cases per query (same condition group): median {int(np.median(rel['n_relevant']))}, "
          f"min {int(rel['n_relevant'].min())}, max {int(rel['n_relevant'].max())}.",
          f"- Queries whose gold English question contains >= 1 lexicon concept: "
          f"{rel['query_has_concept'].mean():.1%} (the rest can only reach grade 1).",
          f"- Grade-2 cases per query: median {int(np.median(rel['grade_counts'][:, 2]))}.", ""]
    (OUT / "retrieval_v2_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
