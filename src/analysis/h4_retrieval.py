"""H04 -- the retrieval-stage code-mixing penalty.

    H04: retrieval quality does not differ between code-mixed Hinglish queries
         and semantically equivalent English renderings, with encoder, index and
         relevance criterion held constant.

Three query conditions over the SAME 3,015 pairs and the SAME FAISS index:

    Q1  hinglish_query                     the deployed path
    Q2  English question, caption stripped the translation ceiling
    Q3  full English summary incl. caption the multimodal ceiling

WHY Q2 EXISTS. MMCQSD's `english_summary` is a restated question followed by an
image caption of the form "The image here shows a medical condition related to
swollen_tonsils." That caption contains the underscore-joined condition-group
label -- which is exactly the relevance label for this evaluation. Retrieving
with the full summary therefore retrieves with the answer key, and any
"English vs Hinglish" gap measured that way is inflated. Q2 removes it.

Q3 is retained deliberately: the Q2->Q3 gap is the headroom a perfect image
reader would add, which is an evidence-based argument for multimodal work
without performing any multimodal work.

Relevance: a retrieved case is relevant iff its `condition_group` equals the
query's. Coarse (18 groups) but free and label-consistent -- state as a
limitation.

No API calls. LaBSE on CPU + the existing FAISS index.

Writes: results/h4_retrieval/h4_metrics.csv
        results/h4_retrieval/h4_per_condition.csv
        results/h4_retrieval/h4_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.encoding.text_encoder import TextEncoder
from src.retrieval.indexer import FAISSIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PAIRS_PATH = Path("data/processed/mmcqsd_multicare_paired.csv")
INDEX_PATH = Path("data/faiss_index/evidence.index")
META_PATH = Path("data/faiss_index/evidence_metadata.csv")
OUTPUT_DIR = Path("results/h4_retrieval")

TOP_K = 10
K_VALUES = (1, 3, 5, 10)
N_BOOTSTRAP = 10_000
SEED = 42

# MUST match how the index was built (`build_index.py` sets
# `encoder.model.max_seq_length = 128`). Queries and documents have to be encoded
# under the same truncation or the vectors are not comparable and every
# similarity is subtly wrong. It is also ~2-4x faster, since attention cost grows
# quadratically in sequence length.
MAX_SEQ_LENGTH = 128
# Benchmarked on the target laptop (4 cores / 12 GB): 16 is fastest. Larger
# batches are slower here because the machine is memory-bound, not compute-bound
# (0.46 texts/s at bs=16 vs 0.38 at bs=64).
ENCODE_BATCH = 16

EMB_CACHE_DIR = Path("data/embeddings/h4")

# Everything from "The image ..." onward is the caption.
CAPTION_RE = re.compile(r"\s*The image\b.*$", re.IGNORECASE | re.DOTALL)
LEAD_RE = re.compile(r"^\s*summary\s*:\s*", re.IGNORECASE)


def strip_caption(summary: str) -> str:
    return CAPTION_RE.sub("", LEAD_RE.sub("", str(summary))).strip()


def assert_no_leakage(q2: pd.Series, conditions: pd.Series) -> None:
    """Hard gate: Q2 must not contain the caption or the machine-readable label.

    NOTE: we deliberately do NOT assert the absence of the condition's individual
    WORDS. A patient question legitimately describes its own symptom ("a lump in
    my neck"), and 79.6% of stripped questions do. That is the query, not
    leakage. The leak is the templated underscore-joined group label.
    """
    still_captioned = q2.str.contains(r"The image", case=False, regex=True).sum()
    if still_captioned:
        raise AssertionError(f"{still_captioned} Q2 rows still contain a caption")

    labels = sorted(set(conditions.astype(str)))
    hits = 0
    for lab in labels:
        if "_" not in lab:
            continue
        hits += q2.str.contains(re.escape(lab), case=False, regex=True).sum()
    if hits:
        raise AssertionError(f"{hits} Q2 rows still contain an underscore-joined label")

    empty = (q2.str.len() == 0).sum()
    if empty:
        raise AssertionError(f"{empty} Q2 rows are empty after stripping")
    logger.info("Leakage gate PASSED (no captions, no group labels, no empties)")


def encode_cached(texts: list[str], tag: str, encoder: TextEncoder) -> np.ndarray:
    """Encode with an on-disk cache.

    Encoding is the dominant cost on CPU (~0.4 texts/s on the target laptop), so
    a partial or interrupted run must never have to redo finished work. The cache
    key includes the text content and max_seq_length, so changing either
    correctly invalidates it.
    """
    key = hashlib.sha1(
        ("\x00".join(texts) + f"|msl={MAX_SEQ_LENGTH}").encode("utf-8")
    ).hexdigest()[:16]
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = EMB_CACHE_DIR / f"{tag}_{key}.npy"
    if cache.exists():
        logger.info("Using cached embeddings for %s (%s)", tag, cache.name)
        return np.load(cache)

    emb = encoder.encode(texts, batch_size=ENCODE_BATCH, show_progress=True, normalize=True)
    np.save(cache, emb)
    logger.info("Cached embeddings for %s -> %s", tag, cache.name)
    return emb


def retrieve(texts: list[str], tag: str, encoder: TextEncoder, indexer: FAISSIndexer) -> np.ndarray:
    """Return the top-k index ids for each text, shape (n, TOP_K).

    Uses the underlying FAISS index directly for a batched search:
    `FAISSIndexer.search` is single-query by contract (it returns `scores[0]`).
    """
    emb = encode_cached(texts, tag, encoder)
    _, idx = indexer.index.search(emb.astype(np.float32), TOP_K)
    return idx


def rank_metrics(hits: np.ndarray) -> dict[str, float]:
    """hits: (n, TOP_K) boolean relevance matrix."""
    out: dict[str, float] = {}
    for k in K_VALUES:
        out[f"recall@{k}"] = float(hits[:, :k].any(axis=1).mean())
    # reciprocal rank of the first relevant hit
    first = np.argmax(hits, axis=1)
    has = hits.any(axis=1)
    rr = np.where(has, 1.0 / (first + 1), 0.0)
    out["MRR@10"] = float(rr.mean())
    # nDCG with a single relevant grade; IDCG = 1 since ideal puts a hit at rank 1
    disc = 1.0 / np.log2(np.arange(2, TOP_K + 2))
    out["nDCG@10"] = float((hits * disc).sum(axis=1).clip(max=1.0).mean())
    return out


def reciprocal_ranks(hits: np.ndarray) -> np.ndarray:
    first = np.argmax(hits, axis=1)
    return np.where(hits.any(axis=1), 1.0 / (first + 1), 0.0)


def bootstrap_delta(a: np.ndarray, b: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    """Percentile CI for mean(a) - mean(b), paired resampling."""
    rng = np.random.default_rng(seed)
    n = len(a)
    boot = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        boot[i] = a[idx].mean() - b[idx].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar on paired binary outcomes. Returns (n01, n10, p)."""
    n01 = int(((~a.astype(bool)) & b.astype(bool)).sum())
    n10 = int((a.astype(bool) & (~b.astype(bool))).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    p = float(stats.binomtest(n10, n, 0.5).pvalue)
    return n01, n10, p


def random_floor(query_conditions: pd.Series, meta: pd.DataFrame) -> dict[str, float]:
    """Analytic prevalence-weighted floor: P(a random doc matches the condition)."""
    prev = meta["condition_group"].value_counts(normalize=True)
    p = query_conditions.map(prev).fillna(0.0).to_numpy()
    return {f"recall@{k}": float((1.0 - (1.0 - p) ** k).mean()) for k in K_VALUES}


def stratified_sample(pairs: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Condition-stratified subsample, proportional, with every group represented."""
    if n >= len(pairs):
        return pairs
    frac = n / len(pairs)
    parts = []
    for _, grp in pairs.groupby("condition_query"):
        take = max(1, int(round(len(grp) * frac)))
        parts.append(grp.sample(n=min(take, len(grp)), random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info("Stratified subsample: %d of %d pairs (seed=%d)", len(out), len(pairs), seed)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="H04 retrieval-stage code-mixing penalty")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Condition-stratified subsample size (0 = all 3,015). CPU encoding runs "
             "at ~0.4 texts/s, so the full run is ~6 h on a 4-core laptop; use a GPU "
             "session for the definitive numbers.",
    )
    args = ap.parse_args()

    for pth in (PAIRS_PATH, INDEX_PATH, META_PATH):
        if not pth.exists():
            raise SystemExit(f"Missing {pth} -- extract handoff-tier1-essential.zip first.")

    pairs = pd.read_csv(PAIRS_PATH)
    meta = pd.read_csv(META_PATH)
    logger.info("Loaded %d pairs, %d indexed cases", len(pairs), len(meta))
    if args.limit:
        pairs = stratified_sample(pairs, args.limit)

    q1 = pairs["hinglish_query"].astype(str)
    q3 = pairs["english_summary"].astype(str)
    q2 = q3.apply(strip_caption)
    assert_no_leakage(q2, pairs["condition_query"])

    try:
        import torch

        torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    except Exception:  # threading is an optimisation, never a hard requirement
        logger.warning("Could not set torch thread count; continuing at the default")

    encoder = TextEncoder(device="cpu")
    encoder.load_model()
    encoder.model.max_seq_length = MAX_SEQ_LENGTH
    logger.info("max_seq_length pinned to %d to match build_index.py", MAX_SEQ_LENGTH)
    indexer = FAISSIndexer()
    indexer.load_index(INDEX_PATH)
    if indexer.index.ntotal != len(meta):
        raise SystemExit(
            f"Index/metadata misalignment: index has {indexer.index.ntotal} vectors "
            f"but metadata has {len(meta)} rows. Relevance labels would be wrong."
        )

    meta_cond = meta["condition_group"].to_numpy()
    gold = pairs["condition_query"].astype(str).to_numpy()

    variants = {"Q1_hinglish": q1, "Q2_english_question": q2, "Q3_english_plus_caption": q3}
    hits: dict[str, np.ndarray] = {}
    rows = []
    for name, texts in variants.items():
        logger.info("Encoding + retrieving: %s", name)
        idx = retrieve(texts.tolist(), name, encoder, indexer)
        h = meta_cond[idx] == gold[:, None]
        hits[name] = h
        rows.append({"variant": name, **rank_metrics(h)})

    floor = random_floor(pairs["condition_query"], meta)
    rows.append({"variant": "random_floor(analytic)", **floor, "MRR@10": np.nan, "nDCG@10": np.nan})
    metrics = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "h4_metrics.csv", index=False)

    # ---- paired tests -------------------------------------------------------
    comparisons = [("Q1_hinglish", "Q2_english_question"), ("Q2_english_question", "Q3_english_plus_caption")]
    comp_rows = []
    for lo, hi in comparisons:
        a1, b1 = hits[lo][:, 0], hits[hi][:, 0]
        n01, n10, p_mc = mcnemar(a1, b1)
        rr_a, rr_b = reciprocal_ranks(hits[lo]), reciprocal_ranks(hits[hi])
        w = stats.wilcoxon(rr_a, rr_b, zero_method="zsplit")
        ci_lo, ci_hi = bootstrap_delta(b1.astype(float), a1.astype(float))
        comp_rows.append(
            {
                "comparison": f"{hi} - {lo}",
                "delta_recall@1": float(b1.mean() - a1.mean()),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "mcnemar_n01": n01,
                "mcnemar_n10": n10,
                "mcnemar_p": p_mc,
                "wilcoxon_rr_p": float(w.pvalue),
                "delta_MRR": float(rr_b.mean() - rr_a.mean()),
            }
        )
    comps = pd.DataFrame(comp_rows)

    # ---- per condition ------------------------------------------------------
    per_cond = []
    for cond, grp in pairs.groupby("condition_query"):
        m = pairs["condition_query"] == cond
        row = {"condition": cond, "n": int(m.sum())}
        for name in variants:
            row[f"{name}_R@1"] = float(hits[name][m.to_numpy(), 0].mean())
        per_cond.append(row)
    per_cond_df = pd.DataFrame(per_cond).sort_values("n", ascending=False)
    per_cond_df.to_csv(OUTPUT_DIR / "h4_per_condition.csv", index=False)

    # ---- report -------------------------------------------------------------
    L = [
        "# H04 - Retrieval-Stage Code-Mixing Penalty",
        "",
        f"n = {len(pairs)} pairs (of 3,015 available), index = {len(meta)} MultiCaRe cases, "
        f"encoder = LaBSE (max_seq_length={MAX_SEQ_LENGTH}), top-k = {TOP_K}.",
        "Relevance: retrieved case `condition_group` == query `condition_query`.",
        "No API calls.",
        "",
        "## Query conditions",
        "",
        "| Variant | Meaning |",
        "|---|---|",
        "| `Q1_hinglish` | The deployed path |",
        "| `Q2_english_question` | Translation ceiling (caption stripped) |",
        "| `Q3_english_plus_caption` | Multimodal ceiling (caption retained) |",
        "",
        "> Q3's caption contains the underscore-joined condition-group label, which is",
        "> the relevance label itself. Q3 is an upper bound, **not** an English baseline.",
        "> The unconfounded code-mixing penalty is Q1 vs Q2.",
        "",
        "## Retrieval quality",
        "",
        "| Variant | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in metrics.iterrows():
        def f(v):
            return "--" if pd.isna(v) else f"{v:.4f}"
        L.append(
            f"| `{r['variant']}` | {f(r['recall@1'])} | {f(r['recall@3'])} | {f(r['recall@5'])} | "
            f"{f(r['recall@10'])} | {f(r.get('MRR@10'))} | {f(r.get('nDCG@10'))} |"
        )

    L += ["", "## Paired comparisons", "",
          "| Comparison | dR@1 | 95% CI | McNemar n01/n10 | McNemar p | Wilcoxon(RR) p | dMRR |",
          "|---|---:|---|---|---:|---:|---:|"]
    for _, r in comps.iterrows():
        L.append(
            f"| {r['comparison']} | {r['delta_recall@1']:+.4f} | "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['mcnemar_n01']}/{r['mcnemar_n10']} | "
            f"{r['mcnemar_p']:.3g} | {r['wilcoxon_rr_p']:.3g} | {r['delta_MRR']:+.4f} |"
        )

    q1r, q2r = metrics.loc[0, "recall@1"], metrics.loc[1, "recall@1"]
    fl = floor["recall@1"]
    L += [
        "",
        "## Reading",
        "",
        f"- Deployed Hinglish R@1 = **{q1r:.4f}**, against an analytic random floor of "
        f"**{fl:.4f}** ({q1r / fl:.2f}x the floor).",
        f"- English question R@1 = **{q2r:.4f}**; the unconfounded code-mixing penalty is "
        f"**{q2r - q1r:+.4f}** absolute.",
        "- The Q2->Q3 increment is the headroom a perfect image reader would add.",
        "",
        "Per-condition breakdown: `h4_per_condition.csv`.",
    ]
    report = "\n".join(L)
    (OUTPUT_DIR / "h4_report.md").write_text(report, encoding="utf-8")
    comps.to_csv(OUTPUT_DIR / "h4_comparisons.csv", index=False)
    print(report)


if __name__ == "__main__":
    main()
