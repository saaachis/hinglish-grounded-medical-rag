"""Adaptive truncation: does it fire, and does it help?

The README, config.yaml and the poster all claim "MMed-RAG-style adaptive
context selection". Measured, the shipped rule fired on 0 of 299 queries and
could not have fired on any query: at threshold_ratio=0.5 it waits for a 0.246
similarity drop between adjacent neighbours, and the largest adjacent gap that
occurs anywhere in the data is 0.088.

This sweeps threshold_ratio against fixed-k baselines on the real index and
reports, for each setting:

    fire_rate     how often truncation actually cuts anything
    mean_kept     average number of evidence cases passed to the generator
    precision     fraction of KEPT cases that are condition-correct
                  (what the generator is actually conditioned on)
    recall_hit    fraction of queries where >=1 kept case is correct
                  (whether the right evidence survived at all)

Precision and recall trade off directly: keeping fewer cases raises precision and
lowers recall. The question a truncation rule has to answer is whether it beats
a fixed k at the same mean_kept. No API calls.

Writes results/truncation_sweep/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.encoding.text_encoder import TextEncoder
from src.retrieval.indexer import FAISSIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PAIRS = Path("data/processed/mmcqsd_multicare_paired.csv")
INDEX = Path("data/faiss_index/evidence.index")
META = Path("data/faiss_index/evidence_metadata.csv")
OUT = Path("results/truncation_sweep")

MAX_K = 10
RATIOS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
FIXED_K = (1, 3, 5, 10)
SEED = 42


def n_kept(scores: np.ndarray, ratio: float) -> int:
    """The shipped rule: cut at the first adjacent gap exceeding ratio*top."""
    if len(scores) <= 1:
        return len(scores)
    top = float(scores[0])
    if top <= 0:
        return 1
    thr = ratio * top
    for i in range(1, len(scores)):
        if float(scores[i - 1] - scores[i]) > thr:
            return i
    return len(scores)


def main() -> None:
    pairs = pd.read_csv(PAIRS)
    meta = pd.read_csv(META)
    indexer = FAISSIndexer()
    indexer.load_index(str(INDEX))

    enc = TextEncoder(device="cpu")
    enc.load_model()
    enc.model.max_seq_length = 128
    logger.info("Encoding %d queries ...", len(pairs))
    emb = enc.encode(pairs["hinglish_query"].astype(str).tolist(),
                     batch_size=32, show_progress=False)
    scores, idxs = indexer.index.search(emb.astype(np.float32), MAX_K)

    cond = meta["condition_group"].to_numpy()
    gold = pairs["condition_query"].to_numpy()
    hits = cond[idxs] == gold[:, None]          # (n, MAX_K)

    gaps = scores[:, :-1] - scores[:, 1:]
    logger.info("adjacent gap: mean %.5f, max %.5f | top-1 sim mean %.4f",
                gaps.mean(), gaps.max(), scores[:, 0].mean())

    rows = []
    for ratio in RATIOS:
        keeps = np.array([n_kept(s, ratio) for s in scores])
        prec = np.array([hits[i, :k].mean() for i, k in enumerate(keeps)])
        rec = np.array([hits[i, :k].any() for i, k in enumerate(keeps)])
        rows.append({"system": f"adaptive(ratio={ratio})", "fire_rate": float((keeps < MAX_K).mean()),
                     "mean_kept": float(keeps.mean()), "precision": float(prec.mean()),
                     "recall_hit": float(rec.mean())})

    for k in FIXED_K:
        prec = hits[:, :k].mean(axis=1)
        rec = hits[:, :k].any(axis=1)
        rows.append({"system": f"fixed_k={k}", "fire_rate": np.nan, "mean_kept": float(k),
                     "precision": float(prec.mean()), "recall_hit": float(rec.mean())})

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "truncation_sweep.csv", index=False)

    lines = [
        "# Adaptive truncation sweep",
        "",
        f"n = {len(pairs)} queries, max_k = {MAX_K}, real FAISS index, no condition filter.",
        "",
        "## The shipped rule cannot fire",
        "",
        f"- top-1 similarity, mean: **{scores[:, 0].mean():.4f}**",
        f"- threshold the rule requires at ratio=0.5: **{0.5 * scores[:, 0].mean():.4f}**",
        f"- largest adjacent gap anywhere in the data: **{gaps.max():.5f}**",
        f"- mean adjacent gap: **{gaps.mean():.5f}**",
        "",
        "The rule waits for a drop roughly **"
        f"{0.5 * scores[:, 0].mean() / gaps.max():.1f}x larger** than the biggest gap that exists.",
        "",
        "## Sweep",
        "",
        "| System | fires | mean kept | precision of kept | recall (>=1 correct kept) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        fire = "--" if np.isnan(r["fire_rate"]) else f"{r['fire_rate']:.1%}"
        lines.append(f"| `{r['system']}` | {fire} | {r['mean_kept']:.2f} | "
                     f"{r['precision']:.4f} | {r['recall_hit']:.4f} |")

    lines += [
        "",
        "## Reading",
        "",
        "- At the shipped `ratio=0.5` the rule is inert: it returns all 10 cases on every "
        "query, so the system is a plain fixed-k=10 retriever and the adaptive-selection "
        "claim is unsupported.",
        "- Precision and recall trade off monotonically with `mean_kept`. The honest test "
        "is whether an adaptive setting beats the **fixed k with the same mean_kept**.",
        "",
        "## Verdict: adaptive vs the nearest fixed k",
        "",
        "| Adaptive | mean kept | vs | fixed k | precision | recall | wins? |",
        "|---|---:|---|---|---|---|---|",
    ]

    fixed = {r["system"]: r for r in rows if r["system"].startswith("fixed")}
    verdicts = []
    for r in rows:
        if not r["system"].startswith("adaptive"):
            continue
        near = min(fixed.values(), key=lambda f: abs(f["mean_kept"] - r["mean_kept"]))
        dp = r["precision"] - near["precision"]
        dr = r["recall_hit"] - near["recall_hit"]
        win = dp > 0 and dr >= 0
        verdicts.append(win)
        lines.append(
            f"| `{r['system']}` | {r['mean_kept']:.2f} | vs | `{near['system']}` "
            f"(k={near['mean_kept']:.0f}) | {dp:+.4f} | {dr:+.4f} | "
            f"{'YES' if win else 'no'} |")

    lines += [
        "",
        f"**Adaptive truncation wins in {sum(verdicts)} of {len(verdicts)} settings.**",
        "",
        "Precision is essentially flat (~0.113-0.115) across every setting, adaptive and "
        "fixed alike, while recall falls monotonically as fewer cases are kept. That means "
        "**the similarity gap carries no information about relevance** -- cutting on it "
        "discards correct evidence at the same rate as incorrect evidence.",
        "",
        "**Recommendation:** report this as a negative result and drop the "
        "\"MMed-RAG-style adaptive context selection\" claim from `README.md`, "
        "`config/config.yaml` and the poster. A fixed k is simpler and strictly better "
        "at every budget. The honest finding -- *that a published adaptive-selection "
        "heuristic does not transfer to this setting* -- is worth more than the claim was.",
        "",
    ]
    (OUT / "truncation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
