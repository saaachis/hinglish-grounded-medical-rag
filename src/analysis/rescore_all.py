"""Re-score EVERY cached generation through the repaired lexicon. Single entrypoint.

WHY THIS EXISTS. Every result in this repo was scored before three metric defects
were found, so every number downstream of a generation is stale:

1. Negation scoped forward only, so post-posed Hinglish denials counted as
   assertions -- `extract_concepts("rash nahi hai")` returned `{"rash"}`. Grounded
   outputs are overwhelmingly hedges of that shape.
2. `factual_support` is precision with no recall term, so its degenerate optimum is
   a one-word answer. Absolute levels are uninterpretable without a baseline row.
3. `hallucination` is exactly `1 - factual_support`, so the published
   "+73.5% factual AND -44% hallucination" pair double-counts a single result.

This re-scores the cached generations -- no API calls, nothing regenerated -- and
emits precision / recall / F1 against two references, with degenerate baselines
attached to every table.

    evidence-based   the retrieved case (CIRCULAR: the grounded arm saw this text)
    caption-based    the MMCQSD image description (UNBIASED: neither arm saw it)

Run:  python -m src.analysis.rescore_all
Writes results/rescored/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.baselines import baseline_rows
from src.evaluation.caption_reference import cluster_bootstrap_ci, extract_description
from src.evaluation.concept_lexicon import extract_concepts

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PAIRS = Path("data/processed/mmcqsd_multicare_paired.csv")
OUT = Path("results/rescored")

#: Every cached generation set, and which columns hold its arms.
SOURCES: dict[str, dict] = {
    "llama_oracle_n1165": {
        "path": Path("results/combined_h1h2/combined_scored.csv"),
        "arms": {"zero": "zero_shot_output", "grounded": "grounded_output"},
        "note": "the published headline; generator llama-3.1-8b-instant (decommissioned)",
    },
    "gptoss20b_n268": {
        "path": Path("results/h1_real_retrieval/h1_real_scored.csv"),
        "arms": {"zero": "zero_shot_output", "oracle": "grounded_output_oracle",
                 "real": "grounded_output_real"},
        "note": "gpt-oss-20b; refuses ~83%, so its means describe refusal text",
    },
    "gptoss120b": {
        "path": Path("results/h1_real_120b/h1_real_scored.csv"),
        "arms": {"zero": "zero_shot_output", "oracle": "grounded_output_oracle",
                 "real": "grounded_output_real"},
        "note": "gpt-oss-120b; refuses ~31%, the usable replication",
    },
}


def prf(output: str, reference: str) -> tuple[float, float, float, int, int]:
    """Precision, recall, F1, |output concepts|, |reference concepts|."""
    o, r = extract_concepts(output), extract_concepts(reference)
    n_o, n_r = len(o), len(r)
    if n_o == 0:
        return np.nan, np.nan, np.nan, n_o, n_r
    hit = len(o & r)
    p = hit / n_o
    if n_r == 0:
        return p, np.nan, np.nan, n_o, n_r
    rc = hit / n_r
    f = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
    return p, rc, f, n_o, n_r


def score_source(name: str, cfg: dict, pairs: pd.DataFrame) -> pd.DataFrame | None:
    if not cfg["path"].exists():
        logger.warning("skip %s -- %s missing", name, cfg["path"])
        return None
    df = pd.read_csv(cfg["path"]).merge(
        pairs[["pair_id", "evidence_text", "english_summary", "condition_query"]],
        on="pair_id", how="left", suffixes=("", "_p"))
    df["caption_ref"] = df["english_summary"].apply(extract_description)
    logger.info("%s: %d rows", name, len(df))

    for arm, col in cfg["arms"].items():
        for ref_name, ref_col in [("ev", "evidence_text"), ("cap", "caption_ref")]:
            vals = [prf(a, b) for a, b in zip(df[col].astype(str), df[ref_col].astype(str))]
            v = pd.DataFrame(vals, columns=["p", "r", "f", "n_out", "n_ref"], index=df.index)
            df[f"{arm}_{ref_name}_precision"] = v.p
            df[f"{arm}_{ref_name}_recall"] = v.r
            df[f"{arm}_{ref_name}_f1"] = v.f
            df[f"{arm}_{ref_name}_n_out"] = v.n_out
    df["source"] = name
    return df


def contrasts(df: pd.DataFrame, arms: list[str]) -> list[dict]:
    """Every grounded arm vs zero-shot, on both references, all three metrics."""
    rows = []
    for arm in [a for a in arms if a != "zero"]:
        for ref in ("ev", "cap"):
            for metric in ("precision", "recall", "f1"):
                gc, zc = f"{arm}_{ref}_{metric}", f"zero_{ref}_{metric}"
                if gc not in df or zc not in df:
                    continue
                s = df[[gc, zc]].dropna()
                if len(s) < 10:
                    continue
                d = s[gc] - s[zc]
                p = stats.wilcoxon(d)[1] if d.abs().sum() > 0 else 1.0
                rows.append({
                    "arm": arm, "reference": "evidence (circular)" if ref == "ev" else "caption (unbiased)",
                    "metric": metric, "n_paired": len(s),
                    "zero": s[zc].mean(), "grounded": s[gc].mean(), "delta": d.mean(),
                    "cohens_d": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
                    "wilcoxon_p": p,
                })
    return rows


def main() -> None:
    pairs = pd.read_csv(PAIRS)
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows, all_frames = [], []
    for name, cfg in SOURCES.items():
        df = score_source(name, cfg, pairs)
        if df is None:
            continue
        df.to_csv(OUT / f"rescored_{name}.csv", index=False, encoding="utf-8")
        all_frames.append(df)
        for r in contrasts(df, list(cfg["arms"])):
            all_rows.append({"source": name, **r})

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT / "contrasts.csv", index=False)

    # Benjamini-Hochberg across the WHOLE family -- these are ~30 related tests,
    # and reporting raw p-values across that many contrasts invites the obvious
    # multiple-comparisons objection.
    if len(res):
        order = res.wilcoxon_p.rank(method="first")
        res["p_bh"] = (res.wilcoxon_p * len(res) / order).clip(upper=1.0)
        res = res.sort_values(["source", "arm", "reference", "metric"])
        res.to_csv(OUT / "contrasts.csv", index=False)

    # Degenerate baselines against the unbiased reference.
    caption_refs = pd.concat([f["caption_ref"] for f in all_frames]).dropna()
    caption_refs = caption_refs[caption_refs.str.len() > 0]
    base = baseline_rows(caption_refs)
    base.to_csv(OUT / "degenerate_baselines.csv", index=False)

    L = ["# Re-scored results (repaired lexicon)", "",
         "Every cached generation re-scored with negation fixed and precision/recall/F1",
         "reported separately. No API calls. `hallucination` is omitted throughout: it is",
         "exactly `1 - precision`, so reporting it separately double-counts one result.", "",
         "## Contrasts (grounded vs zero-shot), BH-corrected across the whole family", "",
         "| Source | Arm | Reference | Metric | n | zero | grounded | delta | d | p (BH) |",
         "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        L.append(f"| {r['source']} | {r['arm']} | {r['reference']} | {r['metric']} | "
                 f"{r['n_paired']} | {r['zero']:.4f} | {r['grounded']:.4f} | "
                 f"**{r['delta']:+.4f}** | {r['cohens_d']:.3f} | {r['p_bh']:.3g} |")

    L += ["", "## ⚠️ Degenerate baselines on the unbiased (caption) reference", "",
          "`precision` has no recall term, so its optimum is a one-word answer. Any absolute",
          "level below these rows is a metric failure, not a system failure.", "",
          "| System | Answer | precision | n |", "|---|---|---:|---:|"]
    for _, r in base.iterrows():
        L.append(f"| `{r['system']}` | {r['answer'][:44]} | {r['factual_support']:.4f} | {r['n']} |")

    L += ["", "## Refusal / coverage -- report beside every number above", "",
          "| Source | Arm | outputs asserting >=1 concept |", "|---|---|---:|"]
    for f in all_frames:
        src = f["source"].iloc[0]
        for arm in SOURCES[src]["arms"]:
            col = f"{arm}_ev_n_out"
            if col in f:
                L.append(f"| {src} | {arm} | {(f[col] > 0).mean():.1%} |")

    L += ["", "> Rows where an arm asserts no concept score `nan` and vanish from a naive",
          "> mean. Coverage must be reported or the system looks healthiest where it fails.", ""]

    (OUT / "rescored_report.md").write_text("\n".join(L), encoding="utf-8")
    logger.info("Wrote %s", OUT / "rescored_report.md")
    print("\n".join(L))


if __name__ == "__main__":
    main()
