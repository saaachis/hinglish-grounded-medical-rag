"""Re-score the cached 1,165 generations under M4' and the unified lexicon.

Produces the paper's central metric table: the SAME generations scored three ways.

    evidence-based   circular  -- the grounded arm was conditioned on this text
    M4' caption      unbiased  -- neither arm ever saw the image description
    coverage         diagnostic -- how often the metric measures anything at all

No API calls. Writes results/m4_caption/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.caption_reference import cluster_bootstrap_ci, score_frame
from src.evaluation.concept_lexicon import score as concept_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CACHED = Path("results/combined_h1h2/combined_scored.csv")
PAIRS = Path("data/processed/mmcqsd_multicare_paired.csv")
OUT = Path("results/m4_caption")

ARMS = {"grounded": "grounded_output", "zero": "zero_shot_output"}


def main() -> None:
    df = pd.read_csv(CACHED).merge(
        pd.read_csv(PAIRS)[["pair_id", "evidence_text", "english_summary", "condition_query"]],
        on="pair_id", how="left",
    )
    df["condition"] = df["condition"].fillna(df["condition_query"])
    logger.info("Loaded %d cached generations", len(df))

    # ---- evidence-based, unified lexicon (the circular but generous metric) ----
    for label, col in ARMS.items():
        s = df.apply(lambda r: concept_score(r[col], r["evidence_text"]), axis=1, result_type="expand")
        df[f"{label}_ev_factual"] = s["factual_support"]
        df[f"{label}_ev_halluc"] = s["hallucination"]
        df[f"{label}_ev_has_concepts"] = s["output_has_concepts"]

    # ---- M4' caption-reference (the unbiased but narrow metric) ----
    per_row, per_cond = score_frame(df, ARMS, summary_col="english_summary",
                                    condition_col="condition")

    OUT.mkdir(parents=True, exist_ok=True)
    per_row.to_csv(OUT / "m4_scored.csv", index=False, encoding="utf-8")
    per_cond.to_csv(OUT / "m4_per_condition.csv", index=False, encoding="utf-8")

    # ---- headline comparison, with cluster-aware CIs for M4' ----
    lines = [
        "# M4' -- reference-based factuality vs the circular evidence-based metric",
        "",
        f"n = {len(per_row)} cached generations. No API calls. "
        "Both metrics use the unified word-boundary lexicon with no 0.25 default.",
        "",
        "## Headline",
        "",
        "| Metric | Reference | Zero-shot | Grounded | Delta |",
        "|---|---|---:|---:|---:|",
    ]

    summary_rows = []
    for metric, ref, suffix in [
        ("Evidence-based (circular)", "retrieved evidence", "ev_factual"),
        ("M4' caption (unbiased)", "image description", "m4_factual"),
    ]:
        z = per_row[f"zero_{suffix}"]
        g = per_row[f"grounded_{suffix}"]
        both = per_row[[f"zero_{suffix}", f"grounded_{suffix}"]].dropna()
        d = both[f"grounded_{suffix}"] - both[f"zero_{suffix}"]
        p = stats.wilcoxon(d)[1] if len(d) > 10 and d.abs().sum() > 0 else np.nan
        eff = d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
        lines.append(f"| {metric} | {ref} | {z.mean():.4f} | {g.mean():.4f} | "
                     f"**{g.mean() - z.mean():+.4f}** |")
        summary_rows.append({"metric": metric, "n_paired": len(d),
                             "zero": z.mean(), "grounded": g.mean(),
                             "delta": g.mean() - z.mean(), "cohens_d": eff,
                             "wilcoxon_p": p})

    lines += ["", "## Paired tests", "",
              "| Metric | n paired | Cohen's d | Wilcoxon p |", "|---|---:|---:|---:|"]
    for r in summary_rows:
        lines.append(f"| {r['metric']} | {r['n_paired']} | {r['cohens_d']:.3f} | {r['wilcoxon_p']:.3e} |")

    # The correct test: cluster-bootstrap the PAIRED DIFFERENCE. Marginal CIs on
    # each arm can overlap while the paired delta is still clearly non-zero.
    lines += ["", "## Grounding effect with CLUSTER-bootstrap CIs on the paired delta", "",
              "Resamples descriptions, not rows -- one description covers 22% of the corpus, "
              "so a row-level bootstrap would badly understate these intervals.", "",
              "| Metric | delta | 95% cluster CI | excludes 0? |", "|---|---:|---|---|"]
    for metric, suffix in [("Evidence-based (circular)", "ev_factual"),
                           ("M4' caption (unbiased)", "m4_factual")]:
        s = per_row[[f"zero_{suffix}", f"grounded_{suffix}", "m4_description"]].dropna(
            subset=[f"zero_{suffix}", f"grounded_{suffix}"]).copy()
        s["_v"] = s[f"grounded_{suffix}"] - s[f"zero_{suffix}"]
        m, lo, hi = cluster_bootstrap_ci(s, "_v")
        lines.append(f"| {metric} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     f"{'yes' if lo > 0 or hi < 0 else 'NO -- n.s.'} |")

    lines += ["", "### Per-arm marginal CIs (for reference only -- not the test)", "",
              "| Arm | mean | 95% CI |", "|---|---:|---|"]
    for label in ARMS:
        m, lo, hi = cluster_bootstrap_ci(per_row, f"{label}_m4_factual")
        lines.append(f"| {label} | {m:.4f} | [{lo:.4f}, {hi:.4f}] |")

    n_uni = per_row.loc[per_row.m4_description.str.len() > 0, "m4_description"].nunique()
    lines += [
        "", "> The reference has only "
        f"{n_uni} distinct values across {int((per_row.m4_description.str.len() > 0).sum())} rows, "
        "so a row-level bootstrap would badly understate these intervals. "
        "The CIs above resample descriptions.", "",
        "## Metric coverage (how often anything is measured at all)", "",
        "| Arm | evidence-based | M4' |", "|---|---:|---:|",
    ]
    for label in ARMS:
        lines.append(f"| {label} | {per_row[f'{label}_ev_has_concepts'].mean():.1%} | "
                     f"{per_row[f'{label}_m4_output_has_concepts'].mean():.1%} |")

    lines += ["", "Per-condition table: `m4_per_condition.csv` -- **do not quote the aggregate "
              "without it** (see module docstring).", ""]

    pd.DataFrame(summary_rows).to_csv(OUT / "m4_summary.csv", index=False)
    (OUT / "m4_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
