"""M4' -- reference-based factuality against the MMCQSD image caption.

THE CIRCULARITY PROBLEM. Every published H1 number scores both arms against the
retrieved evidence. The grounded arm was conditioned on that exact text; the
zero-shot arm never saw it. Much of the measured "gain" is therefore a property
of the metric, not of the system.

WHY THE OBVIOUS FIX FAILS. Scoring against MMCQSD's `english_summary` was
proposed as the cure. It is not: that field is a restated *question* plus an
image caption, so scoring an answer against it measures question-echo. Worse,
the caption names the condition_group label in 96.2% of rows.

WHAT ACTUALLY WORKS. Strip the question clause AND the boilerplate label clause,
and what remains is a human-written description of the IMAGE:

    "The back of the throat has swelling with whitish mass accumulation."
    "Little red pinkish dots on the hands of the baby."

Neither text model ever saw this. Scoring both arms against it is genuinely
reference-based with no circularity.

============================ READ BEFORE USE ============================
The reference is LOW-CARDINALITY. Across the 2,988 rows that carry a
description (99.1% of the corpus) there are only 412 unique strings -- 13.8%
distinct. One covers 671 rows (22.3% of the corpus), and skin_rash's 1,046
rows share just 80 descriptions.

Two consequences, both enforced by this module:

1. M4' measures "does the answer name the canonical visual findings for this
   condition", not "is this answer factual for this patient". Report it PER
   CONDITION -- `score_frame` refuses to hand back a bare aggregate.
2. Rows sharing a description are NOT independent. Significance tests must
   cluster on the description; `cluster_bootstrap_ci` resamples clusters, not
   rows. Treating 2,923 rows as independent makes p-values anti-conservative.

M4' is also NARROWER than evidence-based scoring (~1.5 vs ~3.9 concepts per
reference): it covers visible findings only, so an answer that is correct about
cause or treatment scores zero. Report it as the UNBIASED metric beside the
evidence-based GENEROUS one -- the pair is the contribution, not either alone.
=========================================================================
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.evaluation.concept_lexicon import score as concept_score

#: Everything from "The image ..." onward is the caption.
CAPTION_RX = re.compile(r"\s*The image\b.*$", re.IGNORECASE | re.DOTALL)

#: The templated clause that names the condition_group -- the relevance answer
#: key. Must be removed: it is a dataset label, not a clinical observation.
BOILERPLATE_RX = re.compile(
    r"^The image(?:\s+here)?\s+shows\s+"
    r"(?:the condition of|a medical condition related to)?\s*[A-Za-z_ ]*?\s*[.,]",
    re.IGNORECASE,
)

MIN_DESCRIPTION_CHARS = 8


def split_summary(summary: str) -> tuple[str, str]:
    """Return ``(question_clause, caption_clause)``."""
    s = str(summary)
    q = CAPTION_RX.sub("", s).strip()
    return q, s[len(q):].strip()


def extract_description(summary: str) -> str:
    """Return the human-written image description, boilerplate label removed."""
    _, caption = split_summary(summary)
    if not caption:
        return ""
    desc = BOILERPLATE_RX.sub("", caption).strip()
    if not desc:
        # No trailing description -- caption was label boilerplate only.
        return ""
    return desc if len(desc) >= MIN_DESCRIPTION_CHARS else ""


def score_answer(answer: str, summary: str) -> dict[str, float]:
    """Score one answer against its image description."""
    desc = extract_description(summary)
    if not desc:
        return {"m4_factual": np.nan, "m4_halluc": np.nan, "m4_concept_f1": np.nan,
                "m4_has_reference": False, "m4_output_has_concepts": False,
                "m4_description": ""}
    s = concept_score(answer, desc)
    return {"m4_factual": s["factual_support"], "m4_halluc": s["hallucination"],
            "m4_concept_f1": s["concept_f1"], "m4_has_reference": True,
            "m4_output_has_concepts": s["output_has_concepts"],
            "m4_description": desc}


def cluster_bootstrap_ci(
    df: pd.DataFrame, value_col: str, cluster_col: str = "m4_description",
    n_boot: int = 10_000, seed: int = 42, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap a mean CI by resampling CLUSTERS, not rows.

    With one description covering 653 rows, row-level resampling would treat 653
    copies of the same reference as independent evidence and produce a CI far
    too narrow. Returns ``(mean, lo, hi)``.
    """
    sub = df[[value_col, cluster_col]].dropna(subset=[value_col])
    if sub.empty:
        return np.nan, np.nan, np.nan
    groups = [g[value_col].to_numpy() for _, g in sub.groupby(cluster_col)]
    rng = np.random.default_rng(seed)
    n = len(groups)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        means[b] = np.concatenate([groups[i] for i in pick]).mean()
    return (float(sub[value_col].mean()),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def score_frame(
    df: pd.DataFrame, answer_cols: dict[str, str], summary_col: str = "english_summary",
    condition_col: str = "condition",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score several answer columns and return ``(per_row, per_condition)``.

    A per-condition frame is returned ALONGSIDE the rows deliberately -- see the
    module docstring. Do not publish an aggregate without it.
    """
    out = df.copy()
    out["m4_description"] = out[summary_col].apply(extract_description)

    for label, col in answer_cols.items():
        scored = out.apply(lambda r: score_answer(r[col], r[summary_col]), axis=1, result_type="expand")
        for k in ("m4_factual", "m4_halluc", "m4_concept_f1", "m4_output_has_concepts"):
            out[f"{label}_{k}"] = scored[k]

    rows = []
    for cond, g in out.groupby(condition_col):
        rec: dict[str, object] = {"condition": cond, "n": len(g),
                                  "n_unique_descriptions": g["m4_description"].nunique()}
        for label in answer_cols:
            col = f"{label}_m4_factual"
            m, lo, hi = cluster_bootstrap_ci(
                g.rename(columns={col: "_v"}), "_v") if col in g else (np.nan,) * 3
            rec[f"{label}_factual"] = m
            rec[f"{label}_ci_lo"] = lo
            rec[f"{label}_ci_hi"] = hi
            rec[f"{label}_coverage"] = g[f"{label}_m4_output_has_concepts"].mean()
        rows.append(rec)

    return out, pd.DataFrame(rows).sort_values("n", ascending=False)


__all__ = ["extract_description", "split_summary", "score_answer",
           "score_frame", "cluster_bootstrap_ci"]
