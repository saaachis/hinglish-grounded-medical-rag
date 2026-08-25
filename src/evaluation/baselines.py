"""Degenerate baselines that every results table must carry.

WHY THIS EXISTS. `concept_lexicon.score` reports `factual_support = hit / |output
concepts|` -- precision with no recall term. Its degenerate optimum is therefore to
assert exactly one concept that is common in the corpus. Measured against the M4'
caption reference on 1,154 rows:

    constant "swelling"                     0.7192
    constant "swelling and erythema"        0.6620
    constant "erythema"                     0.6049
    GROUNDED SYSTEM                         0.1528
    ZERO-SHOT SYSTEM                        0.1066
    constant "pain"                         0.0035

A single hard-coded word beats the real system by 4.7x. So an absolute
`factual_support` number carries no information about answer quality: it mostly
reports how FEW concepts the answer asserted. (Note "pain" scores 0.0035 -- the
baseline value depends entirely on how common the chosen concept is in the
reference, which is itself the point.)

WHAT IS STILL VALID. Paired deltas between two arms scored the same way remain
meaningful, because the degenerate advantage applies equally to both. What is NOT
valid is quoting a level -- "the system achieves 0.55 factual support" -- as
evidence of quality.

WHAT TO DO. Ship `baseline_rows()` in every results table. A reviewer who sees the
system beating a constant-answer row is reassured; one who sees it losing learns
something important, and better from the authors than from a referee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.concept_lexicon import score

#: Fixed answers that assert 1-2 very common concepts. These probe the metric's
#: floor: anything a real system cannot beat is a metric failure, not a system one.
CONSTANT_ANSWERS: dict[str, str] = {
    "const:swelling": "swelling",
    "const:erythema": "erythema",
    "const:pain": "pain",
    "const:swelling+erythema": "swelling and erythema",
    "const:six-common": "rash swelling pain erythema infection lesion",
}


def baseline_rows(
    references: pd.Series, metric: str = "factual_support",
) -> pd.DataFrame:
    """Score every constant answer against `references`.

    Also includes a `copy:reference` row -- an oracle that echoes the reference
    verbatim. That is the metric's true ceiling and shows how much headroom the
    scale actually has.
    """
    refs = references.dropna().astype(str)
    rows = []
    for label, answer in CONSTANT_ANSWERS.items():
        vals = np.array([score(answer, r)[metric] for r in refs], dtype=float)
        rows.append({"system": label, "answer": answer,
                     metric: np.nanmean(vals), "n": int(np.sum(~np.isnan(vals)))})

    copied = np.array([score(r, r)[metric] for r in refs], dtype=float)
    rows.append({"system": "copy:reference", "answer": "<the reference verbatim>",
                 metric: np.nanmean(copied), "n": int(np.sum(~np.isnan(copied)))})
    return pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)


def annotate(results: pd.DataFrame, references: pd.Series,
             metric: str = "factual_support") -> pd.DataFrame:
    """Append the baseline rows to a results frame, tagged for the reader."""
    b = baseline_rows(references, metric=metric)
    b["kind"] = "degenerate baseline"
    out = results.copy()
    if "kind" not in out:
        out["kind"] = "system"
    return pd.concat([out, b], ignore_index=True)


__all__ = ["CONSTANT_ANSWERS", "baseline_rows", "annotate"]
