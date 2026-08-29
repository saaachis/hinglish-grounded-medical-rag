"""Analyse H1 under oracle vs real retrieval.

Answers the four questions the oracle problem left open:

    Q1  Does the grounding benefit survive REAL retrieval?
        (the deployed-system H1 -- the number the paper must report)
    Q2  How much does oracle retrieval inflate it?
        (ceiling vs deployed, as a designed contrast)
    Q3  What does the model DO when retrieval fails -- hallucinate, or refuse?
    Q4  Is factuality conditional on retrieval being correct?
        (if not, the retriever is not doing the work the architecture claims)

Q3 matters more than it looks. With R@1 ~ 0.11 the grounded arm is usually
handed another patient's case report, and a well-behaved model then declines.
Refusals assert no clinical concept, so the concept metric returns nan and they
vanish from a naive mean -- which would make the system look FINE precisely when
it is failing. Refusal rate is reported as a first-class outcome.

No API calls. Writes results/h1_real_retrieval/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Defaults preserve the original behaviour. They are overridable because the
# gpt-oss-120b run writes to results/h1_real_120b/, and its report was left
# stale at n=256 while the scored file grew to n=467 -- a committed report that
# contradicted the manuscript. Always regenerate after adding generations.
SCORED = Path("results/h1_real_retrieval/h1_real_scored.csv")
OUT = Path("results/h1_real_retrieval")


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar on paired binary outcomes."""
    n01 = int(((~a) & b).sum())
    n10 = int((a & (~b)).sum())
    if n01 + n10 == 0:
        return n01, n10, 1.0
    p = float(stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
    return n01, n10, p


def paired(df: pd.DataFrame, a: str, b: str) -> dict:
    """Wilcoxon on rows where BOTH arms produced a scoreable answer."""
    s = df[[a, b]].dropna()
    if len(s) < 10:
        return {"n": len(s), "mean_a": np.nan, "mean_b": np.nan,
                "delta": np.nan, "p": np.nan, "d": np.nan}
    d = s[b] - s[a]
    p = stats.wilcoxon(d)[1] if d.abs().sum() > 0 else 1.0
    return {"n": len(s), "mean_a": s[a].mean(), "mean_b": s[b].mean(),
            "delta": d.mean(), "p": float(p),
            "d": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser(description="H1 oracle vs real retrieval")
    ap.add_argument("--scored", type=Path, default=SCORED, help="scored CSV to analyse")
    ap.add_argument("--out", type=Path, default=OUT, help="directory for report + stats")
    args = ap.parse_args()
    scored, out_dir = args.scored, args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scored)
    n = len(df)
    logger.info("Loaded %d rows (model=%s, top_k=%s)", n, df["model"].iloc[0], df["top_k"].iloc[0])

    L = [f"# H1 under oracle vs real retrieval", "",
         f"n = {n} pairs · generator `{df['model'].iloc[0]}` · top-k = {df['top_k'].iloc[0]} · "
         "scored with the unified word-boundary lexicon (no 0.25 default).", "",
         "> The original generator (`llama-3.1-8b-instant`) was decommissioned by Groq "
         "mid-project and returns 404 on every key. All three arms were therefore "
         "regenerated on a current model so that oracle-vs-real is not confounded with "
         "a model change. The cached llama outputs are retained per row.", "",
         "---", "", "## Q3 first: what happens when retrieval fails", "",
         "| Arm | evidence | refusal rate | scoreable (has concepts) |",
         "|---|---|---:|---:|"]

    for arm, ev in [("zero", "none"), ("oracle", "condition-filtered (ceiling)"),
                    ("real", "FAISS top-k (deployed)")]:
        L.append(f"| {arm} | {ev} | {df[f'{arm}_is_refusal'].mean():.1%} | "
                 f"{df[f'{arm}_has_concepts'].mean():.1%} |")

    n01, n10, p = mcnemar(df["oracle_is_refusal"].to_numpy(bool),
                          df["real_is_refusal"].to_numpy(bool))
    L += ["", f"Oracle vs real refusal, McNemar: n01={n01}, n10={n10}, p={p:.4g}", "",
          "**The grounded arm declines rather than confabulates.** Refusals carry no "
          "clinical concept, so they are `nan` under the concept metric and disappear "
          "from a naive mean -- the system would look healthiest exactly where it fails. "
          "Report refusal rate beside every factuality number.", "",
          "---", "", "## Q1/Q2: grounding benefit, ceiling vs deployed", "",
          "| Contrast | n paired | zero-shot | grounded | delta | Cohen's d | Wilcoxon p |",
          "|---|---:|---:|---:|---:|---:|---:|"]

    rows = []
    for label, gcol in [("Oracle evidence (ceiling)", "oracle_factual"),
                        ("Real retrieval (deployed)", "real_factual")]:
        zcol = "zero_factual_vs_oracle" if "oracle" in gcol else "zero_factual_vs_real"
        r = paired(df, zcol, gcol)
        rows.append({"contrast": label, **r})
        L.append(f"| {label} | {r['n']} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | "
                 f"**{r['delta']:+.4f}** | {r['d']:.3f} | {r['p']:.3e} |")

    ro = paired(df, "real_factual", "oracle_factual")
    L += ["", f"Oracle − real grounded factuality: **{ro['delta']:+.4f}** "
          f"(n={ro['n']} both-scoreable, p={ro['p']:.3g}). "
          "This is the inflation the condition filter bought.", "",
          "---", "", "## Q4: is factuality conditional on retrieval being correct?", "",
          "| Retrieval top-1 | n | grounded factual (real) | refusal rate |",
          "|---|---:|---:|---:|"]

    for correct, sub in df.groupby("retrieval_top1_correct"):
        L.append(f"| {'correct' if correct else 'wrong'} | {len(sub)} | "
                 f"{sub['real_factual'].mean():.4f} | {sub['real_is_refusal'].mean():.1%} |")

    ok = df[df.retrieval_top1_correct]["real_factual"].dropna()
    bad = df[~df.retrieval_top1_correct]["real_factual"].dropna()
    if len(ok) >= 5 and len(bad) >= 5:
        u, pu = stats.mannwhitneyu(ok, bad, alternative="two-sided")
        L += ["", f"Mann-Whitney U = {u:.0f}, p = {pu:.4g}. "
              + ("Retrieval correctness predicts factuality -- the retriever is doing "
                 "the work the architecture claims." if pu < 0.05 else
                 "**Retrieval correctness does NOT predict factuality.** Either the "
                 "condition-group label is too coarse to capture usefulness, or the "
                 "generator is leaning on the prompt framing rather than the evidence. "
                 "This is an important negative result -- investigate before writing.")]

    L += ["", f"Retrieval top-1 correct: **{df.retrieval_top1_correct.mean():.1%}** · "
          f"any of top-k correct: **{df.retrieval_any_correct.mean():.1%}**", ""]

    pd.DataFrame(rows).to_csv(out_dir / "h1_oracle_vs_real_stats.csv", index=False)
    (out_dir / "h1_oracle_vs_real_report.md").write_text("\n".join(L), encoding="utf-8")
    logger.info("Wrote %s", out_dir / "h1_oracle_vs_real_report.md")
    print("\n".join(L))


if __name__ == "__main__":
    main()
