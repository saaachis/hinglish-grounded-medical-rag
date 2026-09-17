"""How large must an H03 provenance experiment be? (ICCSDI review, R3.6 / R1.e)

Reviewer 3: "H03 remains underpowered because only 13 queries were scoreable
across all evidence conditions; conclusions regarding evidence provenance should
remain explicitly provisional."
Reviewer 1: "larger provenance experiments".

Both are right, and the paper should say what "larger" would actually cost. Two
things are computed here from the existing n = 160 run:

1.  THE JOINT-SCOREABILITY PROBLEM. The four-way omnibus needs a query that is
    answered (not refused, and asserting at least one scoreable concept) under
    ALL FOUR evidence conditions. With per-condition refusal of 76-88% that joint
    event is rare, so the usable sample collapses. The empirical joint rate is
    measured, and the queries needed to expect 50 / 100 / 200 usable ones follow.

2.  THE EFFECT SIZE THAT WAS DETECTABLE. A retrospective note on what the
    observed pairwise F1 differences would have needed in order to be detected,
    reported as the sample size required for 80% power at alpha = 0.05 rather
    than as post-hoc "observed power", which is not informative.

A refusal-suppressing prompt is already implemented (--prompt direct, 67% -> 0%
refusal on a probe). The cost estimate below assumes it works at scale, so it is
a LOWER bound on the work required.

No API calls; reads the cached n = 160 run.
Writes results/h3_provenance/h3_power.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SCORED = Path("results/h3_provenance/h3_scored.csv")
OUT = Path("results/h3_provenance/h3_power.md")

CORPORA = {"multicare": "MultiCaRe (case reports)", "shuffled": "Shuffled MultiCaRe control",
           "pubmedqa": "PubMedQA (abstracts)", "mmedbench": "MMedBench (exam text)"}
TARGETS = (50, 100, 200)
#: Tokens per query for a four-condition run: 4 conditions x (evidence + prompt +
#: completion). Measured at ~1,150 tokens per condition in the cached run.
TOKENS_PER_QUERY = 4 * 1_150
GROQ_DAILY_TOKENS = 200_000      # per organisation, per model


def n_for_paired_t(d: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """Sample size for a paired comparison at the given effect size (normal approx.)."""
    if not np.isfinite(d) or d == 0:
        return float("inf")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float(np.ceil(((z_a + z_b) / abs(d)) ** 2))


def main() -> None:
    df = pd.read_csv(SCORED)
    n = len(df)

    # SCOREABLE, as the paper counts it: the answer asserts at least one lexicon
    # concept, so an F1 exists. This reproduces the paper's 63/61/70/66 and 13.
    scoreable = pd.DataFrame({c: df[f"{c}_f1"].notna() for c in CORPORA})
    per_condition = scoreable.mean()
    joint = scoreable.all(axis=1)
    joint_rate = float(joint.mean())
    independent_rate = float(np.prod(per_condition.to_numpy()))

    # ANSWERED AND SCOREABLE: additionally not flagged as a refusal. The gap
    # between the two is a validity problem in its own right -- most scoreable
    # answers are ALSO refusals, i.e. hedges that still name a concept
    # ("main confirm nahi kar sakta, lekin rash ho sakta hai"). Table 5's F1
    # comparison therefore rests largely on refusals rather than on answers.
    answered = pd.DataFrame({
        c: (~df[f"{c}_refusal"].astype(bool)) & df[f"{c}_f1"].notna() for c in CORPORA
    })
    refusal_overlap = pd.DataFrame({
        c: df[f"{c}_refusal"].astype(bool) & df[f"{c}_f1"].notna() for c in CORPORA
    })

    rows = []
    for a in CORPORA:
        for b in CORPORA:
            if a >= b:
                continue
            m = scoreable[a] & scoreable[b]
            x, y = df.loc[m, f"{a}_f1"], df.loc[m, f"{b}_f1"]
            if m.sum() < 3:
                continue
            diff = (x - y).to_numpy(float)
            d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
            rows.append({"pair": f"{a} vs {b}", "n_pairwise": int(m.sum()),
                         "mean_diff": float(diff.mean()), "cohens_dz": d,
                         "p": float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
                         "n_for_80pct_power": n_for_paired_t(d)})
    pw = pd.DataFrame(rows).sort_values("n_for_80pct_power")

    L = ["# H03: what a powered provenance experiment would require", "",
         f"Computed from the cached n = {n} run (gpt-oss-120b, four matched corpora). No API calls.", "",
         "## Why the usable sample collapsed", "",
         "| Evidence corpus | Refusal | Scoreable (asserts >= 1 concept) | of which ALSO flagged refusal | Answered *and* scoreable |",
         "|---|---:|---:|---:|---:|"]
    for c, label in CORPORA.items():
        L.append(f"| {label} | {df[f'{c}_refusal'].mean():.1%} | {int(scoreable[c].sum())} "
                 f"({per_condition[c]:.1%}) | {int(refusal_overlap[c].sum())} | "
                 f"{int(answered[c].sum())} ({answered[c].mean():.1%}) |")
    L += ["", "> **A validity problem the reviewers did not raise.** Most scoreable answers are also",
          "> flagged as refusals: hedged replies that decline to answer yet still name a concept",
          f"> (MultiCaRe: {int(refusal_overlap['multicare'].sum())} of {int(scoreable['multicare'].sum())}).",
          "> The paper's Table 5 mean F1 is therefore computed largely on refusals, not on answers.",
          f"> Requiring a genuine answer leaves only {int(answered.all(axis=1).sum())} queries usable",
          "> under all four conditions. Report the F1 column with this caveat or drop it."]
    L += ["", f"- Jointly scoreable under **all four** conditions: **{joint.sum()} of {n}** "
          f"({joint_rate:.1%}).",
          f"- If the four conditions were independent, the expected joint rate would be "
          f"{independent_rate:.1%}; the observed {joint_rate:.1%} is "
          f"{'higher' if joint_rate > independent_rate else 'lower'}, so refusals are "
          f"{'correlated across conditions (some queries are simply harder)' if joint_rate > independent_rate else 'close to independent'}.",
          "",
          "## Queries needed for a four-way omnibus", "",
          "| Target usable queries | Queries to run | Generation calls | Tokens (approx.) | Days at one Groq organisation's quota |",
          "|---:|---:|---:|---:|---:|"]
    for t in TARGETS:
        need = int(np.ceil(t / joint_rate)) if joint_rate > 0 else 0
        tokens = need * TOKENS_PER_QUERY
        L.append(f"| {t} | {need:,} | {need * 4:,} | {tokens:,} | {tokens / GROQ_DAILY_TOKENS:.1f} |")
    L += ["", f"At the observed joint rate, the submitted design would need "
          f"**{int(np.ceil(50 / joint_rate)):,} queries** for a modest 50 usable ones.",
          "",
          "The refusal-suppressing prompt (`--prompt direct`, 67% -> 0% refusal on a probe) is the",
          "lever that changes this: at a 90% per-condition scoreable rate the joint rate would be",
          f"~{0.9 ** 4:.0%}, needing only ~{int(np.ceil(50 / 0.9 ** 4))} queries for 50 usable ones.",
          "That is the design a follow-up should use.", "",
          "## Effect sizes and the sample each would need", "",
          "Pairwise F1 differences on the queries scoreable in BOTH conditions of each pair,",
          "with the n required to detect that effect at 80% power, alpha = 0.05 (two-sided).", "",
          "| Comparison | n available | Mean F1 difference | Cohen's dz | p | n for 80% power |",
          "|---|---:|---:|---:|---:|---:|"]
    for _, r in pw.iterrows():
        need = r.n_for_80pct_power
        L.append(f"| {r.pair} | {r.n_pairwise} | {r.mean_diff:+.4f} | {r.cohens_dz:+.3f} | "
                 f"{r.p:.3g} | {'—' if not np.isfinite(need) else f'{int(need):,}'} |")
    L += ["", "## What the paper should say", "",
          "- H03's primary outcome (answer quality) is **not adequately tested**: neither rejected",
          "  nor supported. Report it as exploratory and provisional.",
          "- The refusal effect (Cochran's Q) is a **secondary outcome**, on a different dependent",
          "  variable from the one H03 names.",
          f"- A properly powered replication needs the low-refusal prompt and roughly "
          f"{int(np.ceil(50 / 0.9 ** 4))}-{int(np.ceil(200 / 0.9 ** 4))} queries. At the refusal rates",
          f"  observed here the same design needs {int(np.ceil(50 / joint_rate)):,}-"
          f"{int(np.ceil(200 / joint_rate)):,} queries "
          f"({int(np.ceil(50 / joint_rate)) * TOKENS_PER_QUERY / 1e6:.1f}-"
          f"{int(np.ceil(200 / joint_rate)) * TOKENS_PER_QUERY / 1e6:.1f}M tokens, "
          f"{int(np.ceil(50 / joint_rate)) * TOKENS_PER_QUERY / GROQ_DAILY_TOKENS:.0f}-"
          f"{int(np.ceil(200 / joint_rate)) * TOKENS_PER_QUERY / GROQ_DAILY_TOKENS:.0f} days of one",
          "  organisation's quota), which is why it was not run.",
          f"- Detecting the largest observed pairwise effect (dz = "
          f"{pw.cohens_dz.abs().max():.2f}) at 80% power needs about "
          f"{int(pw.n_for_80pct_power.min()):,} jointly scoreable pairs.", ""]
    OUT.write_text("\n".join(L), encoding="utf-8")
    pw.to_csv(OUT.with_name("h3_power.csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
