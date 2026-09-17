"""H02 under its PRE-SPECIFIED test (ICCSDI review, Reviewer 3 point 3).

Reviewer 3: "H02 is formulated around the grounded-versus-ungrounded difference,
but tested through separate arm correlations; clarify alignment between
hypothesis and statistical test."

The reviewer is right, and the repository history says what the planned test was.

Two written versions of H2 exist, and both are tested here:

* 9 Feb 2026, README at commit 0917322 (before any analysis):
      H2  "Increasing code-mixing level does not significantly degrade grounded
           model performance"             test: "Two-way ANOVA / Kruskal-Wallis"
* Research proposal, and the submitted paper:
      H02 "the intensity of code-mixing has no effect on the factual difference
           between grounded and ungrounded"

A two-way design (arm x code-mixing level) with arm varying WITHIN query has one
interaction term, and that interaction is exactly whether the paired difference
    gain = grounded - zero_shot
changes across code-mixing levels. With two within-query arms the interaction F
equals a one-way ANOVA F on `gain` across levels, and Kruskal-Wallis on `gain`
is its rank-based version. That is the PRIMARY test of H02 as the paper words it.

The Feb README wording is about the GROUNDED arm alone, so its test is the
code-mixing main effect within that arm. It is reported as the second
pre-specified reading.

Everything else -- the continuous Spearman versions, the zero-shot arm, other
code-mixing measures -- is reported as exploratory.

Code-mixing levels are tertiles of `hindi_prop_v2` (repaired Hindi-word list, the
measure the paper reports). The originally shipped measure is a sensitivity check.

Equivalence: a non-significant test is not evidence of no effect. A descriptive
equivalence check is added with a smallest effect of interest of |rho| = 0.10,
chosen AFTER the data were seen and labelled as such: the effect is called
"equivalent to zero" only if the 90% bootstrap CI for rho lies inside +/-0.10
(the two-one-sided-tests logic at alpha = 0.05).

No API calls. n = 1,165 cached generations.
Writes results/h2_per_arm/h2_prespecified_test.{md,csv}.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.cmi import build_english_vocab, hindi_proportion
from src.analysis.h2_per_arm import benjamini_hochberg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COMBINED = Path("results/combined_h1h2/combined_scored.csv")
OUT = Path("results/h2_per_arm")
SESOI = 0.10
N_BOOT = 10_000
SEED = 42


def boot_spearman(x: np.ndarray, y: np.ndarray, level: float) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(x)
    b = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        with np.errstate(invalid="ignore"):
            b[i] = stats.spearmanr(x[idx], y[idx]).statistic
    b = b[~np.isnan(b)]
    a = (1 - level) / 2 * 100
    return float(np.percentile(b, a)), float(np.percentile(b, 100 - a))


def level_tests(y: np.ndarray, levels: np.ndarray) -> dict[str, float]:
    groups = [y[levels == g] for g in sorted(np.unique(levels))]
    return {"anova_F": float(stats.f_oneway(*groups).statistic),
            "anova_p": float(stats.f_oneway(*groups).pvalue),
            "kruskal_H": float(stats.kruskal(*groups).statistic),
            "kruskal_p": float(stats.kruskal(*groups).pvalue)}


def continuous(y: np.ndarray, x: np.ndarray) -> dict[str, float | str]:
    r = stats.spearmanr(x, y)
    lo95, hi95 = boot_spearman(x, y, 0.95)
    lo90, hi90 = boot_spearman(x, y, 0.90)
    if -SESOI < lo90 and hi90 < SESOI:
        verdict = f"equivalent to zero (90% CI inside +/-{SESOI})"
    elif lo95 > 0 or hi95 < 0:
        verdict = "non-zero"
    else:
        verdict = "inconclusive (neither non-zero nor equivalent)"
    return {"rho": float(r.statistic), "rho_p": float(r.pvalue), "ci95_lo": lo95, "ci95_hi": hi95,
            "ci90_lo": lo90, "ci90_hi": hi90, "equivalence": verdict}


def main() -> None:
    df = pd.read_csv(COMBINED)
    vocab = build_english_vocab()
    df["hindi_prop_v2"] = df.hinglish_query.apply(lambda t: hindi_proportion(t, vocab))
    df["gain"] = df.grounded_factual - df.zero_factual
    df = df.dropna(subset=["gain", "hindi_prop_v2", "cmi_score"]).reset_index(drop=True)
    n = len(df)
    df["level"] = pd.qcut(df.hindi_prop_v2.rank(method="first"), 3, labels=["low", "medium", "high"])
    df["level_legacy"] = pd.qcut(df.cmi_score.rank(method="first"), 3, labels=["low", "medium", "high"])

    specs = [
        # (label, role, outcome column, level column, continuous x column)
        ("H02 as worded in the paper: interaction (gain across levels)", "PRIMARY (pre-specified)",
         "gain", "level", "hindi_prop_v2"),
        ("H2 as worded 9 Feb: grounded arm across levels", "pre-specified (README wording)",
         "grounded_factual", "level", "hindi_prop_v2"),
        ("Zero-shot arm across levels", "exploratory decomposition",
         "zero_factual", "level", "hindi_prop_v2"),
        ("Interaction, originally shipped code-mixing measure", "sensitivity",
         "gain", "level_legacy", "cmi_score"),
    ]
    rows = []
    for label, role, ycol, lcol, xcol in specs:
        y = df[ycol].to_numpy(float)
        rows.append({"test": label, "role": role, "outcome": ycol, "levels": lcol, "n": n,
                     **level_tests(y, df[lcol].to_numpy()),
                     **continuous(y, df[xcol].to_numpy(float))})
    res = pd.DataFrame(rows)
    # BH only across the two PRE-SPECIFIED readings; exploratory and sensitivity rows
    # are not part of the confirmatory family and are left uncorrected.
    pre = res.role.str.startswith(("PRIMARY", "pre-specified"))
    res["kruskal_p_bh"] = np.nan
    res.loc[pre, "kruskal_p_bh"] = benjamini_hochberg(res.loc[pre, "kruskal_p"].tolist())

    means = df.groupby("level", observed=True).agg(
        n=("gain", "size"), hindi_prop=("hindi_prop_v2", "mean"),
        grounded=("grounded_factual", "mean"), zero_shot=("zero_factual", "mean"),
        gain=("gain", "mean")).reset_index()

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "h2_prespecified_test.csv", index=False)
    means.to_csv(OUT / "h2_prespecified_tertile_means.csv", index=False)
    write_report(res, means, n)


def write_report(res: pd.DataFrame, means: pd.DataFrame, n: int) -> None:
    p = res.iloc[0]
    L = ["# H02 under its pre-specified test", "",
         f"n = {n} cached generations (llama-3.1-8b-instant, oracle evidence). No API calls.",
         "Code-mixing levels = tertiles of `hindi_prop_v2` (repaired Hindi-word list).", "",
         "## Why this exists", "",
         "Reviewer 3: H02 concerns the grounded-minus-ungrounded difference but the paper tested",
         "each arm separately. The planned test (README, commit 0917322, 9 Feb 2026) was a",
         "two-way ANOVA / Kruskal-Wallis. With arm varying within query, the arm x level",
         "interaction is the test of whether the paired gain changes across levels.", "",
         "## Results", "",
         "| Test | Role | ANOVA F (p) | Kruskal-Wallis H (p) | KW BH p (pre-specified pair) | Spearman rho [95% CI], p | Equivalence (SESOI 0.10, post hoc) |",
         "|---|---|---|---|---:|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r.test} | {r.role} | {r.anova_F:.2f} ({r.anova_p:.3g}) | "
                 f"{r.kruskal_H:.2f} ({r.kruskal_p:.3g}) | "
                 f"{'—' if np.isnan(r.kruskal_p_bh) else format(r.kruskal_p_bh, '.3g')} | "
                 f"{r.rho:+.3f} [{r.ci95_lo:+.3f}, {r.ci95_hi:+.3f}], p = {r.rho_p:.3g} | {r.equivalence} |")
    L += ["", "## Means by code-mixing tertile", "",
          "| Level | n | Hindi proportion | Grounded | Zero-shot | Gain |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in means.iterrows():
        L.append(f"| {r.level} | {r.n} | {r.hindi_prop:.3f} | {r.grounded:.4f} | {r.zero_shot:.4f} | {r.gain:+.4f} |")
    g = res.iloc[1]
    lo_gain, hi_gain = means.gain.iloc[0], means.gain.iloc[-1]
    L += ["", "## Reading", "",
          "The two pre-specified tertile tests DISAGREE, so neither is reported alone:",
          f"- **Interaction (H02 as worded in the paper):** ANOVA F = {p.anova_F:.2f}, p = {p.anova_p:.3g}; "
          f"Kruskal-Wallis H = {p.kruskal_H:.2f}, p = {p.kruskal_p:.3g}. The plan named both. The outcome is "
          "bounded and non-normal, which favours the rank test, but choosing the test after seeing both "
          "p-values is exactly what must not be done. **Evidence against H02 is borderline under the "
          "pre-specified tertile tests.**",
          f"- **Continuous version (more power, not pre-specified):** rho = {p.rho:+.3f} "
          f"[{p.ci95_lo:+.3f}, {p.ci95_hi:+.3f}], p = {p.rho_p:.3g}. The grounding gain RISES with Hindi "
          f"proportion, from {lo_gain:+.4f} in the lowest tertile to {hi_gain:+.4f} in the highest.",
          f"- **Why it rises:** the grounded arm is flat -- statistically equivalent to zero effect "
          f"(rho = {g.rho:+.3f}, 90% CI [{g.ci90_lo:+.3f}, {g.ci90_hi:+.3f}] inside +/-{SESOI}) -- while the "
          "zero-shot arm degrades. The gain grows because the baseline falls, not because grounding improves.",
          "- **Grounded arm alone (H2 as worded 9 Feb):** not rejected, and equivalent to flat.",
          f"- **Sensitivity:** under the originally shipped code-mixing measure the interaction is "
          f"non-significant (KW p = {res.iloc[3].kruskal_p:.3g}). The paper's statement that the difference "
          "test is non-significant reflects that measure, not the repaired one, and must be updated.",
          "",
          "**Defensible H02 statement:** grounded factual support is stable across code-mixing intensity "
          "(equivalent to flat); the grounding gain increases with Hindi proportion on the continuous "
          "measure, with borderline support under the pre-specified tertile tests, because ungrounded "
          "performance degrades. The per-arm results are the decomposition of this test, not a replacement "
          "for it.", ""]
    (OUT / "h2_prespecified_test.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
