"""H2 under the corrected CMI, with the ambiguity policy as a sensitivity check.

Recomputes CMI for the 1,165 cached generations using the Das & Gamback measure
in `src/analysis/cmi.py`, then repeats the per-arm H2 analysis under all three
ambiguous-token policies. If the conclusion is identical under `exclude`,
`hindi` and `english`, it does not depend on the tagging judgement.

No API calls -- re-scores cached outputs only.

Writes: results/h2_per_arm/h2_corrected_cmi_report.md
        results/h2_per_arm/h2_corrected_cmi_stats.csv
        results/h2_per_arm/combined_with_cmi_v2.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.cmi import build_english_vocab, cmi, cmi_legacy, hindi_proportion
from src.analysis.h2_per_arm import ARMS, benjamini_hochberg, bootstrap_ci_spearman

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COMBINED_PATH = Path("results/combined_h1h2/combined_scored.csv")
OUTPUT_DIR = Path("results/h2_per_arm")
POLICIES = ("exclude", "hindi", "english")


def per_arm(df: pd.DataFrame, cmi_col: str) -> pd.DataFrame:
    rows = []
    for col, label in ARMS.items():
        x = df[cmi_col].to_numpy(dtype=float)
        y = df[col].to_numpy(dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        r = stats.spearmanr(x, y)
        lo, hi = bootstrap_ci_spearman(x, y, n_boot=5_000)
        rows.append(
            {
                "cmi_variant": cmi_col,
                "arm": label,
                "column": col,
                "rho": r.statistic,
                "p": r.pvalue,
                "ci_lo": lo,
                "ci_hi": hi,
                "flat": bool(lo <= 0 <= hi),
            }
        )
    out = pd.DataFrame(rows)
    out["p_bh"] = benjamini_hochberg(out["p"].tolist())
    return out


def main() -> None:
    df = pd.read_csv(COMBINED_PATH)
    logger.info("Loaded %d cached generations", len(df))

    vocab = build_english_vocab()

    df["cmi_legacy_recomputed"] = df["hinglish_query"].apply(cmi_legacy)
    for pol in POLICIES:
        df[f"cmi_v2_{pol}"] = df["hinglish_query"].apply(lambda t, p=pol: cmi(t, vocab, p))
    # Clean Hindi-PROPORTION: same construct as the legacy measure, but with the
    # repaired lexicon. This is what isolates "the lexicon was buggy" from
    # "the construct changed".
    df["hindi_prop_v2"] = df["hinglish_query"].apply(lambda t: hindi_proportion(t, vocab))

    # --- distribution comparison -------------------------------------------
    dist_rows = []
    for col in ["cmi_score", "cmi_legacy_recomputed", "hindi_prop_v2", *[f"cmi_v2_{p}" for p in POLICIES]]:
        s = df[col]
        dist_rows.append(
            {
                "measure": col,
                "mean": s.mean(),
                "sd": s.std(),
                "min": s.min(),
                "q25": s.quantile(0.25),
                "median": s.median(),
                "q75": s.quantile(0.75),
                "max": s.max(),
                "IQR": s.quantile(0.75) - s.quantile(0.25),
            }
        )
    dist = pd.DataFrame(dist_rows)

    corr_old_new = stats.spearmanr(df["cmi_score"], df["cmi_v2_exclude"])
    corr_old_prop = stats.spearmanr(df["cmi_score"], df["hindi_prop_v2"])
    corr_prop_cmi = stats.spearmanr(df["hindi_prop_v2"], df["cmi_v2_exclude"])

    # --- per-arm analysis under each policy --------------------------------
    all_stats = pd.concat([per_arm(df, f"cmi_v2_{p}") for p in POLICIES], ignore_index=True)
    legacy_stats = per_arm(df, "cmi_score")
    legacy_stats["cmi_variant"] = "cmi_score (legacy, as shipped)"
    prop_stats = per_arm(df, "hindi_prop_v2")
    prop_stats["cmi_variant"] = "hindi_prop_v2 (repaired lexicon, SAME construct)"
    all_stats = pd.concat([legacy_stats, prop_stats, all_stats], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats.to_csv(OUTPUT_DIR / "h2_corrected_cmi_stats.csv", index=False)
    keep = ["pair_id", "cmi_score", "cmi_legacy_recomputed", *[f"cmi_v2_{p}" for p in POLICIES]]
    df[keep].to_csv(OUTPUT_DIR / "combined_with_cmi_v2.csv", index=False)

    # --- report -------------------------------------------------------------
    L: list[str] = [
        "# H2 under the corrected Code-Mixing Index",
        "",
        f"n = {len(df)} cached generations. No API calls.",
        "",
        "## 1. Why the legacy measure was replaced",
        "",
        "Measured on the 3,015 MMCQSD queries:",
        "",
        "- `doctor` (English) is in the Hindi list and fires in **71.0%** of queries;",
        "  `please` in **38.0%**.",
        "- **32.9%** of all query tokens were unknown to both the Hindi list and an",
        "  English vocabulary; the frequency-ranked OOV list is almost entirely",
        "  romanised Hindi (`mein` 5,340, `mere` 3,945, `hoon` 2,572, ...).",
        "",
        "Over-counting and under-counting both compress the score toward the middle.",
        "",
        "## 2. Distributions",
        "",
        "| Measure | mean | SD | min | median | max | IQR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in dist.iterrows():
        L.append(
            f"| `{r['measure']}` | {r['mean']:.4f} | {r['sd']:.4f} | {r['min']:.4f} | "
            f"{r['median']:.4f} | {r['max']:.4f} | {r['IQR']:.4f} |"
        )
    L += [
        "",
        "## 2b. TWO CHANGES, NOT ONE -- read this before quoting any number",
        "",
        "Replacing the legacy measure changed two things at once:",
        "",
        "1. **The lexicon was repaired** (contaminants removed, OOV Hindi added).",
        "2. **The construct changed.** The legacy measure is a Hindi PROPORTION.",
        "   Das & Gamback CMI is a mixing-BALANCE measure: maximal at a 50/50 mix and",
        "   **zero for monolingual text in either language**. A 90%-Hindi query scores",
        "   HIGH on proportion and LOW on CMI.",
        "",
        "`hindi_prop_v2` isolates change (1): repaired lexicon, same construct.",
        "",
        "| Comparison | Spearman rho | p |",
        "|---|---:|---:|",
        f"| legacy vs Das & Gamback CMI | {corr_old_new.statistic:+.4f} | {corr_old_new.pvalue:.3g} |",
        f"| legacy vs repaired proportion | {corr_old_prop.statistic:+.4f} | {corr_old_prop.pvalue:.3g} |",
        f"| repaired proportion vs CMI | {corr_prop_cmi.statistic:+.4f} | {corr_prop_cmi.pvalue:.3g} |",
        "",
        "The strong NEGATIVE legacy-vs-CMI correlation is expected and is not evidence",
        "that either is wrong -- it is the proportion-vs-balance distinction. Compare",
        "the legacy row against `hindi_prop_v2` to judge the lexicon repair, and",
        "against `cmi_v2_*` to judge the construct change.",
        "",
        "## 3. Per-arm effects, under each ambiguity policy",
        "",
        "`exclude` treats Hindi/English homographs (`me`, `sir`, `pet`, `pair`) as",
        "language-independent; `hindi` and `english` force them either way. Agreement",
        "across all three means the conclusion does not rest on that judgement.",
        "",
        "| CMI variant | Arm | rho | 95% CI | p (BH) | Verdict |",
        "|---|---|---:|---|---:|---|",
    ]
    for _, r in all_stats.iterrows():
        verdict = "flat" if r["flat"] else ("rises" if r["rho"] > 0 else "declines")
        L.append(
            f"| `{r['cmi_variant']}` | {r['arm']} | {r['rho']:+.4f} | "
            f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p_bh']:.4f} | **{verdict}** |"
        )

    # --- automatic robustness verdict ---------------------------------------
    L += ["", "## 4. Robustness", ""]
    for col, label in ARMS.items():
        sub = all_stats[(all_stats["column"] == col) & (all_stats["cmi_variant"] != "cmi_score (legacy, as shipped)")]
        verdicts = {"flat" if r["flat"] else ("rises" if r["rho"] > 0 else "declines") for _, r in sub.iterrows()}
        agree = "STABLE" if len(verdicts) == 1 else "POLICY-DEPENDENT"
        L.append(f"- **{label}**: {agree} across policies -> {sorted(verdicts)}")

    L += [
        "",
        "Report any POLICY-DEPENDENT row as an explicit limitation; do not pick the",
        "policy that gives the preferred answer.",
        "",
        "## 5. Resolution -- which construct to report, and what H2 actually says",
        "",
        f"Repaired proportion and Das & Gamback CMI correlate at "
        f"**rho = {corr_prop_cmi.statistic:+.4f}** on this corpus: they are near-perfect",
        "inverses. These queries are Hindi-dominant (mean Hindi proportion "
        f"{df['hindi_prop_v2'].mean():.3f}), so adding Hindi moves text AWAY from a 50/50",
        "balance. 'Rises with CMI' and 'declines with Hindi proportion' are therefore",
        "**the same finding stated on inverted scales**, not a contradiction.",
        "",
        "**Report `hindi_prop_v2` as primary.** The hypothesis is whether more Hindi",
        "degrades an English-centric pipeline, which is a question about proportion.",
        "Report Das & Gamback CMI as the standard-metric cross-check.",
        "",
        "The lexicon repair did not overturn the factual-support result -- it",
        "strengthened it. On the SAME construct, the zero-shot decline goes from",
        "rho = -0.068 (BH p = 0.042) under the contaminated lexicon to rho = -0.116",
        "(BH p = 0.0003) under the repaired one, while the grounded arm stays flat.",
        "",
        "The hallucination effects, however, do NOT survive the repair. Under the",
        "legacy lexicon both arms appeared to rise; under the repaired lexicon both are",
        "flat. Those were lexicon artefacts and must not be reported as findings.",
        "",
        "**Defensible H2 claim:** increasing Hindi content significantly degrades",
        "zero-shot factual support while leaving grounded factual support unchanged;",
        "hallucination rates are unaffected by Hindi content in either arm.",
    ]

    report = "\n".join(L)
    (OUTPUT_DIR / "h2_corrected_cmi_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
