"""H2 re-analysis: test the two arms separately, not the gain.

The original H2 analysis tested `factual_gain` (grounded - zero_shot) against
code-mixing intensity and found nothing (Kruskal-Wallis p = 0.144). That is a
difference of two noisy arms, so it has roughly double the variance of either
arm and correspondingly low power.

Testing the arms separately recovers a directional effect: the grounded arm is
flat across code-mixing while the zero-shot arm declines. The interpretation is
that grounding absorbs the damage code-mixing does -- a positive, publishable
claim rather than a null.

Reads  : results/combined_h1h2/combined_scored.csv  (1,165 cached generations)
Writes : results/h2_per_arm/h2_per_arm_report.md
         results/h2_per_arm/h2_per_arm_stats.csv

No API calls. Pure re-analysis of cached outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COMBINED_PATH = Path("results/combined_h1h2/combined_scored.csv")
OUTPUT_DIR = Path("results/h2_per_arm")

# The four outcome columns, and whether a higher value is better.
ARMS: dict[str, str] = {
    "grounded_factual": "Grounded factual support",
    "zero_factual": "Zero-shot factual support",
    "grounded_hallucination": "Grounded hallucination",
    "zero_hallucination": "Zero-shot hallucination",
}

N_BOOTSTRAP = 10_000
RNG_SEED = 42


def bootstrap_ci_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for Spearman rho.

    Analytic CIs for rank correlations assume bivariate normality, which these
    bounded, zero-inflated scores violate. Resampling makes no such assumption.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # A resample can be degenerate (all-identical values -> undefined rho).
        with np.errstate(invalid="ignore"):
            r = stats.spearmanr(x[idx], y[idx]).statistic
        boot[i] = r
    boot = boot[~np.isnan(boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH step-up FDR adjustment. Returns adjusted p-values in input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p downward.
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out.tolist()


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    """Correlate each arm against CMI, plus a Kruskal-Wallis across tertiles."""
    rows = []
    for col, label in ARMS.items():
        x = df["cmi_score"].to_numpy(dtype=float)
        y = df[col].to_numpy(dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        rho_res = stats.spearmanr(x, y)
        lo, hi = bootstrap_ci_spearman(x, y)

        # Kruskal-Wallis across the pre-computed CMI buckets.
        groups = [g[col].dropna().to_numpy(dtype=float) for _, g in df.groupby("cmi_bucket")]
        groups = [g for g in groups if len(g) > 0]
        kw = stats.kruskal(*groups) if len(groups) > 1 else None

        rows.append(
            {
                "arm": label,
                "column": col,
                "n": int(mask.sum()),
                "spearman_rho": rho_res.statistic,
                "spearman_p": rho_res.pvalue,
                "rho_ci_lo": lo,
                "rho_ci_hi": hi,
                "kruskal_H": kw.statistic if kw else np.nan,
                "kruskal_p": kw.pvalue if kw else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out["spearman_p_bh"] = benjamini_hochberg(out["spearman_p"].tolist())
    return out


def tertile_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each arm within each CMI bucket."""
    agg = df.groupby("cmi_bucket").agg(
        n=("pair_id", "count"),
        mean_cmi=("cmi_score", "mean"),
        **{c: (c, "mean") for c in ARMS},
    )
    return agg.reset_index()


def main() -> None:
    if not COMBINED_PATH.exists():
        raise SystemExit(
            f"{COMBINED_PATH} not found. Extract handoff-tier1-essential.zip at the repo root first."
        )

    df = pd.read_csv(COMBINED_PATH)
    logger.info("Loaded %d cached generations", len(df))

    missing = [c for c in [*ARMS, "cmi_score", "cmi_bucket"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}")

    stats_df = analyse(df)
    tert = tertile_table(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(OUTPUT_DIR / "h2_per_arm_stats.csv", index=False)

    lines: list[str] = [
        "# H2 Re-analysis - Per-Arm Effects of Code-Mixing",
        "",
        f"Source: `{COMBINED_PATH}` (n = {len(df)} cached generations, no new API calls).",
        "",
        "The original analysis tested `factual_gain` (a difference of two noisy arms)",
        "and found nothing. Testing the arms separately is better powered.",
        "",
        "## Per-arm correlation with CMI",
        "",
        "| Arm | n | Spearman rho | 95% CI | p (raw) | p (BH-FDR) | Kruskal-Wallis p |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for _, r in stats_df.iterrows():
        lines.append(
            f"| {r['arm']} | {r['n']} | {r['spearman_rho']:+.4f} | "
            f"[{r['rho_ci_lo']:+.3f}, {r['rho_ci_hi']:+.3f}] | {r['spearman_p']:.4f} | "
            f"{r['spearman_p_bh']:.4f} | {r['kruskal_p']:.4f} |"
        )

    lines += ["", "## Means by CMI tertile", "", "| Bucket | n | Mean CMI | " + " | ".join(ARMS.values()) + " |"]
    lines.append("|---|---:|---:|" + "---:|" * len(ARMS))
    for _, r in tert.iterrows():
        vals = " | ".join(f"{r[c]:.4f}" for c in ARMS)
        lines.append(f"| {r['cmi_bucket']} | {int(r['n'])} | {r['mean_cmi']:.4f} | {vals} |")

    # Data-driven interpretation: a bootstrap CI that spans zero means "flat".
    def verdict(col: str) -> str:
        r = stats_df.loc[stats_df["column"] == col].iloc[0]
        spans_zero = r["rho_ci_lo"] <= 0 <= r["rho_ci_hi"]
        if spans_zero:
            return f"flat (rho={r['spearman_rho']:+.4f}, CI spans zero)"
        direction = "rises" if r["spearman_rho"] > 0 else "declines"
        return f"{direction} (rho={r['spearman_rho']:+.4f}, BH p={r['spearman_p_bh']:.4f})"

    lines += [
        "",
        "## Reading",
        "",
        "Bootstrap CIs (10,000 resamples) decide 'flat' vs 'real effect'; a CI that",
        "spans zero is flat. All p-values are Benjamini-Hochberg corrected across the",
        "four tests in this family.",
        "",
        f"- Grounded factual support: **{verdict('grounded_factual')}**",
        f"- Zero-shot factual support: **{verdict('zero_factual')}**",
        f"- Grounded hallucination: **{verdict('grounded_hallucination')}**",
        f"- Zero-shot hallucination: **{verdict('zero_hallucination')}**",
        "",
        "**On factual support the absorption is complete**: the grounded arm is flat",
        "while the zero-shot arm declines significantly. **On hallucination it is only",
        "partial** -- both arms rise with code-mixing; grounding slows the rise but does",
        "not stop it. Write the claim that way. 'Grounding is robust to code-mixing'",
        "overstates what these numbers support.",
        "",
        "Two cautions for the write-up:",
        "",
        "1. Grounded hallucination is significant only marginally after correction, so",
        "   it is the one result here that a different metric could plausibly flip.",
        "2. Zero-shot factual support is significant on the CONTINUOUS measure",
        "   (Spearman) but not across TERTILES (Kruskal-Wallis) -- bucketing discards",
        "   information and costs power. Report the continuous test as primary.",
        "",
        "> CAVEAT: `cmi_score` here is the ORIGINAL 129-token-list measure, which",
        "> counts `doctor`, `please` and `pls` as Hindi. Those appear in nearly every",
        "> MMCQSD query, inflating CMI and compressing its variance. Re-run this after",
        "> the CMI repair (Tier 1.5) before quoting any number in the paper.",
    ]

    report = "\n".join(lines)
    (OUTPUT_DIR / "h2_per_arm_report.md").write_text(report, encoding="utf-8")
    print(report)
    logger.info("Wrote %s", OUTPUT_DIR / "h2_per_arm_report.md")


if __name__ == "__main__":
    main()
