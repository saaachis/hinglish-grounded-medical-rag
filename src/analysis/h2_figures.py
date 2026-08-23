"""Regenerate the H2 code-mixing figures on the REPAIRED measure.

Every CMI figure in `research-poster-work/` is invalid. Two reasons:

1. They use the legacy `cmi_score`, whose Hindi lexicon contains the English
   words `doctor` (fires in 68.2% of queries) and `please` (35.7%), and misses
   32.9% of romanised-Hindi tokens entirely. Both errors compress the measure
   toward the middle -- its SD was only 0.075.
2. They bucket into tertiles. Bucketing a continuous predictor discards power:
   the zero-shot factual effect is significant continuously (p = 7.7e-05) but
   NOT across tertiles (p = 0.127). The tertile framing manufactured the null
   that was previously reported as "robustness".

These figures use `hindi_prop_v2` (repaired lexicon, same construct as the
legacy measure so the comparison is apples-to-apples) and plot the continuous
relationship with a fitted slope, not buckets.

Outputs 300-DPI PNG and vector PDF, as the ICCSDI template requires.
Writes results/h2_figures/. No API calls.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.cmi import build_english_vocab, hindi_proportion

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CACHED = Path("results/combined_h1h2/combined_scored.csv")
OUT = Path("results/h2_figures")

GROUNDED = "#eb6834"   # orange
ZERO = "#2a78d6"       # blue
INK = "#131C28"
MUTED = "#71808F"
GRID = "#CBD3DC"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _fit(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 42):
    """Least-squares fit with a bootstrap CI band.

    The band is not decoration. Hindi proportion is tightly clustered around
    0.6-0.8 with a thin tail down to 0, so those few low-x rows carry enormous
    leverage on an OLS slope. Without a band the grounded panel draws a visibly
    rising line while Spearman rho is -0.001 -- a plot that contradicts its own
    statistic. The band shows that slope is not distinguishable from flat.
    """
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    xs = np.linspace(x.min(), x.max(), 100)
    sl, ic = np.polyfit(x, y, 1)

    rng = np.random.default_rng(seed)
    n = len(x)
    curves = np.empty((n_boot, len(xs)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        s, i0 = np.polyfit(x[idx], y[idx], 1)
        curves[b] = s * xs + i0
    lo, hi = np.quantile(curves, [0.025, 0.975], axis=0)

    rho, p = stats.spearmanr(x, y)
    return xs, sl * xs + ic, lo, hi, float(rho), float(p)


def _binned_means(x, y, n_bins: int = 8, n_boot: int = 2000, seed: int = 42):
    """Mean of y per quantile-bin of x, with bootstrap CIs.

    Quantile bins (not equal-width) so every point carries the same number of
    observations -- with x concentrated in 0.5-0.9, equal-width bins would put
    almost nothing in the left half and produce meaningless error bars.
    """
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    rng = np.random.default_rng(seed)

    bx, by, blo, bhi = [], [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 10:
            continue
        yy = y[m]
        boots = yy[rng.integers(0, len(yy), (n_boot, len(yy)))].mean(axis=1)
        bx.append(x[m].mean())
        by.append(yy.mean())
        lo, hi = np.quantile(boots, [0.025, 0.975])
        blo.append(lo)
        bhi.append(hi)
    return np.array(bx), np.array(by), np.array(blo), np.array(bhi)


def figure_dose_response(df: pd.DataFrame) -> None:
    """The headline H2 figure: grounded flat, zero-shot declining."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    x = df["hindi_prop_v2"].to_numpy()

    for ax, (col, colour, label) in zip(axes, [
        ("grounded_factual", GROUNDED, "Grounded (RAG)"),
        ("zero_factual", ZERO, "Zero-shot"),
    ]):
        y = df[col].to_numpy()
        ax.scatter(x, y, s=5, alpha=0.10, color=colour, linewidths=0, rasterized=True)

        # Binned means, not a global OLS line. 95% of the mass sits between
        # x=0.5 and x=0.9; an OLS line extrapolated across the empty left half
        # is dragged by a handful of high-leverage rows and draws a visibly
        # RISING grounded trend while Spearman rho is -0.001. Deciles of the
        # observed x show the relationship where data actually exists.
        bx, by, blo, bhi = _binned_means(x, y)
        ax.fill_between(bx, blo, bhi, color=colour, alpha=0.20, linewidth=0)
        ax.plot(bx, by, color=colour, lw=2, marker="o", ms=4,
                markeredgecolor="white", markeredgewidth=0.8)
        _, _, _, _, rho, p = _fit(x, y)
        sig = "n.s." if p >= 0.05 else (f"p = {p:.1e}" if p < 1e-3 else f"p = {p:.3f}")
        ax.set_title(label, fontsize=10, color=INK, pad=7, loc="left")
        ax.text(0.03, 0.06, f"ρ = {rho:+.3f}   {sig}", transform=ax.transAxes,
                fontsize=8.5, color=colour, fontweight="bold")
        ax.set_xlabel("Hindi proportion (repaired lexicon)")
        ax.set_ylim(-0.03, 1.03)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("Factual support")
    fig.suptitle("Grounding absorbs the effect of code-mixing on factual support",
                 fontsize=11, x=0.012, ha="left", y=1.0, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"h2_dose_response.{ext}", bbox_inches="tight")
    plt.close(fig)


def figure_effect_comparison(df: pd.DataFrame) -> None:
    """Forest plot: legacy vs repaired measure, all four outcomes."""
    outcomes = [
        ("grounded_factual", "Grounded factual", GROUNDED),
        ("zero_factual", "Zero-shot factual", ZERO),
        ("grounded_hallucination", "Grounded hallucination", GROUNDED),
        ("zero_hallucination", "Zero-shot hallucination", ZERO),
    ]
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    for i, (col, label, colour) in enumerate(outcomes):
        for j, (xcol, marker, alpha, name) in enumerate(
                [("cmi_score", "o", 0.35, "legacy"), ("hindi_prop_v2", "D", 1.0, "repaired")]):
            sub = df[[xcol, col]].dropna()
            rho, p = stats.spearmanr(sub[xcol], sub[col])
            boots = [stats.spearmanr(s[xcol], s[col])[0]
                     for s in (sub.sample(len(sub), replace=True, random_state=int(r))
                               for r in rng.integers(0, 1e6, 400))]
            lo, hi = np.quantile(boots, [0.025, 0.975])
            ypos = i + (0.18 if j else -0.18)
            ax.plot([lo, hi], [ypos, ypos], color=colour, lw=1.6, alpha=alpha, solid_capstyle="butt")
            ax.plot(rho, ypos, marker, color=colour, ms=5.5, alpha=alpha,
                    markeredgecolor="white", markeredgewidth=0.7,
                    label=name if i == 0 else None)

    ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(range(len(outcomes)))
    ax.set_yticklabels([o[1] for o in outcomes])
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ with code-mixing (95% bootstrap CI)")
    ax.set_title("Repairing the lexicon removed both hallucination effects",
                 fontsize=11, loc="left", color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", title="measure",
              title_fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"h2_effect_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(CACHED)
    vocab = build_english_vocab()
    df["hindi_prop_v2"] = df["hinglish_query"].astype(str).apply(
        lambda t: hindi_proportion(t, vocab))
    logger.info("hindi_prop_v2: mean %.4f SD %.4f (legacy SD %.4f)",
                df.hindi_prop_v2.mean(), df.hindi_prop_v2.std(), df.cmi_score.std())

    OUT.mkdir(parents=True, exist_ok=True)
    figure_dose_response(df)
    figure_effect_comparison(df)
    df[["pair_id", "cmi_score", "hindi_prop_v2"]].to_csv(OUT / "cmi_v2_values.csv", index=False)

    logger.info("Wrote %s", ", ".join(sorted(p.name for p in OUT.iterdir())))
    print("\nSUPERSEDED by these figures (legacy CMI, tertile framing):")
    for f in ("04_h2_cmi_levels.png", "06_cmi_scatter.png",
              "12_h2_grounded_factual_by_cmi.png"):
        print(f"  research-poster-work/poster charts 1/{f}")


if __name__ == "__main__":
    main()
