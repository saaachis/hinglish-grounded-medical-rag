"""Generate all research poster visualizations.

Poster theme (MS Word "Blue Green" + custom picks):
  - Primary teal:  #588894  (RGB 88, 136, 148)
  - Light panel:   #E6EEF0  (RGB 230, 238, 240)
  - Plus darker / lighter teal-greens from the same family for series contrast.

Run from repository root:
  python research-poster-work/generate_plots.py

Outputs high-resolution PNGs (300 dpi) in research-poster-work/.
H1 violin distributions: 02_h1_distributions.png (stacked); 02_h1_distributions_horizontal.png (side by side);
  02_h1_distributions_horizontal_poster_cm.png (6.6 cm × 4.1 cm, tight margins for Word).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

# ──────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).parent
DPI = 300

# Published totals (project-complete-summary.md) — captions / fallback text only
PUBLISHED_N = 1165

# ──────────────────────────────────────────────
#  BLUE-GREEN POSTER THEME
# ──────────────────────────────────────────────
THEME = {
    "teal": "#588894",
    "teal_dark": "#456F7A",
    "teal_darker": "#3A5C66",
    "navy_text": "#2F4858",
    "slate": "#5C7A82",
    "seafoam": "#8FB8A8",
    "mint": "#B8D4CE",
    "panel": "#E6EEF0",
    "figure_bg": "#FFFFFF",
    "grid": "#C5D3D6",
    # Series (zero-shot vs grounded)
    "zero_shot": "#8AA7B0",
    "zero_shot_edge": "#5C7A82",
    "grounded": "#2C5F6B",
    "grounded_light": "#588894",
    # CMI tertiles (teal gradient)
    "cm_low": "#6B9EAE",
    "cm_med": "#588894",
    "cm_high": "#3D5C66",
    # Aliases for older keys in code
    "primary": "#588894",
    "secondary": "#5C7A82",
    "accent": "#456F7A",
    "neutral": "#5C7A82",
    "bg": "#E6EEF0",
    "text": "#2F4858",
    "light_primary": "#B8D4CE",
    "light_secondary": "#D8E4E8",
    "light_accent": "#C5D9D4",
}

FONT_FAMILY = "sans-serif"
TITLE_SIZE = 10
LABEL_SIZE = 8.5
TICK_SIZE = 7.5
LEGEND_SIZE = 7
ANNO_SIZE = 6.5


def apply_theme():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.facecolor": THEME["panel"],
        "figure.facecolor": THEME["figure_bg"],
        "axes.edgecolor": THEME["grid"],
        "axes.labelcolor": THEME["text"],
        "xtick.color": THEME["text"],
        "ytick.color": THEME["text"],
        "text.color": THEME["text"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.fontsize": LEGEND_SIZE,
    })


def save(fig, name: str, **kwargs) -> None:
    opts = dict(
        dpi=DPI,
        bbox_inches="tight",
        facecolor=THEME["figure_bg"],
        pad_inches=0.02,
    )
    opts.update(kwargs)
    fig.savefig(OUT_DIR / f"{name}.png", **opts)
    plt.close(fig)
    print(f"  Saved {name}.png")


def _resolve_csv(rel: str) -> Path:
    return REPO_ROOT / rel


def load_data() -> pd.DataFrame:
    path = _resolve_csv("results/combined_h1h2/combined_scored.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run evaluation / combine scripts from repo root, then re-run."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} clean pairs (summary target n={PUBLISHED_N})")
    return df


def compute_h1_pvalues(df: pd.DataFrame) -> tuple[float, float]:
    fg = df["grounded_factual"] - df["zero_factual"]
    hr = df["zero_hallucination"] - df["grounded_hallucination"]
    # Wilcoxon on differences; two-sided
    _, p_f = stats.wilcoxon(fg, zero_method="wilcox", alternative="two-sided")
    _, p_h = stats.wilcoxon(hr, zero_method="wilcox", alternative="two-sided")
    return float(p_f), float(p_h)


def compute_h2_kruskal(df: pd.DataFrame) -> tuple[float, float]:
    buckets = ["low_cm", "medium_cm", "high_cm"]
    groups_fg = [df.loc[df["cmi_bucket"] == b, "factual_gain"].values for b in buckets]
    groups_hr = [df.loc[df["cmi_bucket"] == b, "halluc_reduction"].values for b in buckets]
    _, p_fg = stats.kruskal(*groups_fg)
    _, p_hr = stats.kruskal(*groups_hr)
    return float(p_fg), float(p_hr)


# ──────────────────────────────────────────────
#  PLOT 1: H1 Bar Chart
# ──────────────────────────────────────────────
def plot_h1_bar(df: pd.DataFrame) -> None:
    metrics = ["Factual\nSupport", "Hallucination\nScore"]
    zero = [df["zero_factual"].mean(), df["zero_hallucination"].mean()]
    grounded = [df["grounded_factual"].mean(), df["grounded_hallucination"].mean()]

    x = np.arange(len(metrics))
    width = 0.34
    n = len(df)

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    bars1 = ax.bar(
        x - width / 2, zero, width, label="Zero-shot",
        color=THEME["zero_shot"], edgecolor=THEME["zero_shot_edge"], linewidth=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, grounded, width, label="Grounded (RAG)",
        color=THEME["grounded"], edgecolor=THEME["teal_darker"], linewidth=0.8,
    )

    ymax = max(max(zero), max(grounded)) * 1.42
    ax.set_ylim(0, ymax)
    pct_factual = 100.0 * (grounded[0] - zero[0]) / max(zero[0], 1e-9)
    pct_hall_red = 100.0 * (zero[1] - grounded[1]) / max(zero[1], 1e-9)
    pct_labels = [f"+{pct_factual:.1f}%", f"−{pct_hall_red:.1f}%"]
    lab_kw = dict(ha="center", va="bottom", fontsize=7, color=THEME["navy_text"])

    for bar, val in zip(bars1, zero):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{val:.3f}", **lab_kw)
    for bar, val, pl in zip(bars2, grounded, pct_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
            f"{val:.3f}\n({pl})", **lab_kw,
        )

    ax.set_facecolor("white")
    ax.set_ylabel("Mean Score", fontsize=9)
    ax.set_title(f"H1: Grounded Vs Zero-Shot (N = {n:,})", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=True, facecolor="white", edgecolor=THEME["grid"], loc="upper right", fontsize=7)

    save(fig, "01_h1_bar_comparison")


# ──────────────────────────────────────────────
#  PLOT 2: H1 Distributions (stacked + side-by-side variants)
# ──────────────────────────────────────────────
_PALETTE_H1_VIOLIN = {"Zero-shot": "#C5D8DE", "Grounded": "#1A4F5C"}


def _h1_violin_panel(
    ax,
    score_zero: pd.Series,
    score_grounded: pd.Series,
    title: str,
    *,
    ylabel: str | None = "Score",
    title_fs: float = 9,
    axis_label_fs: float = 8,
    tick_fs: float = 8,
    mean_label_fs: float = 6,
    mean_lw: float = 1.5,
    violin_lw: float = 0.7,
    inner: str = "box",
    spine_lw: float | None = None,
    y_max_ticks: int | None = None,
    mean_fmt: str = ".3f",
    split_x_labels: bool = False,
    x_tick_fs: float | None = None,
    mean_x_pad: float = 0.42,
    mean_hline_hw: float = 0.28,
    mean_label_dy: float = 0.0,
    mean_label_va: str = "center",
    xlim_right: float = 1.75,
    title_pad: float | None = None,
    label_pad: float | None = None,
    inner_line_lw: float | None = None,
    saturation: float | None = None,
    tick_pad_y: float | None = None,
    tick_pad_x: float | None = None,
) -> None:
    n = len(score_zero)
    data = pd.DataFrame({
        "Score": pd.concat([score_zero, score_grounded]),
        "Mode": ["Zero-shot"] * n + ["Grounded"] * n,
    })
    vkw: dict = dict(
        data=data, x="Mode", y="Score", hue="Mode", palette=_PALETTE_H1_VIOLIN,
        inner=inner, cut=0, ax=ax, legend=False, linewidth=violin_lw,
    )
    if saturation is not None:
        vkw["saturation"] = saturation
    sns.violinplot(**vkw)
    ax.set_facecolor("white")
    tkw = dict(fontsize=title_fs, fontweight="bold")
    if title_pad is not None:
        tkw["pad"] = title_pad
    ax.set_title(title, **tkw)
    ax.set_xlabel("")
    if ylabel:
        lkw = dict(fontsize=axis_label_fs)
        if label_pad is not None:
            lkw["labelpad"] = label_pad
        ax.set_ylabel(ylabel, **lkw)
    else:
        ax.set_ylabel("")
    ytp = dict(axis="y", labelsize=tick_fs, length=2.5, width=0.45)
    if tick_pad_y is not None:
        ytp["pad"] = tick_pad_y
    ax.tick_params(**ytp)
    xfs = x_tick_fs if x_tick_fs is not None else tick_fs
    xtp = dict(axis="x", labelsize=xfs, length=2.5, width=0.45)
    if tick_pad_x is not None:
        xtp["pad"] = tick_pad_x
    ax.tick_params(**xtp)
    if split_x_labels:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Zero-\nshot", "Ground-\ned"], fontsize=xfs, linespacing=0.9)
    if y_max_ticks is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_max_ticks, prune=None))
    if spine_lw is not None:
        for s in ax.spines.values():
            s.set_linewidth(spine_lw)
    if inner_line_lw is not None:
        for line in ax.lines:
            line.set_linewidth(inner_line_lw)
            line.set_color("#222222")
    z_mean = float(score_zero.mean())
    g_mean = float(score_grounded.mean())
    for i, m in enumerate([z_mean, g_mean]):
        ax.hlines(
            m, i - mean_hline_hw, i + mean_hline_hw,
            colors="#1a1a1a", linewidth=mean_lw, zorder=5,
        )
        ax.text(
            i + mean_x_pad, m + mean_label_dy, format(m, mean_fmt),
            ha="left", va=mean_label_va, fontsize=mean_label_fs,
            color=THEME["navy_text"], zorder=6,
        )
    ax.set_xlim(-0.52, xlim_right)


def plot_h1_distribution(df: pd.DataFrame) -> None:
    n = len(df)
    fig, axes = plt.subplots(2, 1, figsize=(3.65, 4.5), sharex=False)
    fig.subplots_adjust(hspace=0.4, top=0.92, right=0.96)
    fig.patch.set_facecolor("white")

    _h1_violin_panel(axes[0], df["zero_factual"], df["grounded_factual"], "Factual Support")
    _h1_violin_panel(axes[1], df["zero_hallucination"], df["grounded_hallucination"], "Hallucination Score")

    fig.suptitle(f"H1: Distributions (N = {n:,})", fontsize=10, fontweight="bold", y=1.0)
    save(fig, "02_h1_distributions")


def _plot_h1_distribution_horizontal(
    df: pd.DataFrame,
    *,
    figsize_in: tuple[float, float],
    save_name: str,
    wspace: float,
    top: float,
    bottom: float,
    right: float,
    left: float | None = None,
    suptitle_fs: float,
    suptitle_y: float,
    panel_kw: dict,
    sharey: bool = False,
    save_kwargs: dict | None = None,
) -> None:
    n = len(df)
    fig, axes = plt.subplots(1, 2, figsize=figsize_in, sharey=sharey)
    adj = dict(wspace=wspace, top=top, bottom=bottom, right=right)
    if left is not None:
        adj["left"] = left
    fig.subplots_adjust(**adj)
    fig.patch.set_facecolor("white")

    _h1_violin_panel(axes[0], df["zero_factual"], df["grounded_factual"], "Factual Support", **panel_kw)
    _h1_violin_panel(
        axes[1], df["zero_hallucination"], df["grounded_hallucination"], "Hallucination Score",
        ylabel="", **panel_kw,
    )

    fig.suptitle(f"H1: Distributions (N = {n:,})", fontsize=suptitle_fs, fontweight="bold", y=suptitle_y)
    save(fig, save_name, **(save_kwargs or {}))


def plot_h1_distribution_horizontal(df: pd.DataFrame) -> None:
    """Same violins as stacked `02`, arranged in one row for wide layouts."""
    _plot_h1_distribution_horizontal(
        df,
        figsize_in=(6.4, 2.85),
        save_name="02_h1_distributions_horizontal",
        wspace=0.38,
        top=0.86,
        bottom=0.16,
        right=0.97,
        left=None,
        suptitle_fs=10,
        suptitle_y=1.02,
        panel_kw={},
        sharey=False,
        save_kwargs=None,
    )


def plot_h1_distribution_horizontal_poster_cm(df: pd.DataFrame) -> None:
    """Word/poster slot: 6.6 cm × 4.1 cm; margins tuned so violins fill the frame (minimal inner whitespace)."""
    w_in = 6.6 / 2.54
    h_in = 4.1 / 2.54
    _plot_h1_distribution_horizontal(
        df,
        figsize_in=(w_in, h_in),
        save_name="02_h1_distributions_horizontal_poster_cm",
        wspace=0.20,
        top=0.868,
        bottom=0.175,
        right=0.965,
        left=0.125,
        suptitle_fs=6.25,
        suptitle_y=0.992,
        panel_kw=dict(
            title_fs=5.1,
            title_pad=0.45,
            axis_label_fs=4.75,
            label_pad=1.0,
            tick_fs=4.5,
            tick_pad_y=2.8,
            tick_pad_x=2.6,
            mean_label_fs=3.75,
            mean_lw=0.55,
            mean_fmt=".2f",
            mean_hline_hw=0.22,
            mean_x_pad=0.33,
            mean_label_dy=0.014,
            mean_label_va="bottom",
            xlim_right=1.70,
            violin_lw=0.28,
            inner="quartile",
            inner_line_lw=0.45,
            spine_lw=0.35,
            y_max_ticks=5,
            split_x_labels=False,
            x_tick_fs=3.35,
            saturation=0.88,
        ),
        sharey=True,
        save_kwargs=dict(pad_inches=0.006),
    )


# ──────────────────────────────────────────────
#  PLOT 3: Factual gain histogram
# ──────────────────────────────────────────────
def plot_factual_gain_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    gains = df["factual_gain"]
    ax.hist(gains, bins=26, color=THEME["grounded_light"], edgecolor="white", linewidth=0.35, alpha=0.9)
    ax.axvline(gains.mean(), color=THEME["teal_darker"], linewidth=1.1, linestyle="--",
               label=f"Mean {gains.mean():+.3f}")
    ax.axvline(0, color=THEME["slate"], linewidth=0.7, linestyle=":")
    ax.set_xlabel("Factual Gain", fontsize=8)
    ax.set_ylabel("Pairs", fontsize=8)
    ax.set_title("H1: Per-Pair Factual Gain", fontsize=9, fontweight="bold")
    ax.set_facecolor("white")
    ax.tick_params(labelsize=7)
    ax.legend(frameon=True, facecolor="white", edgecolor=THEME["grid"], loc="upper right", fontsize=7)
    save(fig, "03_factual_gain_distribution")


# ──────────────────────────────────────────────
#  PLOT 4: H2 CMI levels
# ──────────────────────────────────────────────
def plot_h2_grouped(df: pd.DataFrame) -> None:
    buckets = ["low_cm", "medium_cm", "high_cm"]
    labels = ["Low", "Medium", "High"]
    fg_means = [df[df["cmi_bucket"] == b]["factual_gain"].mean() for b in buckets]
    hr_means = [df[df["cmi_bucket"] == b]["halluc_reduction"].mean() for b in buckets]
    fg_se = [df[df["cmi_bucket"] == b]["factual_gain"].sem() for b in buckets]
    hr_se = [df[df["cmi_bucket"] == b]["halluc_reduction"].sem() for b in buckets]

    p_fg, p_hr = compute_h2_kruskal(df)
    fg_overall = df["factual_gain"].mean()
    hr_overall = df["halluc_reduction"].mean()

    x = np.arange(len(buckets))
    w = 0.26
    offset = 0.3
    c_fg = THEME["grounded"]
    c_hr = THEME["teal"]  # same blue-green family as primary #588894

    fig, ax = plt.subplots(figsize=(3.6, 3.45))
    fig.subplots_adjust(bottom=0.2)
    ax.set_facecolor(THEME["panel"])

    ax.bar(
        x - offset / 2, fg_means, w, yerr=fg_se,
        label="Factual gain", color=c_fg, edgecolor=THEME["teal_darker"], linewidth=0.65,
        capsize=2.5, error_kw={"linewidth": 0.7, "color": THEME["navy_text"]},
    )
    ax.bar(
        x + offset / 2, hr_means, w, yerr=hr_se,
        label="Hallucination reduction", color=c_hr, edgecolor=THEME["teal_darker"], linewidth=0.65,
        capsize=2.5, error_kw={"linewidth": 0.7, "color": THEME["navy_text"]},
    )

    ymax = max(max(np.array(fg_means) + np.array(fg_se)), max(np.array(hr_means) + np.array(hr_se))) * 1.28
    ax.set_ylim(0, ymax)
    ax.axhline(fg_overall, color=c_fg, linestyle="--", linewidth=1.15, alpha=0.95)
    ax.axhline(hr_overall, color=THEME["teal_darker"], linestyle=":", linewidth=1.15, alpha=0.95)

    for i, xi in enumerate(x):
        ax.text(xi - offset / 2, fg_means[i] + fg_se[i] + 0.012, f"{fg_means[i]:.2f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold", color=THEME["navy_text"])
        ax.text(xi + offset / 2, hr_means[i] + hr_se[i] + 0.012, f"{hr_means[i]:.2f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold", color=THEME["navy_text"])

    ax.set_ylabel("Mean Improvement", fontsize=8)
    ax.set_title(f"H2: RAG Benefit By CMI (N = {len(df):,})", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(frameon=True, facecolor="white", edgecolor=THEME["grid"], loc="upper right", fontsize=6.5)

    fig.text(
        0.5, 0.04,
        f"Kruskal–Wallis: factual p = {p_fg:.2f}, hallucination p = {p_hr:.2f} (n.s.); stable across mixing.",
        ha="center", fontsize=7.5, fontweight="bold", color=THEME["navy_text"],
    )
    save(fig, "04_h2_cmi_levels")


# ──────────────────────────────────────────────
#  PLOT 5: Per-condition gain
# ──────────────────────────────────────────────
def plot_per_condition(df: pd.DataFrame) -> None:
    all_c = (
        df.groupby("condition", observed=True)
        .agg(n=("factual_gain", "count"), factual_gain=("factual_gain", "mean"))
        .reset_index()
    )
    cond_data = all_c.nlargest(10, "factual_gain").sort_values("factual_gain", ascending=True)

    fig, ax = plt.subplots(figsize=(3.6, 3.8))
    ax.set_facecolor("white")
    y = np.arange(len(cond_data))
    colors = [THEME["grounded_light"] if g >= 0 else THEME["teal_darker"] for g in cond_data["factual_gain"]]
    ax.barh(y, cond_data["factual_gain"], color=colors, edgecolor="white", linewidth=0.4, height=0.58)
    ax.set_yticks(y)
    labels = [
        f"{row['condition'].replace('_', ' ').title()} (n={row['n']})"
        for _, row in cond_data.iterrows()
    ]
    ax.set_yticklabels(labels, fontsize=7)
    for i, t in enumerate(ax.get_yticklabels()):
        if i >= len(labels) - 4:
            t.set_fontweight("bold")

    ax.axvline(0, color=THEME["slate"], linewidth=0.7)
    mean_gain = df["factual_gain"].mean()
    ax.axvline(
        mean_gain, color=THEME["teal_darker"], linewidth=1.1, linestyle="--",
        label=f"Mean {mean_gain:+.3f}",
    )
    ax.set_xlabel("Mean Factual Gain", fontsize=8)
    ax.set_title("H1: Top Conditions By Gain", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=THEME["grid"], fontsize=6.5)

    save(fig, "05_per_condition_gain")


# ──────────────────────────────────────────────
#  PLOT 6: CMI scatter
# ──────────────────────────────────────────────
def plot_cmi_scatter(df: pd.DataFrame) -> None:
    rho_fg, p_fg = stats.spearmanr(df["cmi_score"], df["factual_gain"])
    rho_hr, p_hr = stats.spearmanr(df["cmi_score"], df["halluc_reduction"])

    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.65))
    fig.subplots_adjust(wspace=0.35, top=0.88)
    x_line = np.linspace(df["cmi_score"].min(), df["cmi_score"].max(), 80)

    axes[0].scatter(df["cmi_score"], df["factual_gain"], alpha=0.18, s=8,
                    color=THEME["grounded"], edgecolors="none", rasterized=True)
    z = np.polyfit(df["cmi_score"], df["factual_gain"], 1)
    axes[0].plot(x_line, np.poly1d(z)(x_line), color=THEME["teal_darker"], linewidth=1.2,
                 label=f"ρ = {rho_fg:.3f}, p = {p_fg:.3f}")
    axes[0].set_xlabel("CMI", fontsize=8)
    axes[0].set_ylabel("Factual Gain", fontsize=8)
    axes[0].set_title("Factual Gain", fontsize=8, fontweight="bold")
    axes[0].legend(frameon=True, facecolor="white", edgecolor=THEME["grid"], loc="best", fontsize=6.5)

    axes[1].scatter(df["cmi_score"], df["halluc_reduction"], alpha=0.18, s=8,
                    color=THEME["seafoam"], edgecolors="none", rasterized=True)
    z2 = np.polyfit(df["cmi_score"], df["halluc_reduction"], 1)
    axes[1].plot(x_line, np.poly1d(z2)(x_line), color=THEME["teal_darker"], linewidth=1.2,
                 label=f"ρ = {rho_hr:.3f}, p = {p_hr:.3f}")
    axes[1].set_xlabel("CMI", fontsize=8)
    axes[1].set_ylabel("Hallucination Reduction", fontsize=8)
    axes[1].set_title("Hallucination Reduction", fontsize=8, fontweight="bold")
    axes[1].legend(frameon=True, facecolor="white", edgecolor=THEME["grid"], loc="best", fontsize=6.5)
    for a in axes:
        a.set_facecolor("white")
        a.tick_params(labelsize=7)

    fig.suptitle("H2: CMI Vs Benefit (Spearman)", fontsize=9, fontweight="bold", y=1.0)
    save(fig, "06_cmi_scatter")


# ──────────────────────────────────────────────
#  PLOT 7: 11 → 3,015 (no overlapping pies)
# ──────────────────────────────────────────────
def plot_improvement() -> None:
    """Two bars only; log Y so Open-i (11) and MultiCaRe (3,015) are both visible."""
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    fig.subplots_adjust(left=0.18, bottom=0.2, right=0.96, top=0.88)
    ax.set_facecolor("white")

    names = ["Open-i", "MultiCaRe"]
    values = [11, 3015]
    x = np.arange(len(names))
    colors = [THEME["zero_shot"], THEME["grounded"]]

    bars = ax.bar(
        x, values, width=0.52, color=colors,
        edgecolor=[THEME["zero_shot_edge"], THEME["teal_darker"]], linewidth=0.75,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Usable Pairs (Log Scale)", fontsize=8)
    ax.set_title("Evidence Scale-Up (MMCQSD)", fontsize=9, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(6, 5500)
    ax.tick_params(axis="y", labelsize=7)
    ax.yaxis.grid(True, which="major", color=THEME["grid"], linewidth=0.35, alpha=0.7)
    ax.set_axisbelow(True)

    for bar, v in zip(bars, values):
        top = v * 1.12
        ax.text(
            bar.get_x() + bar.get_width() / 2, top, f"{v:,}",
            ha="center", va="bottom", fontsize=7, color=THEME["navy_text"],
        )

    save(fig, "07_limitation_improvement")


# ──────────────────────────────────────────────
#  PLOT 8: Pipeline diagram (compact, no overlap)
# ──────────────────────────────────────────────
def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 2.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    ax.axis("off")

    box = dict(boxstyle="round,pad=0.28", linewidth=1.0, edgecolor=THEME["teal_dark"])
    arr = dict(arrowstyle="-|>", color=THEME["teal_darker"], linewidth=1.0,
               mutation_scale=10, shrinkA=2, shrinkB=2)

    stages = [
        (0.55, 1.15, "Hinglish\nquery\n(MMCQSD)", THEME["cm_low"]),
        (2.05, 1.15, "LaBSE\nencode", THEME["teal_dark"]),
        (3.55, 1.15, "FAISS\nretrieve", THEME["grounded"]),
        (5.05, 1.15, "Adaptive\ncontext", THEME["seafoam"]),
        (6.55, 1.15, "Llama-3.1-8B\n(Groq)", THEME["teal_darker"]),
        (8.05, 1.15, "Hinglish\nanswer", THEME["cm_high"]),
    ]

    for i, (x, y, label, color) in enumerate(stages):
        ax.annotate(
            label, (x, y), fontsize=6.5, ha="center", va="center", fontweight="bold",
            color="white", bbox=dict(facecolor=color, **box),
        )
        if i < len(stages) - 1:
            x0, y0 = stages[i][0], stages[i][1]
            x1, y1 = stages[i + 1][0], stages[i + 1][1]
            ax.annotate("", xy=(x1 - 0.42, y1), xytext=(x0 + 0.42, y0), arrowprops=arr)

    ax.annotate(
        "MultiCaRe evidence\n(61,316 filtered;\n10K indexed)", (3.55, 2.15),
        fontsize=6, ha="center", va="center", fontweight="bold", color="white",
        bbox=dict(facecolor=THEME["grounded"], boxstyle="round,pad=0.22", linewidth=0.8,
                  edgecolor=THEME["teal_dark"]),
    )
    ax.annotate("", xy=(3.55, 1.48), xytext=(3.55, 1.82),
                arrowprops=dict(arrowstyle="-|>", color=THEME["teal_darker"], linewidth=1.0,
                                mutation_scale=9))

    leg = dict(boxstyle="round,pad=0.2", linewidth=0.6, edgecolor=THEME["grid"])
    ax.annotate("Grounded path\n(evidence in prompt)", (6.55, 0.35), fontsize=5.8,
                ha="center", va="center", color=THEME["navy_text"],
                bbox=dict(facecolor=THEME["panel"], **leg))
    ax.set_title("Pipeline Overview", fontsize=9, fontweight="bold", pad=4)
    save(fig, "08_pipeline_architecture")


# ──────────────────────────────────────────────
#  PLOT 9: Match quality
# ──────────────────────────────────────────────
def plot_match_quality() -> None:
    path = _resolve_csv("data/processed/mmcqsd_multicare_paired.csv")
    if not path.exists():
        print("  Skipping 09_match_quality (mmcqsd_multicare_paired.csv not found)")
        return

    pairs = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(4.2, 2.85))
    fig.subplots_adjust(wspace=0.4, left=0.12, right=0.98, top=0.82, bottom=0.2)
    fig.patch.set_facecolor(THEME["figure_bg"])

    sim = pairs["similarity_score"]
    m = float(sim.mean())
    x_hi = float(sim.max()) + 0.02

    n_bins = 34
    axes[0].hist(
        sim, bins=n_bins, color=THEME["mint"], edgecolor="white", linewidth=0.45, alpha=0.95,
    )
    axes[0].axvspan(m, x_hi, alpha=0.32, color=THEME["grounded_light"], zorder=0)
    axes[0].axvline(m, color=THEME["teal_darker"], linewidth=1.2, linestyle="--", label=f"Mean = {m:.3f}")
    axes[0].set_xlabel("LaBSE Similarity", fontsize=8)
    axes[0].set_ylabel("Count", fontsize=8)
    axes[0].set_title("Similarity Distribution", fontsize=8, fontweight="bold")
    axes[0].set_facecolor("white")
    axes[0].tick_params(labelsize=7)
    axes[0].legend(loc="upper left", frameon=True, fontsize=6.5)

    quality_counts = pairs["match_quality"].value_counts()
    hi = int(quality_counts.get("high", 0))
    med = int(quality_counts.get("medium", 0))
    lo = int(quality_counts.get("low", 0))
    total = hi + med + lo
    p_hi, p_med, p_lo = 100.0 * hi / total, 100.0 * med / total, 100.0 * lo / total
    sizes = [hi, med, lo]
    lbls = [f"High\n{p_hi:.1f}%", f"Medium\n{p_med:.1f}%", f"Low\n{p_lo:.1f}%"]
    colors_p = [THEME["grounded"], THEME["teal"], THEME["light_secondary"]]
    axes[1].pie(
        sizes, labels=lbls, colors=colors_p, startangle=140,
        textprops={"fontsize": 7, "fontweight": "bold"},
        wedgeprops=dict(linewidth=0.6, edgecolor="white"),
    )
    axes[1].set_title("Quality Tier", fontsize=8, fontweight="bold")

    pct_med_hi = 100.0 * (hi + med) / total
    fig.suptitle(
        f"Retrieval Match (N = {total:,}) · {pct_med_hi:.0f}% Medium+High",
        fontsize=9, fontweight="bold", y=1.0,
    )

    save(fig, "09_match_quality")


# ──────────────────────────────────────────────
#  PLOT 10: Significance summary
# ──────────────────────────────────────────────
def plot_significance_summary(df: pd.DataFrame) -> None:
    p_f, p_h = compute_h1_pvalues(df)
    p_kw_fg, p_kw_hr = compute_h2_kruskal(df)
    rho_fg, p_sp_fg = stats.spearmanr(df["cmi_score"], df["factual_gain"])

    tests = [
        "H1: Wilcoxon Factual Support",
        "H1: Wilcoxon Hallucination Reduction",
        "H2: Kruskal–Wallis Factual Gain",
        "H2: Kruskal–Wallis Hallucination Reduction",
        "H2: Spearman CMI × Factual Gain",
    ]
    p_values = [p_f, p_h, p_kw_fg, p_kw_hr, p_sp_fg]
    log_p = [-np.log10(max(p, 1e-300)) for p in p_values]
    sig_green = "#2D6A4F"
    ns_grey = "#ADB5BD"
    colors = [sig_green if p < 0.05 else ns_grey for p in p_values]

    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.set_facecolor("white")
    y = np.arange(len(tests))
    ax.barh(y, log_p, color=colors, edgecolor="white", linewidth=0.45, height=0.55)
    thr = -np.log10(0.05)
    ax.axvline(thr, color="#333333", linewidth=1.0, linestyle="--", label="α = 0.05")
    ax.set_yticks(y)
    short_tests = [
        "H1 Wilcoxon Factual",
        "H1 Wilcoxon Hallucination",
        "H2 K–W Factual",
        "H2 K–W Hallucination",
        "H2 Spearman CMI×FG",
    ]
    ax.set_yticklabels(short_tests, fontsize=6.5)
    ax.set_xlabel("−log₁₀(p)", fontsize=8)
    ax.set_title(f"Tests (N = {len(df):,})", fontsize=9, fontweight="bold")

    xmax = max(max(log_p) * 1.08, thr * 1.35)
    ax.set_xlim(0, xmax)
    for i, (lp, p) in enumerate(zip(log_p, p_values)):
        lbl = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"
        ax.text(
            min(lp + 0.15, xmax * 0.97), i, lbl,
            va="center", fontsize=6.5, fontweight="bold",
            color="#1a1a1a" if p < 0.05 else "#555555",
        )

    leg = [
        mpatches.Patch(facecolor=sig_green, edgecolor="white", label="p < 0.05"),
        mpatches.Patch(facecolor=ns_grey, edgecolor="white", label="p ≥ 0.05"),
    ]
    ax.legend(handles=leg, loc="lower right", frameon=True, fontsize=6, facecolor="white")

    save(fig, "10_significance_summary")


# ──────────────────────────────────────────────
#  PLOT 11: Ablation
# ──────────────────────────────────────────────
def plot_ablation() -> None:
    ablation_path = _resolve_csv("results/multicare_h1_ablation/ablation_scored.csv")
    if not ablation_path.exists():
        print("  Skipping 11_ablation_comparison (no ablation_scored.csv)")
        return

    struct = pd.read_csv(ablation_path)
    raw = pd.read_csv(_resolve_csv("results/combined_h1h2/combined_scored.csv"))
    common_ids = set(struct["pair_id"].tolist())
    raw = raw[raw["pair_id"].isin(common_ids)]

    metrics = ["Factual\n(Grounded)", "Hallucination\n(Grounded)", "Factual\nGain", "Hallucination\nReduction"]
    raw_vals = [
        raw["grounded_factual"].mean(),
        raw["grounded_hallucination"].mean(),
        (raw["grounded_factual"] - raw["zero_factual"]).mean(),
        (raw["zero_hallucination"] - raw["grounded_hallucination"]).mean(),
    ]
    struct_vals = [
        struct["grounded_factual"].mean(),
        struct["grounded_hallucination"].mean(),
        (struct["grounded_factual"] - struct["zero_factual"]).mean(),
        (struct["zero_hallucination"] - struct["grounded_hallucination"]).mean(),
    ]

    x = np.arange(len(metrics))
    w = 0.34
    fig, ax = plt.subplots(figsize=(4.0, 2.65))
    ax.bar(x - w / 2, raw_vals, w, label="Raw evidence", color=THEME["slate"], edgecolor="white", linewidth=0.35)
    ax.bar(x + w / 2, struct_vals, w, label="Structured evidence", color=THEME["grounded"],
           edgecolor="white", linewidth=0.35)

    top = max(max(raw_vals), max(struct_vals)) * 1.18
    ax.set_ylim(0, top)
    for i in x:
        ax.text(i - w / 2, raw_vals[i] + 0.012, f"{raw_vals[i]:.3f}", ha="center", fontsize=5.8)
        ax.text(i + w / 2, struct_vals[i] + 0.012, f"{struct_vals[i]:.3f}", ha="center", fontsize=5.8)

    ax.set_ylabel("Mean Score")
    ax.set_title(f"Ablation: Raw Vs Structured (N = {len(struct):,})", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=6.5)
    ax.set_facecolor("white")
    ax.legend(frameon=True, facecolor="white", edgecolor=THEME["grid"], loc="upper right")
    save(fig, "11_ablation_comparison")


# ──────────────────────────────────────────────
#  NEW PLOT 12: Grounded factual by CMI (robustness)
# ──────────────────────────────────────────────
def plot_h2_grounded_factual_bars(df: pd.DataFrame) -> None:
    buckets = ["low_cm", "medium_cm", "high_cm"]
    labels = ["Low", "Medium", "High"]
    means = [df[df["cmi_bucket"] == b]["grounded_factual"].mean() for b in buckets]
    se = [df[df["cmi_bucket"] == b]["grounded_factual"].sem() for b in buckets]
    ns = [len(df[df["cmi_bucket"] == b]) for b in buckets]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(buckets))
    cols = [THEME["cm_low"], THEME["cm_med"], THEME["cm_high"]]
    ax.bar(x, means, yerr=se, color=cols, edgecolor="white", linewidth=0.4, capsize=2,
           error_kw={"linewidth": 0.6, "color": THEME["navy_text"]})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Mean Grounded Factual", fontsize=8)
    ax.set_ylim(0, max(np.array(means) + np.array(se)) * 1.25)
    for xi, m, n in zip(x, means, ns):
        ax.text(xi, m + 0.02, f"{m:.3f}\n(n={n})", ha="center", va="bottom", fontsize=5.8)
    ax.set_title(f"H2: Grounded Factual By CMI (N = {len(df):,})", fontsize=9, fontweight="bold")
    ax.set_facecolor("white")
    save(fig, "12_h2_grounded_factual_by_cmi")


# ──────────────────────────────────────────────
#  NEW PLOT 13: Cohen's d (paired) for H1
# ──────────────────────────────────────────────
def plot_h1_cohens_d(df: pd.DataFrame) -> None:
    fg = df["grounded_factual"] - df["zero_factual"]
    hr = df["zero_hallucination"] - df["grounded_hallucination"]
    # Paired Cohen's d: mean(diff) / std(diff)
    d_f = fg.mean() / fg.std(ddof=1)
    d_h = hr.mean() / hr.std(ddof=1)
    labels = ["Factual Gain\n(Paired)", "Hallucination Reduction\n(Paired)"]
    vals = [d_f, d_h]

    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    cols = [THEME["grounded"], THEME["seafoam"]]
    ax.barh(labels, vals, color=cols, edgecolor="white", linewidth=0.4, height=0.5)
    ax.axvline(0, color=THEME["slate"], linewidth=0.7)
    ax.axvline(0.2, color=THEME["mint"], linewidth=0.8, linestyle=":", label="d = 0.2 (small)")
    ax.axvline(0.5, color=THEME["light_secondary"], linewidth=0.8, linestyle=":", label="d = 0.5 (medium)")
    ax.set_xlabel("Cohen's D (Paired)", fontsize=9, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=6.5, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=THEME["grid"], fontsize=5.5)
    ax.set_title(f"H1: Cohen's D (Paired, N = {len(df):,})", fontsize=9, fontweight="bold")
    ax.set_facecolor("white")
    save(fig, "13_h1_cohens_d")


# ──────────────────────────────────────────────
#  NEW PLOT 14: Summary “lollipop” — mean deltas
# ──────────────────────────────────────────────
def plot_h1_delta_lollipop(df: pd.DataFrame) -> None:
    fg = df["factual_gain"].mean()
    hr = df["halluc_reduction"].mean()
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    y = [0, 1]
    ax.hlines(y, [0, 0], [fg, hr], color=THEME["teal_dark"], linewidth=2)
    ax.scatter([fg, hr], y, color=THEME["grounded"], s=55, zorder=3, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=THEME["slate"], linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(["Δ Factual Support", "Δ Hallucination Reduction"], fontsize=6.5)
    ax.set_xlabel("Mean Paired Change (Grounded − Zero / Vice Versa)")
    ax.set_title(f"H1: Mean Deltas (N = {len(df):,})", fontsize=9, fontweight="bold")
    ax.set_facecolor("white")
    for yi, v in zip(y, [fg, hr]):
        ax.text(v + 0.01, yi, f"{v:+.3f}", va="center", fontsize=6.5, fontweight="bold")
    save(fig, "14_h1_mean_deltas_lollipop")


# ──────────────────────────────────────────────
def main() -> None:
    apply_theme()
    print("Loading data...")
    df = load_data()
    print()

    print("Generating plots...")
    plot_h1_bar(df)
    plot_h1_distribution(df)
    plot_h1_distribution_horizontal(df)
    plot_h1_distribution_horizontal_poster_cm(df)
    plot_factual_gain_hist(df)
    plot_h2_grouped(df)
    plot_per_condition(df)
    plot_cmi_scatter(df)
    plot_improvement()
    plot_pipeline()
    plot_match_quality()
    plot_significance_summary(df)
    plot_ablation()
    plot_h2_grounded_factual_bars(df)
    plot_h1_cohens_d(df)
    plot_h1_delta_lollipop(df)

    print()
    print(f"All plots saved to {OUT_DIR}")
    print()
    print("THEME (hex):")
    for name in ("teal", "teal_dark", "panel", "grounded", "zero_shot"):
        print(f"  {name:12s} {THEME[name]}")


if __name__ == "__main__":
    main()
