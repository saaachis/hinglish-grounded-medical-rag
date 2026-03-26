"""Generate all research poster visualizations.

Theme:
  - Primary:   #2563EB  (blue)
  - Secondary: #DC2626  (red)
  - Accent:    #059669  (green)
  - Neutral:   #6B7280  (gray)
  - Background:#F9FAFB  (off-white)

To adapt to your poster template later, change the THEME dict below.
All plots are saved as both PNG (300 dpi) and PDF (vector) in this directory.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")

# ──────────────────────────────────────────────
#  THEME — change these to match your poster
# ──────────────────────────────────────────────
THEME = {
    "primary":    "#2563EB",
    "secondary":  "#DC2626",
    "accent":     "#059669",
    "neutral":    "#6B7280",
    "bg":         "#F9FAFB",
    "text":       "#1F2937",
    "light_primary": "#93C5FD",
    "light_secondary": "#FCA5A5",
    "light_accent": "#6EE7B7",
    "cm_low":     "#3B82F6",
    "cm_med":     "#F59E0B",
    "cm_high":    "#EF4444",
}

FONT_FAMILY = "sans-serif"
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
DPI = 300

OUT_DIR = Path(__file__).parent


def apply_theme():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": TICK_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.facecolor": THEME["bg"],
        "figure.facecolor": "white",
        "axes.edgecolor": THEME["neutral"],
        "axes.labelcolor": THEME["text"],
        "xtick.color": THEME["text"],
        "ytick.color": THEME["text"],
        "text.color": THEME["text"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}.png")


def load_data():
    df = pd.read_csv("results/combined_h1h2/combined_scored.csv")
    print(f"Loaded {len(df)} clean pairs")
    return df


# ──────────────────────────────────────────────
#  PLOT 1: H1 Bar Chart — Grounded vs Zero-Shot
# ──────────────────────────────────────────────
def plot_h1_bar(df):
    metrics = ["Factual Support", "Hallucination Score"]
    zero = [df["zero_factual"].mean(), df["zero_hallucination"].mean()]
    grounded = [df["grounded_factual"].mean(), df["grounded_hallucination"].mean()]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars1 = ax.bar(x - width / 2, zero, width, label="Zero-Shot",
                   color=THEME["secondary"], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, grounded, width, label="Grounded (RAG)",
                   color=THEME["primary"], edgecolor="white", linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                color=THEME["secondary"], fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                color=THEME["primary"], fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("H1: Grounded RAG vs Zero-Shot Generation")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=True, facecolor="white", edgecolor=THEME["neutral"])

    save(fig, "01_h1_bar_comparison")


# ──────────────────────────────────────────────
#  PLOT 2: H1 Distribution — Violin/Box Plots
# ──────────────────────────────────────────────
def plot_h1_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Factual support
    data_factual = pd.DataFrame({
        "Score": pd.concat([df["zero_factual"], df["grounded_factual"]]),
        "Mode": ["Zero-Shot"] * len(df) + ["Grounded"] * len(df),
    })
    palette_f = {"Zero-Shot": THEME["secondary"], "Grounded": THEME["primary"]}
    sns.violinplot(data=data_factual, x="Mode", y="Score", palette=palette_f,
                   inner="box", cut=0, ax=axes[0])
    axes[0].set_title("Factual Support Distribution")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Factual Support Score")

    # Hallucination
    data_halluc = pd.DataFrame({
        "Score": pd.concat([df["zero_hallucination"], df["grounded_hallucination"]]),
        "Mode": ["Zero-Shot"] * len(df) + ["Grounded"] * len(df),
    })
    palette_h = {"Zero-Shot": THEME["secondary"], "Grounded": THEME["primary"]}
    sns.violinplot(data=data_halluc, x="Mode", y="Score", palette=palette_h,
                   inner="box", cut=0, ax=axes[1])
    axes[1].set_title("Hallucination Score Distribution")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Hallucination Score")

    fig.suptitle("H1: Score Distributions (n=447)", fontsize=TITLE_SIZE, y=1.02)
    fig.tight_layout()
    save(fig, "02_h1_distributions")


# ──────────────────────────────────────────────
#  PLOT 3: H1 Factual Gain Distribution
# ──────────────────────────────────────────────
def plot_factual_gain_hist(df):
    fig, ax = plt.subplots(figsize=(6, 4))

    gains = df["factual_gain"]
    ax.hist(gains, bins=30, color=THEME["primary"], edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(gains.mean(), color=THEME["secondary"], linewidth=2,
               linestyle="--", label=f"Mean = {gains.mean():+.3f}")
    ax.axvline(0, color=THEME["neutral"], linewidth=1, linestyle=":")

    ax.set_xlabel("Factual Gain (Grounded - Zero-Shot)")
    ax.set_ylabel("Number of Pairs")
    ax.set_title("Distribution of Factual Gain per Pair")
    ax.legend(frameon=True, facecolor="white", edgecolor=THEME["neutral"])

    save(fig, "03_factual_gain_distribution")


# ──────────────────────────────────────────────
#  PLOT 4: H2 Grouped Bar — 3 CMI Levels
# ──────────────────────────────────────────────
def plot_h2_grouped(df):
    buckets = ["low_cm", "medium_cm", "high_cm"]
    labels = ["Low CM\n(more English)", "Medium CM", "High CM\n(more Hindi)"]
    colors = [THEME["cm_low"], THEME["cm_med"], THEME["cm_high"]]

    fg_means = [df[df["cmi_bucket"] == b]["factual_gain"].mean() for b in buckets]
    hr_means = [df[df["cmi_bucket"] == b]["halluc_reduction"].mean() for b in buckets]
    fg_se = [df[df["cmi_bucket"] == b]["factual_gain"].sem() for b in buckets]
    hr_se = [df[df["cmi_bucket"] == b]["halluc_reduction"].sem() for b in buckets]

    x = np.arange(len(buckets))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, fg_means, width, yerr=fg_se,
                   label="Factual Gain", color=[c for c in colors],
                   edgecolor="white", linewidth=0.5, capsize=3)
    bars2 = ax.bar(x + width / 2, hr_means, width, yerr=hr_se,
                   label="Halluc. Reduction", color=[c for c in colors],
                   edgecolor="white", linewidth=0.5, capsize=3, alpha=0.55)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score Improvement")
    ax.set_title("H2: RAG Benefit by Code-Mixing Intensity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.40)
    ax.legend(frameon=True, facecolor="white", edgecolor=THEME["neutral"])

    ax.text(0.98, 0.95, "Kruskal-Wallis p = 0.238 (n.s.)\nRAG benefit is robust across CMI levels",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=THEME["neutral"], style="italic")

    save(fig, "04_h2_cmi_levels")


# ──────────────────────────────────────────────
#  PLOT 5: Per-Condition Factual Gain (Horizontal)
# ──────────────────────────────────────────────
def plot_per_condition(df):
    cond_data = df.groupby("condition").agg(
        n=("factual_gain", "count"),
        factual_gain=("factual_gain", "mean"),
        halluc_reduction=("halluc_reduction", "mean"),
    ).reset_index()
    cond_data = cond_data.sort_values("factual_gain", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    colors = [THEME["primary"] if g >= 0 else THEME["secondary"]
              for g in cond_data["factual_gain"]]
    bars = ax.barh(range(len(cond_data)), cond_data["factual_gain"],
                   color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    ax.set_yticks(range(len(cond_data)))
    labels = [f"{row['condition'].replace('_', ' ').title()} (n={row['n']})"
              for _, row in cond_data.iterrows()]
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color=THEME["neutral"], linewidth=0.8)
    ax.set_xlabel("Mean Factual Gain (Grounded - Zero-Shot)")
    ax.set_title("Factual Gain by Medical Condition")

    mean_gain = df["factual_gain"].mean()
    ax.axvline(mean_gain, color=THEME["accent"], linewidth=1.5, linestyle="--",
               label=f"Overall mean = {mean_gain:+.3f}")
    ax.legend(frameon=True, facecolor="white", edgecolor=THEME["neutral"],
              loc="lower right", fontsize=9)

    save(fig, "05_per_condition_gain")


# ──────────────────────────────────────────────
#  PLOT 6: CMI Scatter / Correlation
# ──────────────────────────────────────────────
def plot_cmi_scatter(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Factual gain vs CMI
    axes[0].scatter(df["cmi_score"], df["factual_gain"],
                    alpha=0.25, s=12, color=THEME["primary"], edgecolors="none")
    z = np.polyfit(df["cmi_score"], df["factual_gain"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["cmi_score"].min(), df["cmi_score"].max(), 100)
    axes[0].plot(x_line, p(x_line), color=THEME["secondary"], linewidth=2,
                 label=f"Trend (rho=0.080, p=0.091)")
    axes[0].set_xlabel("Code-Mixing Index (CMI)")
    axes[0].set_ylabel("Factual Gain")
    axes[0].set_title("CMI vs Factual Gain")
    axes[0].legend(frameon=True, facecolor="white", edgecolor=THEME["neutral"], fontsize=9)

    # Halluc reduction vs CMI
    axes[1].scatter(df["cmi_score"], df["halluc_reduction"],
                    alpha=0.25, s=12, color=THEME["accent"], edgecolors="none")
    z2 = np.polyfit(df["cmi_score"], df["halluc_reduction"], 1)
    p2 = np.poly1d(z2)
    axes[1].plot(x_line, p2(x_line), color=THEME["secondary"], linewidth=2,
                 label=f"Trend (rho=0.017, p=0.725)")
    axes[1].set_xlabel("Code-Mixing Index (CMI)")
    axes[1].set_ylabel("Hallucination Reduction")
    axes[1].set_title("CMI vs Hallucination Reduction")
    axes[1].legend(frameon=True, facecolor="white", edgecolor=THEME["neutral"], fontsize=9)

    fig.suptitle("H2: Correlation Between Code-Mixing and RAG Performance", fontsize=TITLE_SIZE, y=1.02)
    fig.tight_layout()
    save(fig, "06_cmi_scatter")


# ──────────────────────────────────────────────
#  PLOT 7: Limitation Improvement (11 → 3,015)
# ──────────────────────────────────────────────
def plot_improvement(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: pair count comparison
    categories = ["Open-i\n(Before)", "MultiCaRe\n(After)"]
    values = [11, 3015]
    colors = [THEME["secondary"], THEME["primary"]]
    bars = axes[0].bar(categories, values, color=colors, edgecolor="white",
                       linewidth=0.5, width=0.5)
    axes[0].set_ylabel("Number of Usable Pairs")
    axes[0].set_title("Dataset Pairing Improvement")
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f"{val:,}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[0].text(0.5, 0.85, "274x increase", transform=axes[0].transAxes,
                 ha="center", fontsize=13, fontweight="bold", color=THEME["accent"])
    axes[0].set_ylim(0, 3600)

    # Right: condition coverage
    cov_before = [1, 17]
    cov_after = [18, 0]
    labels = ["Covered", "Not Covered"]
    colors_pie = [THEME["primary"], THEME["light_secondary"]]

    wedges1, _ = axes[1].pie([1, 17], colors=[THEME["light_secondary"], THEME["secondary"]],
                              startangle=90, radius=0.65, center=(-0.35, 0),
                              wedgeprops=dict(linewidth=1, edgecolor="white"))
    wedges2, _ = axes[1].pie([18, 0], colors=[THEME["primary"], THEME["light_primary"]],
                              startangle=90, radius=0.65, center=(0.55, 0),
                              wedgeprops=dict(linewidth=1, edgecolor="white"))
    axes[1].text(-0.35, -0.9, "Open-i\n1/18 conditions", ha="center", fontsize=9)
    axes[1].text(0.55, -0.9, "MultiCaRe\n18/18 conditions", ha="center", fontsize=9)
    axes[1].set_title("Medical Condition Coverage")
    axes[1].set_aspect("equal")

    fig.suptitle("Overcoming the Dataset Limitation", fontsize=TITLE_SIZE, y=1.02)
    fig.tight_layout()
    save(fig, "07_limitation_improvement")


# ──────────────────────────────────────────────
#  PLOT 8: Pipeline / Architecture Diagram
# ──────────────────────────────────────────────
def plot_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.4", linewidth=1.5)
    arrow_props = dict(arrowstyle="-|>", color=THEME["neutral"], linewidth=1.5)

    stages = [
        (1.0, 1.75, "MMCQSD\nHinglish Queries\n(3,015)", THEME["cm_med"]),
        (3.5, 1.75, "LaBSE\nEncoder", THEME["primary"]),
        (6.0, 1.75, "FAISS\nRetrieval", THEME["accent"]),
        (8.5, 1.75, "Llama-3.1-8B\nGenerator", THEME["secondary"]),
        (11.0, 1.75, "Hinglish\nResponse", THEME["cm_high"]),
    ]

    for x, y, label, color in stages:
        ax.annotate(label, (x, y), fontsize=9, ha="center", va="center",
                    fontweight="bold", color="white",
                    bbox=dict(facecolor=color, **box_style))

    for i in range(len(stages) - 1):
        ax.annotate("", xy=(stages[i + 1][0] - 0.8, stages[i + 1][1]),
                     xytext=(stages[i][0] + 0.8, stages[i][1]),
                     arrowprops=arrow_props)

    ax.annotate("MultiCaRe\nEvidence Corpus\n(61,316 cases)", (6.0, 3.2),
                fontsize=9, ha="center", va="center", fontweight="bold",
                color="white",
                bbox=dict(facecolor=THEME["primary"], alpha=0.7, **box_style))
    ax.annotate("", xy=(6.0, 2.2), xytext=(6.0, 2.8),
                arrowprops=dict(arrowstyle="-|>", color=THEME["primary"], linewidth=1.5))

    label_box = dict(boxstyle="round,pad=0.4", linewidth=1)
    ax.annotate("Grounded\n(with evidence)", (8.5, 0.4), fontsize=8,
                ha="center", va="center", color=THEME["primary"],
                bbox=dict(facecolor=THEME["light_primary"], **label_box))
    ax.annotate("Zero-Shot\n(no evidence)", (8.5, 3.2), fontsize=8,
                ha="center", va="center", color=THEME["secondary"],
                bbox=dict(facecolor=THEME["light_secondary"], **label_box))

    ax.set_title("System Architecture: Grounded Multimodal RAG for Hinglish Clinical Queries",
                 fontsize=TITLE_SIZE, pad=15)

    save(fig, "08_pipeline_architecture")


# ──────────────────────────────────────────────
#  PLOT 9: Match Quality Distribution
# ──────────────────────────────────────────────
def plot_match_quality():
    pairs = pd.read_csv("data/processed/mmcqsd_multicare_paired.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram of similarity scores
    axes[0].hist(pairs["similarity_score"], bins=40, color=THEME["primary"],
                 edgecolor="white", linewidth=0.5, alpha=0.85)
    axes[0].axvline(pairs["similarity_score"].mean(), color=THEME["secondary"],
                    linewidth=2, linestyle="--",
                    label=f"Mean = {pairs['similarity_score'].mean():.3f}")
    axes[0].set_xlabel("Cosine Similarity (LaBSE)")
    axes[0].set_ylabel("Number of Pairs")
    axes[0].set_title("Similarity Score Distribution")
    axes[0].legend(frameon=True, facecolor="white", edgecolor=THEME["neutral"])

    # Match quality pie
    quality_counts = pairs["match_quality"].value_counts()
    labels = [f"High\n(n={quality_counts.get('high', 0):,})",
              f"Medium\n(n={quality_counts.get('medium', 0):,})",
              f"Low\n(n={quality_counts.get('low', 0)})"]
    sizes = [quality_counts.get("high", 0), quality_counts.get("medium", 0),
             quality_counts.get("low", 0)]
    colors = [THEME["accent"], THEME["cm_med"], THEME["secondary"]]
    wedges, texts = axes[1].pie(sizes, labels=labels, colors=colors,
                                 startangle=90,
                                 wedgeprops=dict(linewidth=1.5, edgecolor="white"))
    for t in texts:
        t.set_fontsize(10)
    axes[1].set_title("Match Quality Breakdown (3,015 pairs)")

    fig.suptitle("LaBSE + FAISS Matching Results", fontsize=TITLE_SIZE, y=1.02)
    fig.tight_layout()
    save(fig, "09_match_quality")


# ──────────────────────────────────────────────
#  PLOT 10: Statistical Significance Summary
# ──────────────────────────────────────────────
def plot_significance_summary(df):
    fig, ax = plt.subplots(figsize=(7, 4))

    tests = [
        "H1: Factual\nSupport",
        "H1: Hallucination\nReduction",
        "H2: CMI Effect\n(Factual Gain)",
        "H2: CMI Effect\n(Halluc. Red.)",
    ]
    p_values = [1.21e-24, 1.14e-19, 0.2375, 0.5193]
    log_p = [-np.log10(max(p, 1e-30)) for p in p_values]
    colors = [THEME["primary"] if p < 0.05 else THEME["neutral"] for p in p_values]

    bars = ax.barh(range(len(tests)), log_p, color=colors,
                   edgecolor="white", linewidth=0.5, height=0.6)

    threshold = -np.log10(0.05)
    ax.axvline(threshold, color=THEME["secondary"], linewidth=1.5, linestyle="--",
               label="p = 0.05 threshold")

    ax.set_yticks(range(len(tests)))
    ax.set_yticklabels(tests, fontsize=10)
    ax.set_xlabel("-log10(p-value)")
    ax.set_title("Statistical Significance of Hypotheses")
    ax.legend(frameon=True, facecolor="white", edgecolor=THEME["neutral"], loc="lower right")

    for bar, p in zip(bars, p_values):
        label = f"p={p:.1e}" if p < 0.05 else f"p={p:.3f}"
        sig = "***" if p < 0.001 else "n.s."
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{label} {sig}", va="center", fontsize=9,
                color=THEME["primary"] if p < 0.05 else THEME["neutral"])

    save(fig, "10_significance_summary")


# ──────────────────────────────────────────────
def plot_ablation():
    """Phase 6 ablation: raw vs structured evidence comparison."""
    ablation_path = Path("results/multicare_h1_ablation/ablation_scored.csv")
    if not ablation_path.exists():
        print("  Skipping ablation plot (no data)")
        return

    struct = pd.read_csv(ablation_path)
    raw = pd.read_csv("results/combined_h1h2/combined_scored.csv")
    common_ids = set(struct["pair_id"].tolist())
    raw = raw[raw["pair_id"].isin(common_ids)]

    metrics = ["Factual Support\n(Grounded)", "Hallucination\n(Grounded)", "Factual Gain", "Halluc Reduction"]
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
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w / 2, raw_vals, w, label="Raw Evidence", color=THEME["neutral"], edgecolor="white")
    bars2 = ax.bar(x + w / 2, struct_vals, w, label="Structured Evidence", color=THEME["accent"], edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title(f"Phase 6 Ablation: Raw vs Structured Evidence (n={len(struct)})")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, max(max(raw_vals), max(struct_vals)) * 1.15)
    fig.tight_layout()
    save(fig, "11_ablation_comparison")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main():
    apply_theme()

    print("Loading data...")
    df = load_data()
    print()

    print("Generating plots...")
    plot_h1_bar(df)
    plot_h1_distribution(df)
    plot_factual_gain_hist(df)
    plot_h2_grouped(df)
    plot_per_condition(df)
    plot_cmi_scatter(df)
    plot_improvement(df)
    plot_pipeline()
    plot_match_quality()
    plot_significance_summary(df)
    plot_ablation()

    print()
    print(f"All plots saved to {OUT_DIR}")
    print()
    print("THEME COLORS (for matching your poster template):")
    for name, color in THEME.items():
        print(f"  {name:20s} {color}")


if __name__ == "__main__":
    main()
