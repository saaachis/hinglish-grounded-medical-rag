"""Publication figures for the retrieval results (Table 1 and the H04 asymmetry).

Two figures, both 300-DPI PNG plus vector PDF as the SN Computer Science template
requires:

  fig1  Table 1 as grouped bars -- four retrieval systems x two query conditions.
  fig2  The code-mixing penalty per system, with bootstrap CIs. This is the
        paper's argument for the architecture and deserves its own figure: BM25
        wins on English and collapses on Hinglish while dense barely moves, so
        the penalty is 4.4x larger for lexical retrieval.

Reads results/retrieval_v2/. No recomputation. CPU only.
Writes results/retrieval_figures/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SRC = Path("results/retrieval_v2")
OUT = Path("results/retrieval_figures")

# Palette validated for CVD separation and contrast against a light surface.
HINGLISH = "#2a78d6"   # blue  -- the deployed, code-mixed path
ENGLISH = "#eb6834"    # orange -- the English ceiling
INK = "#131C28"
MUTED = "#71808F"
GRID = "#CBD3DC"
FLOOR = "#9FADBB"

RANDOM_FLOOR = 0.0626

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

LABELS = {
    "Hybrid-RRF": "Hybrid (RRF)",
    "LaBSE-passages": "LaBSE\n(passages)",
    "BM25-full": "BM25",
    "TFIDF-full": "TF-IDF",
}


def fig_table1(metrics: pd.DataFrame) -> None:
    piv = metrics.pivot(index="system", columns="variant", values="recall@1")
    order = piv.sort_values("Q1_hinglish", ascending=False).index.tolist()
    x = np.arange(len(order))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    q1 = [piv.loc[s, "Q1_hinglish"] for s in order]
    q2 = [piv.loc[s, "Q2_english_question"] for s in order]

    ax.bar(x - w / 2, q1, w, color=HINGLISH, label="Hinglish query (deployed)", zorder=3)
    ax.bar(x + w / 2, q2, w, color=ENGLISH, label="English question", zorder=3)

    for xi, (a, b) in enumerate(zip(q1, q2)):
        ax.text(xi - w / 2, a + 0.004, f"{a:.3f}", ha="center", fontsize=8,
                color=HINGLISH, fontweight="bold")
        ax.text(xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8,
                color=ENGLISH, fontweight="bold")

    ax.axhline(RANDOM_FLOOR, color=FLOOR, lw=1.2, ls=(0, (5, 3)), zorder=2)
    ax.text(len(order) - 0.4, RANDOM_FLOOR + 0.003, "random floor",
            fontsize=7.5, color=MUTED, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(s, s) for s in order])
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0, max(max(q1), max(q2)) * 1.22)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Retrieval quality by system and query language",
                 fontsize=11, loc="left", color=INK, pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_table1.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_penalty(tests: pd.DataFrame) -> None:
    """The asymmetry figure: how much each system loses to code-mixing."""
    t = tests.sort_values("delta").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 2.3))

    for i, r in t.iterrows():
        lex = r.system.startswith(("BM25", "TFIDF"))
        colour = ENGLISH if lex else HINGLISH
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=colour, lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot(r.delta, i, "o", color=colour, ms=8, markeredgecolor="white",
                markeredgewidth=1.2, zorder=4)
        ax.text(r.ci_hi + 0.004, i, f"{r.delta:+.4f}   p = {r.mcnemar_p:.2g}",
                va="center", fontsize=8.5, color=colour, fontweight="bold")

    ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([LABELS.get(s, s).replace("\n", " ") for s in t.system])
    ax.set_xlabel("Recall@1 lost to code-mixing  (English − Hinglish, 95% CI)")
    ax.set_xlim(-0.008, t.ci_hi.max() * 1.42)
    ax.set_title("Lexical retrieval is far more damaged by code-mixing than dense",
                 fontsize=11, loc="left", color=INK, pad=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=HINGLISH, lw=2.4, marker="o", ms=6,
                              markeredgecolor="white", label="dense"),
                       Line2D([], [], color=ENGLISH, lw=2.4, marker="o", ms=6,
                              markeredgecolor="white", label="lexical")],
              frameon=False, fontsize=8, loc="lower right",
              bbox_to_anchor=(1.0, -0.30), ncol=2, handlelength=1.6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_penalty.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = pd.read_csv(SRC / "retrieval_v2_metrics.csv")
    tests = pd.read_csv(SRC / "h4_v2_tests.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_table1(metrics)
    fig_penalty(tests)
    logger.info("wrote %s", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
