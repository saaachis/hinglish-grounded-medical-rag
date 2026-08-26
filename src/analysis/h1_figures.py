"""The H01 figure: how much of the grounding benefit is the metric, not the system.

This is the paper's central methodological claim in one picture. Each contrast is
the SAME generations scored twice:

    evidence (circular)  the retrieved case -- text the grounded arm was conditioned
                         on and the zero-shot arm never saw
    caption (unbiased)   the MMCQSD image description -- text NEITHER arm saw

Under the circular reference grounding always looks strong. Under the unbiased one
the effect shrinks on llama and goes NEGATIVE on gpt-oss-120b, because grounding
trades recall for precision and only sometimes profitably.

F1 is plotted, not `factual_support`. That metric is precision-only, so its optimum
is a one-word answer -- the constant answer "swelling" scores 0.719 against the
caption reference while the real system scores 0.153. Plotting it would rank
terseness, not quality.

Reads results/rescored/. Writes results/h1_figures/. CPU only, no recomputation
beyond bootstrap CIs.
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

SRC = Path("results/rescored")
OUT = Path("results/h1_figures")

CIRCULAR = "#eb6834"   # orange -- the flattering, circular reference
UNBIASED = "#2a78d6"   # blue   -- the honest one
INK = "#131C28"
MUTED = "#71808F"
GRID = "#CBD3DC"

N_BOOT = 5000
SEED = 42

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

#: (source file, arm, display label) -- ordered as they appear on the figure.
ROWS = [
    ("llama_oracle_n1165", "grounded", "llama-3.1-8b\noracle evidence"),
    ("gptoss120b", "oracle", "gpt-oss-120b\noracle evidence"),
    ("gptoss120b", "real", "gpt-oss-120b\nreal retrieval"),
]


def boot_ci(diff: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(diff)
    b = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def collect() -> pd.DataFrame:
    out = []
    for src, arm, label in ROWS:
        f = SRC / f"rescored_{src}.csv"
        if not f.exists():
            logger.warning("missing %s", f)
            continue
        d = pd.read_csv(f)
        for ref, refname in [("ev", "evidence (circular)"), ("cap", "caption (unbiased)")]:
            gc, zc = f"{arm}_{ref}_f1", f"zero_{ref}_f1"
            if gc not in d or zc not in d:
                continue
            s = d[[gc, zc]].dropna()
            if len(s) < 10:
                continue
            diff = (s[gc] - s[zc]).to_numpy()
            lo, hi = boot_ci(diff)
            out.append({"label": label, "reference": refname, "n": len(s),
                        "delta": diff.mean(), "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def figure(df: pd.DataFrame) -> None:
    labels = [l for _, _, l in ROWS if l in set(df.label)]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    off = 0.19

    for i, lab in enumerate(labels):
        for ref, colour, sign in [("evidence (circular)", CIRCULAR, +1),
                                  ("caption (unbiased)", UNBIASED, -1)]:
            r = df[(df.label == lab) & (df.reference == ref)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = i + sign * off
            ax.plot([r.lo, r.hi], [y, y], color=colour, lw=2.4,
                    solid_capstyle="round", zorder=3)
            ax.plot(r.delta, y, "o", color=colour, ms=7.5,
                    markeredgecolor="white", markeredgewidth=1.1, zorder=4)
            ax.text(r.hi + 0.006, y, f"{r.delta:+.3f}", va="center",
                    fontsize=8.5, color=colour, fontweight="bold")

    ax.axvline(0, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Grounding effect on concept F1  (grounded − zero-shot, 95% CI)")
    ax.set_title("The apparent grounding benefit depends on what you score against",
                 fontsize=11, loc="left", color=INK, pad=42)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=CIRCULAR, lw=2.4, marker="o", ms=6,
               markeredgecolor="white", label="evidence the model was given (circular)"),
        Line2D([], [], color=UNBIASED, lw=2.4, marker="o", ms=6,
               markeredgecolor="white", label="image description (neither arm saw it)")],
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.16),
        ncol=1, handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_h1_reference_effect.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "h1_reference_effect.csv", index=False)
    print(df.to_string(index=False))
    figure(df)
    logger.info("wrote %s", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
