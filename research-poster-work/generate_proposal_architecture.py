"""
Four-stage evidence-first RAG architecture diagram (proposal / deck style).

Pastel columns similar to the original team diagram, but aligned to the
implemented pipeline: MMCQSD-style Hinglish queries, MultiCaRe evidence,
LaBSE + FAISS + adaptive truncation, Groq Llama-3.1-8B, text-grounded core.

Run from repo root:
  python research-poster-work/generate_proposal_architecture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).parent
DPI = 300

# Pastel stage fills (warm → cool → green → cream), borders slightly deeper
STAGES = [
    {
        "title": "1. Inputs & evidence index",
        "fill": "#E8D4BC",
        "edge": "#8B6914",
        "items": [
            "Hinglish query — patient-style code-mixed text\n(MMCQSD distribution; ad hoc input in demo)",
            "MultiCaRe — English clinical case narratives;\nfiltered by specialty; offline LaBSE encode\n→ FAISS index for runtime search",
        ],
    },
    {
        "title": "2. Query encoding",
        "fill": "#C8DCEF",
        "edge": "#3A6EA5",
        "items": [
            "LaBSE (sentence-transformers)\n— cross-lingual sentence embeddings\n(768-d), shared space with evidence",
        ],
    },
    {
        "title": "3. Evidence retrieval",
        "fill": "#C8E6C9",
        "edge": "#2E7D32",
        "items": [
            "FAISS inner-product top-k\nover indexed passages",
            "Adaptive truncation — trim tail\nwhere similarity drops sharply\n(MMed-RAG–style context control)",
        ],
    },
    {
        "title": "4. Grounded generation & evaluation",
        "fill": "#F5E6B8",
        "edge": "#B8860B",
        "items": [
            "Prompt injects ranked excerpts;\nLlama-3.1-8B-Instant via Groq API",
            "Hinglish explanation — clinical\ndecision support, not diagnosis",
            "Evaluation (study): evidence-aligned\nfactual support; hallucination score;\nCMI tertiles — H1 / H2",
        ],
    },
]


def _rounded_box(ax, x, y, w, h, facecolor, edgecolor, lw=1.2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
    )
    ax.add_patch(p)
    return p


def main() -> None:
    fig_w, fig_h = 13.0, 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FAF8F3")
    ax.set_facecolor("#FAF8F3")

    ax.text(
        0.5, 0.96,
        "Evidence-first Retrieval-Augmented Generation (RAG) — text-grounded pipeline",
        ha="center", va="top", fontsize=13, fontweight="bold", color="#1A2A4A",
        fontfamily="serif",
    )

    n = len(STAGES)
    gap = 0.012
    col_w = (1.0 - 2 * 0.04 - (n - 1) * gap) / n
    x0 = 0.04
    y_main = 0.10
    h_main = 0.72

    for i, st in enumerate(STAGES):
        x = x0 + i * (col_w + gap)
        _rounded_box(ax, x, y_main, col_w, h_main, st["fill"], st["edge"], lw=1.4)
        ax.text(
            x + col_w / 2, y_main + h_main - 0.028, st["title"],
            ha="center", va="top", fontsize=9.5, fontweight="bold",
            color="#1A1A1A", fontfamily="serif",
        )
        y_text = y_main + h_main - 0.10
        block = "\n\n".join(st["items"])
        ax.text(
            x + col_w / 2, y_text, block,
            ha="center", va="top", fontsize=7.8, color="#222222",
            linespacing=1.35, fontfamily="sans-serif",
        )

        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + col_w + gap * 0.15, y_main + h_main * 0.48),
                    (x + col_w + gap * 0.85, y_main + h_main * 0.48),
                    arrowstyle="-|>", mutation_scale=14, linewidth=1.3,
                    color="#4A4A4A", zorder=5,
                )
            )

    ax.text(
        0.5, 0.035,
        "Core system: text retrieval + text-conditioned generation. Multimodal evidence (e.g. imaging) is a planned extension of the same retrieve-then-generate design.",
        ha="center", va="center", fontsize=7.5, style="italic", color="#444444",
        fontfamily="serif",
    )

    out = OUT_DIR / "15_evidence_first_rag_architecture_updated.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.patch.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
