"""System architecture diagrams — engineering block-diagram style (matplotlib).

Box heights are computed from wrapped line counts so text stays inside rectangles.

Run from repository root:
  python research-poster-work/generate_architecture_diagrams.py
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT_DIR = Path(__file__).parent
DPI = 300

C = {
    "edge": "#456F7A",
    "edge_strong": "#588894",
    "fill": "#FAFCFC",
    "fill_alt": "#EEF4F5",
    "store_fill": "#E6EEF0",
    "dash": "#7BA3B8",
    "text": "#1e3338",
    "text_dim": "#4a5f66",
    "arrow": "#2F4858",
    "white": "#FFFFFF",
}

FONT_TITLE = "DejaVu Sans"
FONT_BODY = "DejaVu Sans Mono"
FS_CAP = 7.2
FS_BODY_V = 5.15
FS_BODY_H = 5.05
FS_TITLE = 9.0
FS_SMALL = 5.0


def _wrap(text: str, width: int) -> list[str]:
    parts = textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=True)
    return parts if parts else [""]


def _box_draw(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    subtitle: str | None,
    lines: list[str],
    *,
    fill=None,
    edge=None,
    fs_body: float,
    pad_top: float = 0.11,
    pad_bot: float = 0.1,
    line_skip: float | None = None,
):
    """Draw rectangle and centered text; text block sized to fit inside (pad_top/pad_bot)."""
    fill = fill or C["fill"]
    edge = edge or C["edge_strong"]
    if line_skip is None:
        line_skip = max(0.145, (h - pad_top - pad_bot - (0.24 if subtitle else 0)) / max(len(lines), 1))

    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.05",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.1,
        zorder=2,
    )
    ax.add_patch(patch)

    cy = y + h - pad_top
    if subtitle:
        ax.text(
            x + w / 2, cy, subtitle,
            ha="center", va="top", fontsize=FS_CAP, fontweight="bold",
            color=C["text"], family=FONT_TITLE, zorder=3,
        )
        cy -= 0.24
    for i, line in enumerate(lines):
        ax.text(
            x + w / 2, cy - i * line_skip, line,
            ha="center", va="top", fontsize=fs_body, color=C["text_dim"],
            family=FONT_BODY, zorder=3,
        )
    return patch


def _content_height(n_lines: int, has_subtitle: bool, pad_top: float, pad_bot: float, line_skip: float) -> float:
    sub = 0.24 if has_subtitle else 0.0
    return pad_top + sub + max(n_lines, 1) * line_skip + pad_bot


def _arrow_v(ax, x, y_start_hi, y_end_lo):
    """Arrow from y_start_hi (larger y, upper on page) down to y_end_lo (smaller y)."""
    ax.annotate(
        "",
        xy=(x, y_end_lo),
        xytext=(x, y_start_hi),
        arrowprops=dict(
            arrowstyle="-|>", color=C["arrow"], lw=1.1,
            shrinkA=2, shrinkB=2, mutation_scale=10, fill=True,
        ),
        zorder=1,
    )


def _arrow_h(ax, x_left, x_right, y):
    ax.annotate(
        "",
        xy=(x_right, y),
        xytext=(x_left, y),
        arrowprops=dict(
            arrowstyle="-|>", color=C["arrow"], lw=1.1,
            shrinkA=2, shrinkB=2, mutation_scale=10, fill=True,
        ),
        zorder=1,
    )


def draw_vertical():
    # Wider canvas: main column + evidence column (no text collision)
    fig_w, fig_h = 6.45, 10.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    xlim_r = 10.6
    ax.set_xlim(0, xlim_r)
    ax.set_ylim(0, 15.6)
    ax.axis("off")
    fig.patch.set_facecolor(C["white"])
    ax.set_facecolor(C["white"])

    ax.text(
        xlim_r / 2, 15.35,
        "Grounded cross-lingual RAG — system architecture",
        ha="center", va="top", fontsize=FS_TITLE, fontweight="bold",
        color=C["text"], family=FONT_TITLE,
    )
    ax.text(
        xlim_r / 2, 14.88,
        "Hinglish query → English case evidence → Hinglish answer (clinical CDS prototype)",
        ha="center", va="top", fontsize=FS_SMALL + 0.15, color=C["text_dim"], family=FONT_BODY,
    )

    x, w_main = 0.45, 5.35
    gap = 0.22
    wrap_main = 38
    pad_top, pad_bot = 0.11, 0.11
    line_skip = 0.158

    raw_steps = [
        {
            "sub": "1  Query",
            "raw": [
                "Roman-script Hinglish patient text (MMCQSD-style).",
                "3,015 LaBSE–MultiCaRe matched pairs; eval n=1,165.",
            ],
            "fill": C["fill"],
            "edge": C["edge_strong"],
        },
        {
            "sub": "2  Cross-lingual encoder",
            "raw": [
                "sentence-transformers / LaBSE (max_seq_len=128).",
                "768-d L2-normalized vector per query.",
            ],
            "fill": C["fill_alt"],
            "edge": C["edge_strong"],
        },
        {
            "sub": "3  Dense retrieval",
            "raw": [
                "FAISS IndexFlatIP (inner product on unit vectors).",
                "top-k over 10,000 pre-encoded MultiCaRe passages.",
            ],
            "fill": C["fill"],
            "edge": C["edge_strong"],
        },
        {
            "sub": "4  Context assembly",
            "raw": [
                "Ranked passages → adaptive truncation (score-drop gate).",
                "Prompt template: system rules + delimiter-separated evidence.",
            ],
            "fill": C["fill_alt"],
            "edge": C["edge_strong"],
        },
        {
            "sub": "5  Decoder LM",
            "raw": [
                "Llama-3.1-8B-Instruct via Groq API (cloud inference).",
                "Evidence-conditioned generation; same backbone for ablations.",
            ],
            "fill": C["fill"],
            "edge": C["edge_strong"],
        },
        {
            "sub": "6  Downstream output",
            "raw": [
                "Hinglish explanation text returned to client application.",
            ],
            "fill": C["fill_alt"],
            "edge": C["edge_strong"],
        },
        {
            "sub": "7  Automated evaluation (offline)",
            "raw": [
                "Concept lexicon + negation patterns.",
                "Factual-support ratio vs. evidence.",
                "Hallucination-style ratio vs. evidence.",
                "Paired grounded vs. zero-shot comparison.",
            ],
            "fill": C["fill_alt"],
            "edge": C["edge"],
        },
    ]

    # Stack from top: y_next_top = upper edge of next box to place
    y_next_top = 14.58
    placed = []
    for spec in raw_steps:
        lines = []
        for t in spec["raw"]:
            lines.extend(_wrap(t, wrap_main))
        h = _content_height(len(lines), True, pad_top, pad_bot, line_skip)
        y_b = y_next_top - h
        _box_draw(
            ax, x, y_b, w_main, h, spec["sub"], lines,
            fill=spec["fill"], edge=spec["edge"], fs_body=FS_BODY_V,
            pad_top=pad_top, pad_bot=pad_bot, line_skip=line_skip,
        )
        placed.append({"y": y_b, "h": h})
        y_next_top = y_b - gap

    cx = x + w_main / 2
    for i in range(len(placed) - 1):
        ya, ha = placed[i]["y"], placed[i]["h"]
        yb, hb = placed[i + 1]["y"], placed[i + 1]["h"]
        # Downward: upper box bottom (ya) → lower box top (yb + hb)
        _arrow_v(ax, cx, ya - 0.02, yb + hb + 0.02)

    # Evidence corpus: right column, aligned to boxes 3–4 only (indices 2 and 3)
    p3 = placed[2]
    p4 = placed[3]
    top_e = p3["y"] + p3["h"]
    bot_e = p4["y"]
    sx = x + w_main + 0.32
    sw = xlim_r - sx - 0.35
    sh = top_e - bot_e
    store = FancyBboxPatch(
        (sx, bot_e), sw, sh,
        boxstyle="round,pad=0.008,rounding_size=0.05",
        facecolor=C["store_fill"],
        edgecolor=C["edge"],
        linewidth=1.0,
        linestyle=(0, (5, 3)),
        zorder=2,
    )
    ax.add_patch(store)

    store_title = "Evidence corpus"
    store_raw = [
        "MultiCaRe — PMC open-access case narratives.",
        "61,316 specialty-filtered cases; 18 MMCQSD condition groups.",
        "Offline index build; query-time condition filter feeds FAISS.",
    ]
    wrap_s = max(22, int(sw * 5.8))
    s_lines: list[str] = []
    for t in store_raw:
        s_lines.extend(_wrap(t, wrap_s))
    s_pad_t, s_pad_b = 0.1, 0.1
    s_skip = min(0.152, (sh - s_pad_t - s_pad_b - 0.24) / max(len(s_lines), 1))
    scy = bot_e + sh - s_pad_t
    ax.text(sx + sw / 2, scy, store_title, ha="center", va="top", fontsize=FS_CAP, fontweight="bold",
            color=C["text"], family=FONT_TITLE, zorder=3)
    scy -= 0.23
    for i, line in enumerate(s_lines):
        ax.text(sx + sw / 2, scy - i * s_skip, line, ha="center", va="top", fontsize=FS_BODY_V - 0.15,
                color=C["text_dim"], family=FONT_BODY, zorder=3)

    # Dashed link into retrieval box (box 3)
    mid_y = p3["y"] + p3["h"] * 0.5
    ax.plot([x + w_main, sx], [mid_y, mid_y], color=C["dash"], lw=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text((x + w_main + sx) / 2, mid_y + 0.14, "indexed from", ha="center", fontsize=FS_SMALL - 0.3,
            color=C["dash"], family=FONT_BODY, style="italic", zorder=3)

    fig.savefig(
        OUT_DIR / "15_architecture_workflow_vertical.png",
        dpi=DPI, bbox_inches="tight", facecolor=C["white"], pad_inches=0.14,
    )
    plt.close(fig)
    print("  Saved 15_architecture_workflow_vertical.png")


def draw_horizontal():
    fig_w, fig_h = 14.2, 4.55
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 5.1)
    ax.axis("off")
    fig.patch.set_facecolor(C["white"])

    ax.text(
        10.5, 4.85,
        "Grounded cross-lingual RAG — deployment view (main components)",
        ha="center", va="top", fontsize=FS_TITLE, fontweight="bold",
        color=C["text"], family=FONT_TITLE,
    )

    specs = [
        ("Query", "Hinglish clinical text (MMCQSD-style).", [
            "Roman-script Hindi–English codeswitch.",
        ]),
        ("Encoder", "LaBSE sentence encoder", [
            "768-d L2-normalized embedding.",
            "max_seq_len=128 (memory-safe batching).",
        ]),
        ("Retriever", "FAISS dense search", [
            "IndexFlatIP · inner product on unit vectors.",
            "10K indexed passages; top-k + 18-way condition filter.",
            "Corpus: MultiCaRe — 61,316 PMC-filtered cases.",
        ]),
        ("Context", "Prompt assembly", [
            "Adaptive truncation via retrieval score drop.",
            "Template: rules + delimiter-separated evidence.",
        ]),
        ("LLM", "Llama-3.1-8B-Instruct", [
            "Groq-hosted inference (cloud).",
            "Evidence-conditioned decoding.",
        ]),
        ("Output", "Hinglish generation", [
            "Patient-facing explanation text.",
            "Offline eval: concept overlap vs. evidence.",
        ]),
    ]
    n = len(specs)
    margin_l, margin_r = 0.45, 0.45
    gap = 0.26
    usable = 21.0 - margin_l - margin_r
    bw = (usable - (n - 1) * gap) / n
    x0 = margin_l
    wrap_w = max(20, int(bw * 5.9))

    pad_top, pad_bot = 0.1, 0.1
    line_skip = 0.152

    blocks = []
    for sub, line1, extra in specs:
        raw_lines = [line1] + list(extra)
        lines: list[str] = []
        for L in raw_lines:
            lines.extend(_wrap(L, wrap_w))
        h_need = _content_height(len(lines), True, pad_top, pad_bot, line_skip)
        blocks.append((sub, lines, h_need))

    h_max = max(b[2] for b in blocks) + 0.08
    yb = 0.42

    for i, (sub, lines, _) in enumerate(blocks):
        fill = C["fill_alt"] if i % 2 else C["fill"]
        xi = x0 + i * (bw + gap)
        _box_draw(
            ax, xi, yb, bw, h_max, sub, lines,
            fill=fill, edge=C["edge_strong"], fs_body=FS_BODY_H,
            pad_top=pad_top, pad_bot=pad_bot, line_skip=line_skip,
        )

    y_mid = yb + h_max * 0.55
    for i in range(n - 1):
        xL = x0 + i * (bw + gap) + bw
        xR = x0 + (i + 1) * (bw + gap)
        _arrow_h(ax, xL + 0.06, xR - 0.06, y_mid)

    fig.savefig(
        OUT_DIR / "16_architecture_workflow_horizontal.png",
        dpi=DPI, bbox_inches="tight", facecolor=C["white"], pad_inches=0.12,
    )
    plt.close(fig)
    print("  Saved 16_architecture_workflow_horizontal.png")


def main():
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    draw_vertical()
    draw_horizontal()
    print(f"Done. Files in {OUT_DIR}")


if __name__ == "__main__":
    main()
