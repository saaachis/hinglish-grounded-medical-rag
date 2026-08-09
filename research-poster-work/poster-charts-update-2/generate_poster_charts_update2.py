"""Poster charts v2 — matplotlib / seaborn only.

Canvas: 22 cm (breadth) × 16 cm (height) @ 300 dpi.

`main()` writes only the final poster figures (22×16 cm, black frame ≈1.5 px at `DPI`, no in-chart titles):
  Grounding_Gains.png, Dataset_Scale.png, Robustness_Tests.png,
  Hypothesis_Bars.png, Hypothesis_Scaled.png, Design_Heatmap.png, Challenge_Mitigation.png

Run:
  python research-poster-work/poster-charts-update-2/generate_poster_charts_update2.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent

POSTER_W_CM = 22.0  # breadth
POSTER_H_CM = 16.0  # length
FIG_W_IN = POSTER_W_CM / 2.54
FIG_H_IN = POSTER_H_CM / 2.54
DPI = 300
# Figure outline: ~1.5 device pixels on the saved PNG (matplotlib linewidth is in points)
FIG_BORDER_PT = 1.5 * 72.0 / DPI

PAL = {
    "bg": "#FFFFFF",
    "panel": "#F8FAFC",
    "grid": "#E2E8F0",
    "text": "#0F172A",
    "muted": "#64748B",
    "edge": "#94A3B8",
    "a": "#5B8FC7",
    "b": "#94C794",
    "c": "#E8A598",
    "d": "#B8A9DC",
    "ref": "#F59E0B",
    # Showcase palette (distinct, print-friendly)
    "zs": "#E07A5F",
    "gr": "#3D9970",
    "delta": "#457B9D",
    "tert1": "#81B29A",
    "tert2": "#3A86FF",
    "sig": "#264653",
    "ns": "#ADB5BD",
}


def _style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.0,
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.facecolor": PAL["panel"],
            "figure.facecolor": PAL["bg"],
            "axes.edgecolor": PAL["edge"],
            "axes.labelcolor": PAL["text"],
            "text.color": PAL["text"],
            "xtick.color": PAL["text"],
            "ytick.color": PAL["text"],
            "grid.color": PAL["grid"],
            "grid.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.98,
            "legend.edgecolor": PAL["edge"],
        },
    )


def _save(fig: plt.Figure, name: str, *, pad_inches: float = 0.03) -> None:
    path = HERE / f"{name}.png"
    fig.patch.set_facecolor(PAL["bg"])
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(FIG_BORDER_PT)
    fig.savefig(
        path,
        dpi=DPI,
        facecolor=PAL["bg"],
        edgecolor="black",
        bbox_inches=None,
        pad_inches=pad_inches,
    )
    plt.close(fig)
    print(f"  Saved {path.name} ({POSTER_W_CM:.0f} × {POSTER_H_CM:.0f} cm @ {DPI} dpi)")


def _poster_rc() -> dict:
    """Slightly larger type for 22×16 cm poster viewing distance."""
    return {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
    }


def plot_visual_01_grounding_evidence() -> None:
    """H1: dumbbell (zero-shot vs grounded) + paired improvement magnitudes."""
    with plt.rc_context(_poster_rc()):
        fig, (ax_l, ax_r) = plt.subplots(
            1,
            2,
            figsize=(FIG_W_IN, FIG_H_IN),
            dpi=DPI,
            facecolor=PAL["bg"],
            gridspec_kw={"width_ratios": [1.38, 0.88], "wspace": 0.30},
        )
        fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.20)

        metrics = ["Factual support", "Hallucination\n(lower is better)"]
        zs = np.array([0.319, 0.500])
        gr = np.array([0.554, 0.280])
        y = np.arange(len(metrics))
        x_pad = 0.028
        for yi, z, g in zip(y, zs, gr):
            ax_l.plot([z, g], [yi, yi], color=PAL["edge"], lw=2.4, solid_capstyle="round", zorder=1)
            ax_l.scatter(
                [z], [yi], s=190, c=PAL["zs"], edgecolors=PAL["text"], linewidths=0.9, zorder=3, label="_zs",
            )
            ax_l.scatter(
                [g], [yi], s=190, c=PAL["gr"], edgecolors=PAL["text"], linewidths=0.9, zorder=3, label="_gr",
            )
            ax_l.text(z - x_pad, yi, f"{z:.3f}", ha="right", va="center", fontsize=9, color=PAL["text"])
            ax_l.text(g + x_pad, yi, f"{g:.3f}", ha="left", va="center", fontsize=9, color=PAL["text"])

        ax_l.set_yticks(y)
        ax_l.set_yticklabels(metrics)
        ax_l.set_xlabel("Mean score (paired items, n = 1,165)")
        ax_l.set_xlim(0.22, 0.62)
        ax_l.set_ylim(-0.55, 1.55)
        ax_l.grid(True, axis="x", alpha=0.45)

        h_zs = mlines.Line2D(
            [], [], color=PAL["zs"], marker="o", linestyle="None", markersize=11, markeredgecolor=PAL["text"], markeredgewidth=0.9,
            label="Zero-shot",
        )
        h_gr = mlines.Line2D(
            [], [], color=PAL["gr"], marker="o", linestyle="None", markersize=11, markeredgecolor=PAL["text"], markeredgewidth=0.9,
            label="Grounded RAG",
        )

        delta_labs = ["Factual\npaired Δ", "Hallucination\npaired Δ"]
        deltas = np.array([0.235, 0.220])
        colors_d = [PAL["gr"], PAL["delta"]]
        bars = ax_r.barh(np.arange(len(delta_labs)), deltas, color=colors_d, edgecolor=PAL["text"], linewidth=0.7, height=0.52)
        ax_r.set_yticks(np.arange(len(delta_labs)))
        ax_r.set_yticklabels(delta_labs)
        ax_r.set_xlabel("Mean improvement (grounded − zero-shot)")
        ax_r.set_xlim(0, 0.30)
        ax_r.bar_label(bars, fmt="{:+.3f}", padding=6, fontsize=10, fontweight="bold")
        ax_r.grid(True, axis="x", alpha=0.45)

        fig.legend(
            handles=[h_zs, h_gr],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor=PAL["edge"],
        )

    _save(fig, "Grounding_Gains")


def plot_visual_02_data_corpus() -> None:
    """Horizontal log-scale bar chart of dataset / artifact counts (no inset)."""
    with plt.rc_context(_poster_rc()):
        fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor=PAL["bg"])
        # Balanced margins (plot nearer horizontal center); left still fits long y-labels
        fig.subplots_adjust(left=0.32, right=0.91, top=0.92, bottom=0.18)

        labels = [
            "MultiCaRe (filtered)",
            "FAISS demo index",
            "MMCQSD queries",
            "Aligned query–evidence pairs",
            "Paired evaluation set",
            "Phase 6 ablation",
        ]
        vals = np.array([61316, 10000, 3015, 3015, 1165, 401], dtype=float)
        order = np.argsort(vals)
        labels = [labels[i] for i in order]
        vals = vals[order]
        # Sequential blues (sorted ascending: lighter → deeper navy for larger counts)
        bar_colors = sns.color_palette("Blues", n_colors=len(vals))
        ax.barh(labels, vals, color=bar_colors, edgecolor=PAL["text"], linewidth=0.55)
        ax.set_xscale("log")
        ax.set_xlabel("Count (log₁₀ scale on axis)")
        ax.tick_params(axis="y", labelsize=8.8, pad=3)
        hi, lo = float(np.max(vals)), float(np.min(vals))
        x_min = max(80.0, lo * 0.55)
        # Log x: linear factor is not enough — text extends ~constant width in log-decades past anchor
        label_anchor = hi * 1.06
        x_max = float(10 ** (np.log10(max(label_anchor, 1.0)) + 0.34))
        ax.set_xlim(x_min, x_max)
        for i, v in enumerate(vals):
            ax.text(
                v * 1.06,
                i,
                f"{int(v):,}",
                va="center",
                ha="left",
                fontsize=9,
                color=PAL["text"],
            )

    _save(fig, "Dataset_Scale")


def plot_visual_03_robustness_inference() -> None:
    """H2 tertile gains (seaborn) + significance lollipops; shared hue in legend."""
    with plt.rc_context(_poster_rc()):
        # Blue-forward palette (print-friendly); threshold line kept high-contrast
        col_factual = "#2563EB"
        col_halluc = "#93C5FD"
        col_sig = "#1D4ED8"
        col_ns = "#CBD5E1"
        col_thr = "#C2410C"

        fig, (ax_l, ax_r) = plt.subplots(
            1,
            2,
            figsize=(FIG_W_IN, FIG_H_IN),
            dpi=DPI,
            facecolor=PAL["bg"],
            gridspec_kw={"width_ratios": [1.08, 0.92], "wspace": 0.42},
        )
        fig.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.24)

        tert = ["Low CMI", "Med CMI", "High CMI"]
        rows = []
        for t, f, h in zip(tert, [0.202, 0.241, 0.260], [0.206, 0.208, 0.245]):
            rows.append({"tertile": t, "Paired Δ": f, "metric": "Factual gain"})
            rows.append({"tertile": t, "Paired Δ": h, "metric": "Halluc. reduction"})
        df = pd.DataFrame(rows)
        sns.barplot(
            data=df,
            x="tertile",
            y="Paired Δ",
            hue="metric",
            ax=ax_l,
            palette={"Factual gain": col_factual, "Halluc. reduction": col_halluc},
            edgecolor=PAL["text"],
            linewidth=0.65,
            saturation=0.92,
        )
        ax_l.set_xlabel("Code-mixing (CMI) tertile")
        ax_l.set_ylabel("Mean paired Δ (same n = 1,165)")
        ax_l.set_ylim(0, 0.32)
        for c in ax_l.containers:
            ax_l.bar_label(c, fmt="%.3f", padding=2, fontsize=7.5)
        leg = ax_l.get_legend()
        if leg is not None:
            leg.remove()

        tests = [
            "Wilcoxon\n(factual)",
            "Wilcoxon\n(hallucination)",
            "Kruskal–Wallis\n(factual gain)",
            "Spearman\n(CMI vs gain)",
        ]
        pvals = np.array([3.09e-64, 5.33e-51, 0.144, 0.016], dtype=float)
        neglog = np.array([-np.log10(max(p, 1e-300)) for p in pvals])
        y = np.arange(len(tests), dtype=float)
        sig_mask = pvals < 0.05
        colors = [col_sig if s else col_ns for s in sig_mask]
        ax_r.hlines(y, 0, neglog, colors=colors, linewidth=3, alpha=0.88)
        ax_r.scatter(neglog, y, s=120, c=colors, edgecolors=PAL["text"], linewidths=0.8, zorder=3)
        thr = -np.log10(0.05)
        ax_r.axvline(thr, color=col_thr, linestyle="--", linewidth=1.8, label="α = 0.05")
        ax_r.set_yticks(y)
        ax_r.set_yticklabels(tests)
        ax_r.set_xlabel("Evidence against null (−log₁₀ p)")
        xmax = max(float(neglog.max()), thr) * 1.08
        ax_r.set_xlim(0, xmax)
        ax_r.set_ylim(-0.55, len(tests) - 0.45)
        for yi, nl, p in zip(y, neglog, pvals):
            if p < 1e-50:
                lbl = "p ≪ 0.001"
            elif p >= 0.001:
                lbl = f"p = {p:.3g}"
            else:
                lbl = f"p = {p:.2g}"
            ax_r.text(
                nl,
                yi + 0.22,
                lbl,
                va="bottom",
                ha="center",
                fontsize=7.8,
                color=PAL["text"],
                fontweight="medium",
                zorder=4,
            )

        h_gain = mpatches.Patch(facecolor=col_factual, edgecolor=PAL["text"], label="Factual gain (bars)")
        h_hall = mpatches.Patch(facecolor=col_halluc, edgecolor=PAL["text"], label="Halluc. reduction (bars)")
        h_sig = mlines.Line2D([], [], color=col_sig, marker="o", linestyle="None", markersize=9, label="p < 0.05 (tests)")
        h_ns = mlines.Line2D([], [], color=col_ns, marker="o", linestyle="None", markersize=9, label="p ≥ 0.05 (tests)")
        h_line = mlines.Line2D([], [], color=col_thr, linestyle="--", linewidth=1.8, label="α = 0.05")

        fig.legend(
            handles=[h_gain, h_hall, h_sig, h_ns, h_line],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=True,
            fancybox=False,
            edgecolor=PAL["edge"],
        )

    _save(fig, "Robustness_Tests")


def plot_hypotheses_barplot() -> None:
    """H1 and H2 bar magnitudes; legend top-right."""
    with plt.rc_context(_poster_rc()):
        # Light, still vivid (poster-friendly); dark edges keep bars readable on grid
        col_delta = "#5EB8E8"
        col_second = "#FF9E6D"
        edge = "#1A1A1A"

        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(FIG_W_IN, FIG_H_IN),
            dpi=DPI,
            facecolor=PAL["bg"],
            gridspec_kw={"wspace": 0.36},
        )
        fig.subplots_adjust(left=0.11, right=0.905, top=0.90, bottom=0.24)

        # --- H1 ---
        h1_x = np.arange(2)
        h1_vals = np.array([0.235, 0.576])
        h1_labs = ["Paired factual Δ\n(grounded − zero-shot)", "Cohen's d\n(factual)"]
        b1 = ax1.bar(h1_x, h1_vals, width=0.55, color=[col_delta, col_second], edgecolor=edge, linewidth=0.75)
        ax1.bar_label(b1, fmt="%.3f", padding=5, fontsize=10, fontweight="bold", color=PAL["text"])
        ax1.set_xticks(h1_x)
        ax1.set_xticklabels(h1_labs, fontsize=9)
        ax1.tick_params(axis="x", pad=6)
        ax1.set_ylabel("Magnitude")
        ax1.set_ylim(0, 0.82)

        # --- H2 ---
        h2_x = np.arange(2)
        h2_vals = np.array([0.019, 0.144])
        h2_labs = ["Grounded factual\nrange (max − min\nacross CMI tertiles)", "KW p-value\n(gain differs\nacross tertiles?)"]
        b2 = ax2.bar(h2_x, h2_vals, width=0.55, color=[col_delta, col_second], edgecolor=edge, linewidth=0.75)
        ax2.bar_label(b2, fmt="%.3f", padding=5, fontsize=10, fontweight="bold", color=PAL["text"])
        ax2.set_xticks(h2_x)
        ax2.set_xticklabels(h2_labs, fontsize=8.5)
        ax2.tick_params(axis="x", pad=6)
        ax2.set_ylabel("Magnitude (spread or p)")
        ax2.set_ylim(0, 0.44)

        h_leg = [
            mpatches.Patch(facecolor=col_delta, edgecolor=edge, label="First bar: Δ (H1) or spread (H2)"),
            mpatches.Patch(facecolor=col_second, edgecolor=edge, label="Second bar: Cohen's d (H1) or KW p (H2)"),
        ]
        # H2 has spare vertical space (ylim ≫ max bar); legend in panel top-right
        fig.legend(
            handles=h_leg,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.98),
            bbox_transform=ax2.transAxes,
            frameon=True,
            fancybox=False,
            edgecolor=PAL["edge"],
            fontsize=7.0,
        )

    _save(fig, "Hypothesis_Bars", pad_inches=0.09)


def plot_hypotheses_barplot_2() -> None:
    """Same H1/H2 metrics as Hypothesis_Bars, but per-panel max-normalized bar heights.

    Colors and edges match `plot_hypotheses_barplot`; legend is centered at the bottom (scaled figure only).
    """
    with plt.rc_context(_poster_rc()):
        col_delta = "#5EB8E8"
        col_second = "#FF9E6D"
        edge = "#1A1A1A"

        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(FIG_W_IN, FIG_H_IN),
            dpi=DPI,
            facecolor=PAL["bg"],
            gridspec_kw={"wspace": 0.36},
        )
        fig.subplots_adjust(left=0.11, right=0.905, top=0.90, bottom=0.25)

        h1_x = np.arange(2)
        h1_vals = np.array([0.235, 0.576])
        h1_plot = h1_vals / np.max(h1_vals)
        h1_labs = ["Paired factual Δ\n(grounded − zero-shot)", "Cohen's d\n(factual)"]
        b1 = ax1.bar(h1_x, h1_plot, width=0.55, color=[col_delta, col_second], edgecolor=edge, linewidth=0.75)
        ax1.bar_label(b1, labels=[f"{v:.3f}" for v in h1_vals], padding=5, fontsize=10, fontweight="bold", color=PAL["text"])
        ax1.set_xticks(h1_x)
        ax1.set_xticklabels(h1_labs, fontsize=9)
        ax1.tick_params(axis="x", pad=6)
        ax1.set_ylabel("Relative scale (panel max = 1)")
        ax1.set_ylim(0, 1.22)

        h2_x = np.arange(2)
        h2_vals = np.array([0.019, 0.144])
        h2_plot = h2_vals / np.max(h2_vals)
        h2_labs = ["Grounded factual\nrange (max − min\nacross CMI tertiles)", "KW p-value\n(gain differs\nacross tertiles?)"]
        b2 = ax2.bar(h2_x, h2_plot, width=0.55, color=[col_delta, col_second], edgecolor=edge, linewidth=0.75)
        ax2.bar_label(b2, labels=[f"{v:.3f}" for v in h2_vals], padding=5, fontsize=10, fontweight="bold", color=PAL["text"])
        ax2.set_xticks(h2_x)
        ax2.set_xticklabels(h2_labs, fontsize=8.5)
        ax2.tick_params(axis="x", pad=6)
        ax2.set_ylabel("Relative scale (panel max = 1)")
        ax2.set_ylim(0, 1.22)

        h_leg = [
            mpatches.Patch(facecolor=col_delta, edgecolor=edge, label="First bar: Δ (H1) or spread (H2)"),
            mpatches.Patch(facecolor=col_second, edgecolor=edge, label="Second bar: Cohen's d (H1) or KW p (H2)"),
        ]
        fig.legend(
            handles=h_leg,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.090),
            bbox_transform=fig.transFigure,
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor=PAL["edge"],
            fontsize=7.0,
        )

    _save(fig, "Hypothesis_Scaled", pad_inches=0.06)


def plot_system_evidence_heatmap() -> None:
    """Heatmap: pipeline layers × design properties (not hypothesis tests)."""
    with plt.rc_context(_poster_rc()):
        fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor=PAL["bg"])
        # Room for x tick labels + xlabel (no bottom fig.legend — colorbar encodes scale)
        fig.subplots_adjust(left=0.22, right=0.88, top=0.92, bottom=0.26)

        rows = ["Ingest\n(MMCQSD)", "Corpus\n(MultiCaRe)", "Encode\n(LaBSE)", "Index\n(FAISS)", "Gate\n(condition)", "Generate\n(Llama+Groq)", "Score\n(concepts)"]
        cols = ["Hinglish\nqueries", "English\nevidence", "Cross-\nlingual", "Vector\nretrieval", "Clinical\nnarratives", "Paired\nprotocol", "Automated\nmetrics"]
        # 0=absent, 1=partial, 2=core role
        mat = np.array(
            [
                [2, 0, 1, 0, 0, 1, 1],
                [0, 2, 1, 1, 2, 0, 0],
                [1, 1, 2, 1, 1, 1, 0],
                [0, 0, 1, 2, 1, 1, 0],
                [1, 1, 1, 1, 1, 2, 0],
                [2, 2, 2, 2, 2, 2, 1],
                [1, 1, 1, 1, 1, 2, 2],
            ],
            dtype=float,
        )
        # Blue-only: 0 = light, 2 = dark (core reads strong); top shade kept dark enough for black annot
        cmap_blue = ListedColormap(["#D8E6F5", "#3D7AB8", "#153A6E"])
        hm = sns.heatmap(
            mat,
            ax=ax,
            cmap=cmap_blue,
            vmin=0,
            vmax=2,
            linewidths=0.9,
            linecolor=PAL["edge"],
            xticklabels=cols,
            yticklabels=rows,
            annot=True,
            fmt=".0f",
            annot_kws={"size": 8},
            cbar_kws={"shrink": 0.52, "aspect": 14, "ticks": [0, 1, 2]},
        )
        ax.set_xlabel("Design / evidence property", labelpad=10)
        ax.set_ylabel("Pipeline layer", labelpad=12)
        cbar = hm.collections[0].colorbar
        cbar.set_ticklabels(["0 none", "1 partial", "2 core"])
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Role in stack", fontsize=9)

    _save(fig, "Design_Heatmap")


def plot_limitations_mitigation_parallel() -> None:
    """Bars (0–1) + dedicated text column (sharey) so notes are never clipped."""
    with plt.rc_context(_poster_rc()):
        # Limitations-only — saturated enough for poster visibility (do not reuse for other poster charts)
        c_b = "#A78BFA"
        r_b = "#2DD4BF"
        edge = "#475569"

        fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor=PAL["bg"])
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.11)
        ax = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[0, 1], sharey=ax)

        fig.subplots_adjust(left=0.22, right=0.80, top=0.92, bottom=0.22)

        labels = [
            "Corpus too narrow\n(Open-i chest)",
            "Answers without\nevidence (hallucination)",
            "No clinician\nvalidation",
            "Text-only\npipeline",
        ]
        challenge = np.array([0.92, 0.88, 0.78, 0.55])
        response = np.array([0.88, 0.82, 0.52, 0.42])
        notes = [
            "MultiCaRe pivot + LaBSE/FAISS pairing → 3,015 aligned pairs.",
            "Text-grounded RAG + scoring on same retrieved evidence (H1).",
            "Automated factual/hallucination proxies + Streamlit demo.",
            "Multimodal retrieval deferred; current stack is text-first.",
        ]

        y = np.arange(len(labels))
        h = 0.40
        ax.barh(y - h / 2, challenge, height=h, label="Challenge (how pressing)", color=c_b, edgecolor=edge, linewidth=0.65, alpha=1.0)
        ax.barh(y + h / 2, response, height=h, label="Our response (what we built)", color=r_b, edgecolor=edge, linewidth=0.65, alpha=1.0)
        ax.set_ylim(-0.5, len(labels) - 0.5)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.8)
        ax.tick_params(axis="y", pad=6)
        ax.set_xlabel("Relative scale (higher = stronger)")
        ax.set_xlim(0, 1.05)
        ax.tick_params(axis="x", pad=5)
        ax.grid(True, axis="x", alpha=0.4)

        h_ch = mpatches.Patch(facecolor=c_b, edgecolor=edge, label="Challenge (how pressing)")
        h_rs = mpatches.Patch(facecolor=r_b, edgecolor=edge, label="Our response (what we built)")
        fig.legend(
            handles=[h_ch, h_rs],
            loc="center",
            bbox_to_anchor=(0.5, 0.108),
            bbox_transform=fig.transFigure,
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor=PAL["edge"],
            fontsize=7.3,
        )

        ax_txt.axis("off")
        ax_txt.set_xlim(0, 1)
        for yi, note in zip(y, notes):
            wrapped = textwrap.fill(note, width=34, break_long_words=False, break_on_hyphens=False)
            ax_txt.text(
                0.02,
                yi,
                wrapped,
                va="center",
                ha="left",
                fontsize=7.4,
                color=PAL["text"],
                linespacing=1.22,
            )

    _save(fig, "Challenge_Mitigation", pad_inches=0.08)


def main() -> None:
    _style()
    plt.rcParams["axes.unicode_minus"] = False
    print(f"Generating charts ({POSTER_W_CM:.0f} cm × {POSTER_H_CM:.0f} cm) → {HERE}")
    print("Showcase (3 poster visuals) …")
    plot_visual_01_grounding_evidence()
    plot_visual_02_data_corpus()
    plot_visual_03_robustness_inference()
    print("Hypotheses barplot, system heatmap, limitations (grouped barh) …")
    plot_hypotheses_barplot()
    plot_hypotheses_barplot_2()
    plot_system_evidence_heatmap()
    plot_limitations_mitigation_parallel()
    print("Done.")


if __name__ == "__main__":
    main()
