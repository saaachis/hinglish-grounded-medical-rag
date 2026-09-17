"""One table for the grounding effect under both metrics and both references (R3.5).

Reviewer 3: "Grounding benefits diminish or become negative using the unbiased
reference; the abstract should clearly distinguish reference-dependent improvement
from general factual-quality improvement."

The submitted paper makes that hard to see. Table 3 reports concept PRECISION
against the retrieved evidence, Fig. 2 reports concept F1 against two references,
and the switch of both metric and reference happens in prose between them. The
same condition therefore appears as +0.224 in the table and −0.032 in the figure.

This builds the 2x3 view instead: each generator/evidence condition as a row, with

    precision vs the evidence it was given   (the circular reference)
    F1 vs the evidence it was given
    F1 vs the image description              (the unbiased reference; neither arm saw it)

Reading across a row shows how much of the reported benefit is an artefact of
scoring against the text the grounded arm was conditioned on. That is the
paper's methodological contribution, and it should be visible in one glance.

No API calls; reads results/rescored/contrasts.csv.
Writes results/rescored/reference_matrix.{md,csv}.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CONTRASTS = Path("results/rescored/contrasts.csv")
BASELINES = Path("results/rescored/degenerate_baselines.csv")
OUT = Path("results/rescored/reference_matrix")

#: (source, arm) -> how the paper should name the row.
ROWS = [
    ("llama_oracle_n1165", "grounded", "llama-3.1-8b-instant, oracle evidence"),
    ("gptoss120b", "oracle", "gpt-oss-120b, oracle evidence"),
    ("gptoss120b", "real", "gpt-oss-120b, real retrieval"),
    ("gptoss20b_n268", "oracle", "gpt-oss-20b, oracle evidence"),
    ("gptoss20b_n268", "real", "gpt-oss-20b, real retrieval"),
]

CELLS = [
    ("precision", "evidence (circular)", "precision vs evidence"),
    ("f1", "evidence (circular)", "F1 vs evidence"),
    ("f1", "caption (unbiased)", "F1 vs unbiased"),
]


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def main() -> None:
    c = pd.read_csv(CONTRASTS)
    rows = []
    for source, arm, label in ROWS:
        row = {"condition": label, "source": source, "arm": arm}
        for metric, reference, name in CELLS:
            m = c[(c.source == source) & (c.arm == arm)
                  & (c.metric == metric) & (c.reference == reference)]
            if m.empty:
                row[name] = None
                continue
            r = m.iloc[0]
            row[name] = float(r.delta)
            row[f"{name} n"] = int(r.n_paired)
            row[f"{name} d"] = float(r.cohens_d)
            row[f"{name} p_bh"] = float(r.p_bh)
        rows.append(row)
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT.with_suffix(".csv"), index=False)

    L = ["# The grounding effect under both metrics and both references", "",
         "Delta = grounded - zero-shot, within query, same rows scored both ways.",
         "Significance is BH-corrected across the whole re-scoring family.", "",
         "| Condition | precision vs evidence | F1 vs evidence | F1 vs unbiased |",
         "|---|---|---|---|"]
    for _, r in res.iterrows():
        cells = []
        for _, _, name in CELLS:
            if r.get(name) is None or pd.isna(r.get(name)):
                cells.append("—")
            else:
                cells.append(f"{r[name]:+.4f} {stars(r[f'{name} p_bh'])}<br><sub>n={int(r[f'{name} n'])}, "
                             f"d={r[f'{name} d']:+.2f}</sub>")
        L.append(f"| {r.condition} | {cells[0]} | {cells[1]} | {cells[2]} |")

    L += ["", "`***` BH p < 0.001 · `**` < 0.01 · `*` < 0.05 · `n.s.` not significant", ""]

    if BASELINES.exists():
        b = pd.read_csv(BASELINES)
        sw = b[b.system == "const:swelling"]
        if not sw.empty:
            L += ["> **Absolute levels are not quality.** Against the unbiased reference a constant "
                  f"one-word answer (\"swelling\") scores {float(sw.factual_support.iloc[0]):.4f} on "
                  "precision, above every real system. Only the paired deltas in this table are "
                  "interpretable, and only within a column.", ""]

    # What the rows collectively say, computed rather than asserted.
    circ = res["precision vs evidence"].dropna()
    unb = res["F1 vs unbiased"].dropna()
    L += ["## Reading", "",
          f"- Against the evidence the grounded arm was given, the effect is positive in "
          f"{int((circ > 0).sum())} of {len(circ)} conditions on precision.",
          f"- Against a reference neither arm saw, it is positive in "
          f"{int((unb > 0).sum())} of {len(unb)} conditions on F1.",
          "- The benefit is therefore **reference-dependent**: real under the protocol in common",
          "  use, and much smaller or absent under one that removes the circularity. The abstract",
          "  must state both, in that order.", ""]
    OUT.with_suffix(".md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
