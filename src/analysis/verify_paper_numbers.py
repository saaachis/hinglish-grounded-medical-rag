"""Check every headline number in the manuscript against the committed results.

The repository's own rule is: "if a number is in the paper and this script does
not produce it, it does not go in the paper." Preparing the ICCSDI revision
showed the rule was not being enforced -- three values in the submitted paper
cannot be traced to any committed artifact, and one of them reports a
NON-significant comparison as significant.

This script closes that gap. Each CLAIM below records what the submitted paper
says, where the number should come from, and how to recompute it. Running it
prints VERIFIED / MISMATCH / NOT FOUND per claim, so the camera-ready can be
checked in one command instead of by eye.

    python -m src.analysis.verify_paper_numbers
    python -m src.analysis.verify_paper_numbers --only table3

Tolerance is 0.0005 absolute by default (the paper quotes 4 decimals), and
relative for p-values, which differ in the last digit across scipy versions.

Writes results/paper_verification.md.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("results/paper_verification.md")
TOL = 5e-4


def _csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@dataclass
class Claim:
    """One number as the paper states it, plus how to recompute it."""
    section: str
    text: str                      # how the paper words it
    claimed: float
    getter: Callable[[], float | None]
    source: str
    tol: float = TOL
    note: str = ""
    tags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #
def v2(system: str, variant: str, col: str):
    def f():
        d = _csv("results/retrieval_v2/retrieval_v2_metrics.csv")
        if d is None:
            return None
        r = d[(d.system == system) & (d.variant == variant)]
        if r.empty:
            return None
        if col in r.columns:
            return float(r[col].iloc[0])
        alt = col.replace("success@", "recall@")
        return float(r[alt].iloc[0]) if alt in r.columns else None
    return f


def v2_penalty(system: str, metric: str, col: str = "penalty"):
    def f():
        d = _csv("results/retrieval_v2/h4_v2_tests_by_depth.csv")
        if d is None:
            return None
        r = d[(d.system == system) & (d.metric == metric)]
        return float(r[col].iloc[0]) if not r.empty else None
    return f


def contrast(source: str, arm: str, reference: str, metric: str, col: str):
    def f():
        d = _csv("results/rescored/contrasts.csv")
        if d is None:
            return None
        r = d[(d.source == source) & (d.arm == arm) & (d.reference == reference)
              & (d.metric == metric)]
        return float(r[col].iloc[0]) if not r.empty else None
    return f


def baseline(system: str):
    def f():
        d = _csv("results/rescored/degenerate_baselines.csv")
        if d is None:
            return None
        r = d[d.system == system]
        return float(r.factual_support.iloc[0]) if not r.empty else None
    return f


def cmi_rho(arm: str):
    def f():
        d = _csv("results/h2_per_arm/h2_corrected_cmi_stats.csv")
        if d is None:
            return None
        r = d[(d.cmi_variant.str.startswith("hindi_prop_v2")) & (d.arm == arm)]
        return float(r.rho.iloc[0]) if not r.empty else None
    return f


def h3(corpus: str, col: str):
    def f():
        d = _csv("results/h3_provenance/h3_summary.csv")
        if d is None:
            return None
        r = d[d.corpus == corpus]
        return float(r[col].iloc[0]) if not r.empty else None
    return f


def muril(variant: str):
    def f():
        d = _csv("results/h4_retrieval/h4_baselines.csv")
        if d is None:
            return None
        r = d[(d.system == "MuRIL") & (d.variant == variant)]
        return float(r["recall@1"].iloc[0]) if not r.empty else None
    return f


CLAIMS: list[Claim] = [
    # ---------------- Table 1 / Sect. 5.1 ----------------
    Claim("Table 1", "Hybrid (RRF) Hinglish 0.1751", 0.1751,
          v2("Hybrid-RRF", "Q1_hinglish", "success@1"), "retrieval_v2_metrics.csv", tags=["table1"]),
    Claim("Table 1", "Hybrid (RRF) English 0.1973", 0.1973,
          v2("Hybrid-RRF", "Q2_english_question", "success@1"), "retrieval_v2_metrics.csv", tags=["table1"]),
    Claim("Table 1", "LaBSE Hinglish 0.1280", 0.1280,
          v2("LaBSE-passages", "Q1_hinglish", "success@1"), "retrieval_v2_metrics.csv", tags=["table1"]),
    Claim("Table 1", "LaBSE English 0.1486", 0.1486,
          v2("LaBSE-passages", "Q2_english_question", "success@1"), "retrieval_v2_metrics.csv", tags=["table1"]),
    Claim("Table 1", "BM25 Hinglish 0.0935", 0.0935,
          v2("BM25-full", "Q1_hinglish", "success@1"), "retrieval_v2_metrics.csv", tags=["table1"]),
    Claim("Table 1", "BM25 English 0.1847", 0.1847,
          v2("BM25-full", "Q2_english_question", "success@1"), "retrieval_v2_metrics.csv", tags=["table1"]),
    Claim("Table 1", "TF-IDF Hinglish 0.0842", 0.0842,
          v2("TFIDF-full", "Q1_hinglish", "success@1"), "retrieval_v2_metrics.csv", tags=["table1"]),
    Claim("Table 1", "TF-IDF English 0.1529", 0.1529,
          v2("TFIDF-full", "Q2_english_question", "success@1"), "retrieval_v2_metrics.csv", tags=["table1"]),
    Claim("Table 1", "Hybrid penalty +0.0222", 0.0222,
          v2_penalty("Hybrid-RRF", "success@1"), "h4_v2_tests_by_depth.csv", tags=["table1"]),
    Claim("Table 1", "LaBSE penalty +0.0206", 0.0206,
          v2_penalty("LaBSE-passages", "success@1"), "h4_v2_tests_by_depth.csv", tags=["table1"]),
    Claim("Table 1", "BM25 penalty +0.0912", 0.0912,
          v2_penalty("BM25-full", "success@1"), "h4_v2_tests_by_depth.csv", tags=["table1"]),
    Claim("Sect. 5.1", "BM25/dense penalty ratio ~4.4x", 4.4,
          lambda: (v2_penalty("BM25-full", "success@1")() or np.nan)
          / (v2_penalty("LaBSE-passages", "success@1")() or np.nan),
          "derived from h4_v2_tests_by_depth.csv", tol=0.05, tags=["table1"]),

    # ---------------- MuRIL / script mismatch ----------------
    Claim("Sect. 5.1", "MuRIL Hinglish 0.0640 (chance)", 0.0640,
          muril("Q1_hinglish"), "h4_baselines.csv", tags=["muril"]),
    Claim("Sect. 5.1", "MuRIL English 0.1821", 0.1821,
          muril("Q2_english_question"), "h4_baselines.csv", tags=["muril"]),

    # ---------------- Table 2 / Sect. 5.2 ----------------
    Claim("Table 2", "constant 'swelling' 0.7132", 0.7132,
          baseline("const:swelling"), "degenerate_baselines.csv", tags=["table2"]),
    Claim("Table 2", "verbatim copy 1.0000", 1.0,
          baseline("copy:reference"), "degenerate_baselines.csv", tags=["table2"]),

    # ---------------- Table 3 / Sect. 5.3 ----------------
    Claim("Table 3", "llama oracle, repaired lexicon, delta +0.2962", 0.2962,
          contrast("llama_oracle_n1165", "grounded", "evidence (circular)", "precision", "delta"),
          "contrasts.csv", tags=["table3"]),
    Claim("Table 3", "llama oracle, repaired lexicon, d = 0.720", 0.720,
          contrast("llama_oracle_n1165", "grounded", "evidence (circular)", "precision", "cohens_d"),
          "contrasts.csv", tol=0.002, tags=["table3"]),
    Claim("Table 3", "gpt-oss-120b oracle delta +0.2659", 0.2659,
          contrast("gptoss120b", "oracle", "evidence (circular)", "precision", "delta"),
          "contrasts.csv", tags=["table3"],
          note="Paper also states n = 223, d = 0.662. contrasts.csv has n = 214, d = 0.640; "
               "h1_real_120b reports +0.2849 (n = 124). No committed file matches the paper."),
    Claim("Table 3", "gpt-oss-120b real retrieval delta +0.2238", 0.2238,
          contrast("gptoss120b", "real", "evidence (circular)", "precision", "delta"),
          "contrasts.csv", tags=["table3"],
          note="Paper states n = 237, d = 0.616. contrasts.csv has +0.0704 (n = 221); "
               "h1_real_120b reports +0.2347 (n = 131). No committed file matches the paper."),

    # ---------------- Sect. 5.3 reference sensitivity ----------------
    Claim("Sect. 5.3", "llama oracle F1, circular +0.203", 0.203,
          contrast("llama_oracle_n1165", "grounded", "evidence (circular)", "f1", "delta"),
          "contrasts.csv", tol=0.001, tags=["reference"]),
    Claim("Sect. 5.3", "llama oracle F1, unbiased +0.062", 0.062,
          contrast("llama_oracle_n1165", "grounded", "caption (unbiased)", "f1", "delta"),
          "contrasts.csv", tol=0.001, tags=["reference"]),
    Claim("Sect. 5.3", "gpt-oss-120b oracle F1, circular +0.093", 0.093,
          contrast("gptoss120b", "oracle", "evidence (circular)", "f1", "delta"),
          "contrasts.csv", tol=0.001, tags=["reference"]),
    Claim("Sect. 5.3", "gpt-oss-120b oracle F1, unbiased -0.021", -0.021,
          contrast("gptoss120b", "oracle", "caption (unbiased)", "f1", "delta"),
          "contrasts.csv", tol=0.001, tags=["reference"]),
    Claim("Sect. 5.3", "gpt-oss-120b real F1, circular -0.032", -0.032,
          contrast("gptoss120b", "real", "evidence (circular)", "f1", "delta"),
          "contrasts.csv", tol=0.001, tags=["reference"]),
    Claim("Sect. 5.3", "gpt-oss-120b real F1, unbiased -0.047", -0.047,
          contrast("gptoss120b", "real", "caption (unbiased)", "f1", "delta"),
          "contrasts.csv", tol=0.001, tags=["reference"]),
    Claim("Sect. 5.3", "grounding reduces concept recall by 0.138", -0.138,
          contrast("gptoss120b", "oracle", "caption (unbiased)", "recall", "delta"),
          "contrasts.csv", tol=0.001, tags=["reference"],
          note="GENERATOR NOT STATED IN THE PAPER, and it matters: this is gpt-oss-120b. "
               "The llama arm moves the OPPOSITE way (recall +0.0774, BH p = 9.3e-07), so "
               "'grounding trades recall for precision' is generator-specific, not general. "
               "Name the generator in the sentence."),
    Claim("Sect. 5.3", "oracle beats real retrieval by +0.0875 (p = 0.041)", 0.0875,
          lambda: None, "NOT FOUND in any committed file", tags=["reference"],
          note="CONTRADICTED. h1_real_120b: +0.0748, n = 76, p = 0.226. "
               "h1_real_retrieval: +0.0755, n = 125, p = 0.106. Both NON-significant. "
               "The paper reports this as 'small but significant'. Must be corrected."),

    # ---------------- Table 4 / Sect. 5.4 ----------------
    Claim("Table 4", "grounded factual support rho = -0.001", -0.001,
          cmi_rho("Grounded factual support"), "h2_corrected_cmi_stats.csv", tol=0.001, tags=["table4"]),
    Claim("Table 4", "zero-shot factual support rho = -0.116", -0.116,
          cmi_rho("Zero-shot factual support"), "h2_corrected_cmi_stats.csv", tol=0.001, tags=["table4"]),
    Claim("Table 4", "grounded hallucination rho = -0.022", -0.022,
          cmi_rho("Grounded hallucination"), "h2_corrected_cmi_stats.csv", tol=0.001, tags=["table4"]),
    Claim("Table 4", "zero-shot hallucination rho = +0.042", 0.042,
          cmi_rho("Zero-shot hallucination"), "h2_corrected_cmi_stats.csv", tol=0.001, tags=["table4"]),

    # ---------------- Table 5 / Sect. 5.5 ----------------
    Claim("Table 5", "MultiCaRe refusal 88.1%", 0.881, h3("multicare", "refusal"),
          "h3_summary.csv", tags=["table5"]),
    Claim("Table 5", "MMedBench refusal 76.2%", 0.762, h3("mmedbench", "refusal"),
          "h3_summary.csv", tags=["table5"]),
    Claim("Table 5", "MultiCaRe mean F1 0.3265", 0.3265, h3("multicare", "f1"),
          "h3_summary.csv", tags=["table5"],
          note="See results/h3_provenance/h3_power.md: 59 of the 63 scoreable MultiCaRe "
               "answers are ALSO flagged refusals, so this F1 is computed largely on hedges."),
    Claim("Table 5", "PubMedQA mean F1 0.3476", 0.3476, h3("pubmedqa", "f1"),
          "h3_summary.csv", tags=["table5"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify paper numbers against committed results")
    ap.add_argument("--only", default="", help="substring match on section or tag")
    args = ap.parse_args()

    claims = [c for c in CLAIMS
              if not args.only or args.only.lower() in (c.section + " " + " ".join(c.tags)).lower()]

    rows = []
    for c in claims:
        try:
            actual = c.getter()
        except Exception as e:                      # a broken lookup must not hide the rest
            actual = None
            c.note = (c.note + f" [lookup failed: {type(e).__name__}]").strip()
        if actual is None or (isinstance(actual, float) and np.isnan(actual)):
            status = "NOT FOUND"
        elif abs(actual - c.claimed) <= c.tol:
            status = "VERIFIED"
        else:
            status = "MISMATCH"
        rows.append({"section": c.section, "claim": c.text, "claimed": c.claimed,
                     "actual": actual, "status": status, "source": c.source, "note": c.note})

    res = pd.DataFrame(rows)
    counts = res.status.value_counts().to_dict()

    L = ["# Paper numbers vs committed results", "",
         f"{len(res)} claims checked. "
         f"**{counts.get('VERIFIED', 0)} verified, {counts.get('MISMATCH', 0)} mismatched, "
         f"{counts.get('NOT FOUND', 0)} not found.**", "",
         "A number that is not VERIFIED must not appear in the camera-ready until it is either",
         "recomputed or corrected. Tolerance is 5e-4 unless stated.", "",
         "| Status | Section | Claim in the paper | Claimed | Recomputed | Source |",
         "|---|---|---|---:|---:|---|"]
    for _, r in res.sort_values("status").iterrows():
        mark = {"VERIFIED": "ok", "MISMATCH": "**MISMATCH**", "NOT FOUND": "**NOT FOUND**"}[r.status]
        act = "—" if r.actual is None or (isinstance(r.actual, float) and np.isnan(r.actual)) \
            else f"{r.actual:.4f}"
        L.append(f"| {mark} | {r.section} | {r.claim} | {r['claimed']:.4f} | {act} | `{r.source}` |")

    problems = res[res.status != "VERIFIED"]
    if len(problems):
        L += ["", "## Every claim that did not verify", ""]
        for _, r in problems.iterrows():
            L += [f"### {r.section} — {r.claim}",
                  f"- paper says **{r['claimed']:.4f}**",
                  f"- recomputed: **{'not found' if r.actual is None or (isinstance(r.actual, float) and np.isnan(r.actual)) else format(r.actual, '.4f')}** "
                  f"(from `{r.source}`)"]
            if r.note:
                L.append(f"- {r.note}")
            L.append("")
    OUT.write_text("\n".join(L), encoding="utf-8")
    res.to_csv(OUT.with_suffix(".csv"), index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
