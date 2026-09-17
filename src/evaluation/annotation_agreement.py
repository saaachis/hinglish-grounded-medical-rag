"""Score the completed annotation sheets (ICCSDI review: R3.4, R1.b, R1.c, R1.d).

Reads whatever `results/annotation/annotation_sheet_[AB]_*.csv` files have been
filled in and reports:

SHEET A -- extractor validation and human answer quality
    * concept-extractor precision, recall and F1 against the human labels, which
      is the "clinician-validated lexicon" evidence the reviewers asked for
    * Cohen's kappa between annotators on the per-concept assertion decision
    * mean appropriateness (0/1/2) by arm, with a Mann-Whitney test, which is a
      human quality measure that does not depend on the scoring reference at all

SHEET B -- relevance validation
    * how often the AUTOMATIC condition-group criterion agrees with a human
      relevance judgment, i.e. how good a proxy the paper's Recall@1 label is
    * judged precision@3 and graded nDCG@3 per query language
    * Cohen's kappa between annotators (quadratic weights, since 0/1/2 is ordinal)

Partial sheets are fine: rows with a blank judgment are skipped and the counts
say how many were used. No API calls.

Writes results/annotation/annotation_report.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.retrieval_metrics import discounts

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DIR = Path("results/annotation")
OUT = DIR / "annotation_report.md"
TOP_N = 3


def parse_list(cell: object) -> set[str]:
    if pd.isna(cell):
        return set()
    return {c.strip().lower().rstrip("*").strip() for c in str(cell).split(",") if c.strip()
            and c.strip().lower() not in {"(none)", "none", "-"}}


def cohens_kappa(a: list, b: list, weights: str | None = None) -> float:
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return float("nan")
    i = {c: k for k, c in enumerate(cats)}
    n = len(cats)
    O = np.zeros((n, n))
    for x, y in zip(a, b):
        O[i[x], i[y]] += 1
    O /= O.sum()
    E = O.sum(axis=1)[:, None] * O.sum(axis=0)[None, :]
    if weights == "quadratic":
        idx = np.arange(n)
        W = ((idx[:, None] - idx[None, :]) / (n - 1)) ** 2
        return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() else float("nan")
    po, pe = np.trace(O), np.trace(E)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def load(prefix: str) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(DIR.glob(f"annotation_sheet_{prefix}_*.csv")):
        who = f.stem.split("_")[-1]
        df = pd.read_csv(f)
        judged = (df.appropriateness.notna() if prefix == "A" else df.relevance.notna())
        if judged.sum() == 0:
            logger.info("%s: not filled in yet, skipping", f.name)
            continue
        out[who] = df[judged].copy()
        logger.info("%s: %d judged rows", f.name, len(out[who]))
    return out


def score_sheet_a(sheets: dict[str, pd.DataFrame], L: list[str]) -> None:
    L += ["## Sheet A -- concept extractor vs human labels", ""]
    if not sheets:
        L += ["_No completed sheet A found. Fill in `annotation_sheet_A_annotatorA.csv` "
              "(and optionally `_annotatorB`), then re-run._", ""]
        return

    L += ["| Annotator | Items | Extractor precision | Extractor recall | Extractor F1 |",
          "|---|---:|---:|---:|---:|"]
    for who, df in sheets.items():
        tp = fp = fn = 0
        for _, r in df.iterrows():
            extracted = parse_list(r.extractor_concepts)
            wrong = parse_list(r.wrong_concepts) & extracted
            missed = parse_list(r.missed_concepts)
            tp += len(extracted - wrong)
            fp += len(wrong)
            fn += len(missed)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
        L.append(f"| {who} | {len(df)} | {prec:.3f} | {rec:.3f} | {f1:.3f} |")

    L += ["", "The extractor's own scores are only as good as these numbers. Precision below",
          "~0.8 means the reported factual-support values carry a measurable extraction error,",
          "and the paper must say so.", ""]

    # per-concept assertion agreement
    if len(sheets) >= 2:
        (wa, da), (wb, db) = list(sheets.items())[:2]
        merged = da.merge(db, on="item_id", suffixes=("_a", "_b"))
        la, lb = [], []
        for _, r in merged.iterrows():
            extracted = parse_list(r.extractor_concepts_a)
            universe = extracted | parse_list(r.missed_concepts_a) | parse_list(r.missed_concepts_b)
            for c in universe:
                la.append(int(c in (extracted - parse_list(r.wrong_concepts_a))
                               or c in parse_list(r.missed_concepts_a)))
                lb.append(int(c in (extracted - parse_list(r.wrong_concepts_b))
                               or c in parse_list(r.missed_concepts_b)))
        if la:
            L += [f"**Inter-annotator agreement on concept assertion** ({wa} vs {wb}): "
                  f"Cohen's kappa = {cohens_kappa(la, lb):.3f} over {len(la)} concept decisions.", ""]

    L += ["### Human answer quality by arm (appropriateness 0/1/2)", "",
          "| Annotator | Grounded mean | Zero-shot mean | Mann-Whitney p |", "|---|---:|---:|---:|"]
    for who, df in sheets.items():
        d = df.copy()
        d["score"] = pd.to_numeric(d.appropriateness, errors="coerce")
        g = d.loc[d._hidden_arm == "grounded", "score"].dropna()
        z = d.loc[d._hidden_arm == "zero_shot", "score"].dropna()
        p = float(stats.mannwhitneyu(g, z).pvalue) if len(g) and len(z) else float("nan")
        L.append(f"| {who} | {g.mean():.3f} (n={len(g)}) | {z.mean():.3f} (n={len(z)}) | {p:.3g} |")
    if len(sheets) >= 2:
        (wa, da), (wb, db) = list(sheets.items())[:2]
        m = da.merge(db, on="item_id", suffixes=("_a", "_b"))
        a = pd.to_numeric(m.appropriateness_a, errors="coerce")
        b = pd.to_numeric(m.appropriateness_b, errors="coerce")
        ok = a.notna() & b.notna()
        if ok.sum():
            k = cohens_kappa(a[ok].astype(int).tolist(), b[ok].astype(int).tolist(), "quadratic")
            L += ["", f"Appropriateness agreement ({wa} vs {wb}): quadratic-weighted kappa = "
                  f"{k:.3f} over {int(ok.sum())} items."]
    L += ["", "This is a reference-free quality measure: unlike concept precision it cannot be",
          "gamed by a one-word answer, and it does not depend on which reference is scored against.", ""]


def score_sheet_b(sheets: dict[str, pd.DataFrame], L: list[str]) -> None:
    L += ["## Sheet B -- human relevance vs the automatic condition-group label", ""]
    if not sheets:
        L += ["_No completed sheet B found. Run `src.analysis.retrieval_v2` to cache rankings, "
              "rebuild the sheets, then fill them in._", ""]
        return

    for who, df in sheets.items():
        d = df.copy()
        d["rel"] = pd.to_numeric(d.relevance, errors="coerce")
        d = d[d.rel.notna()]
        same = d._hidden_same_group.astype(bool)
        L += [f"### {who} ({len(d)} judged rows)", "",
              "| Automatic label | n | Human relevant (>=1) | Human directly relevant (2) |",
              "|---|---:|---:|---:|",
              f"| same condition group | {int(same.sum())} | {(d.rel[same] >= 1).mean():.1%} | "
              f"{(d.rel[same] == 2).mean():.1%} |",
              f"| different group | {int((~same).sum())} | {(d.rel[~same] >= 1).mean():.1%} | "
              f"{(d.rel[~same] == 2).mean():.1%} |", ""]
        agree = ((d.rel >= 1) == same).mean()
        k = cohens_kappa((d.rel >= 1).astype(int).tolist(), same.astype(int).tolist())
        L += [f"Automatic label agrees with the human judgment on **{agree:.1%}** of rows "
              f"(kappa = {k:.3f}).", ""]

        rows = []
        for variant, g in d.groupby("_hidden_query_variant"):
            byq = g.sort_values("_hidden_rank").groupby("_hidden_pair_id")
            p3 = byq.rel.apply(lambda r: (r.values[:TOP_N] >= 1).mean()).mean()
            disc = discounts(TOP_N)

            def ndcg(r):
                g_ = (2.0 ** r.values[:TOP_N] - 1)
                dcg = (g_ * disc[:len(g_)]).sum()
                ideal = (np.sort(g_)[::-1] * disc[:len(g_)]).sum()
                return dcg / ideal if ideal else 0.0

            rows.append({"query": variant, "queries": byq.ngroups, "judged_P@3": p3,
                         "judged_nDCG@3_within_list": byq.rel.apply(ndcg).mean()})
        t = pd.DataFrame(rows)
        L += ["| Query language | Queries | Judged P@3 | Judged nDCG@3 (within retrieved list) |",
              "|---|---:|---:|---:|"]
        for _, r in t.iterrows():
            L.append(f"| {r['query']} | {r.queries} | {r['judged_P@3']:.3f} | "
                     f"{r['judged_nDCG@3_within_list']:.3f} |")
        L += ["", "> nDCG here is computed within the retrieved list only (the ideal is the best",
              "> ordering of the judged cases), because the full corpus was not judged. It measures",
              "> ranking quality among retrieved results, not absolute retrieval quality.", ""]

    if len(sheets) >= 2:
        (wa, da), (wb, db) = list(sheets.items())[:2]
        m = da.merge(db, on="item_id", suffixes=("_a", "_b"))
        a = pd.to_numeric(m.relevance_a, errors="coerce")
        b = pd.to_numeric(m.relevance_b, errors="coerce")
        ok = a.notna() & b.notna()
        if ok.sum():
            k = cohens_kappa(a[ok].astype(int).tolist(), b[ok].astype(int).tolist(), "quadratic")
            L += [f"**Inter-annotator agreement on relevance** ({wa} vs {wb}): "
                  f"quadratic-weighted kappa = {k:.3f} over {int(ok.sum())} rows.", ""]


def main() -> None:
    a, b = load("A"), load("B")
    L = ["# Human annotation results", "",
         "Scores the completed blinded sheets. See `README.md` in this directory for the",
         "labelling instructions and `annotation_sample.py` for how the sample was drawn.", ""]
    score_sheet_a(a, L)
    score_sheet_b(b, L)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
