"""Build blinded human-annotation sheets (ICCSDI review: R3.4, R1.b, R1.c, R1.d).

Four reviewer requests are answered by ONE annotation effort:

    R3.4 / R1.c  the 26-concept lexicon is not clinician-validated
    R1.d         richer factuality metrics than concept precision
    R3.2 / R1.b  finer relevance judgments than condition-group matching

Two sheets are produced.

SHEET A -- answer annotation (validates the extractor and scores quality)
    One row per sampled answer, stratified by arm (grounded / zero-shot) and by
    code-mixing tertile, with the arm label REMOVED so the annotator cannot tell
    which system produced the text. The annotator fills three columns:
        wrong_concepts    extractor concepts the answer does not actually assert
                          (including ones it negates) -> extractor precision
        missed_concepts   concepts the answer does assert but the extractor
                          missed                        -> extractor recall
        appropriateness   0 / 1 / 2 clinical usefulness against the gold English
                          question                       -> a human quality metric

SHEET B -- relevance annotation (finer than condition-group matching)
    One row per (query, retrieved case) for the top 3 cases of the deployed
    hybrid system, in both query languages, with system and language blinded and
    rows shuffled. The annotator fills:
        relevance         0 not relevant / 1 same clinical area / 2 answers this
                          patient's question

Both sheets are written twice, `_annotatorA` and `_annotatorB`, so two people can
label independently and Cohen's kappa can be computed. A clinician labelling even
one copy is worth more than two non-clinicians; the sheets are identical either
way, and `annotation_agreement.py` reports who labelled what.

No API calls. Writes results/annotation/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.cmi import build_english_vocab, hindi_proportion
from src.analysis.h4_retrieval import strip_caption
from src.evaluation.concept_lexicon import POSITIVE_CONCEPTS, extract_concepts

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COMBINED = Path("results/combined_h1h2/combined_scored.csv")
PAIRS = Path("data/processed/mmcqsd_multicare_paired.csv")
META = Path("data/faiss_index/evidence_metadata.csv")
RANKINGS = Path("data/passage_index")
OUT = Path("results/annotation")
SEED = 42
TOP_N = 3


def sheet_a(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.read_csv(COMBINED)
    vocab = build_english_vocab()
    df["hindi_prop"] = df.hinglish_query.apply(lambda t: hindi_proportion(t, vocab))
    df["tertile"] = pd.qcut(df.hindi_prop.rank(method="first"), 3,
                            labels=["low", "medium", "high"])

    long = pd.concat([
        df.assign(arm="grounded", answer=df.grounded_output),
        df.assign(arm="zero_shot", answer=df.zero_shot_output),
    ], ignore_index=True)
    long = long[long.answer.notna() & (long.answer.astype(str).str.strip() != "")]

    # Equal cells across the 2 arms x 3 tertiles, then shuffled so the annotator
    # cannot infer the arm from position.
    per_cell = max(1, n // 6)
    cells = [g.sample(min(per_cell, len(g)), random_state=SEED)
             for _, g in long.groupby(["arm", "tertile"], observed=True)]
    picked = (pd.concat(cells, ignore_index=True)
                .sample(frac=1.0, random_state=SEED).reset_index(drop=True))

    rows = []
    for i, r in picked.iterrows():
        concepts = sorted(extract_concepts(str(r.answer)))
        rows.append({
            "item_id": f"A{i + 1:03d}",
            "patient_question": r.hinglish_query,
            "answer_text": str(r.answer),
            "extractor_concepts": ", ".join(concepts) if concepts else "(none)",
            "wrong_concepts": "",
            "missed_concepts": "",
            "appropriateness": "",
            "_hidden_arm": r.arm,
            "_hidden_pair_id": r.pair_id,
            "_hidden_tertile": r.tertile,
        })
    return pd.DataFrame(rows)


def sheet_b(n: int, rng: np.random.Generator) -> pd.DataFrame:
    pairs = pd.read_csv(PAIRS)
    meta = pd.read_csv(META)
    variants = {}
    for name in ("Q1_hinglish", "Q2_english_question"):
        f = RANKINGS / f"rankings_{name}.npz"
        if not f.exists():
            logger.warning("no cached rankings for %s -- run src.analysis.retrieval_v2 first", name)
            return pd.DataFrame()
        variants[name] = np.load(f)["Hybrid-RRF"]

    idx = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    rows = []
    for q in idx:
        query_hi = pairs.hinglish_query.iloc[q]
        query_en = strip_caption(str(pairs.english_summary.iloc[q]))
        for name, ranking in variants.items():
            for rank in range(TOP_N):
                case_row = int(ranking[q, rank])
                if case_row < 0:
                    continue
                rows.append({
                    "patient_question": query_hi,
                    "gold_english_question": query_en,
                    "retrieved_case": " ".join(str(meta.case_text.iloc[case_row]).split()[:180]),
                    "relevance": "",
                    "_hidden_pair_id": pairs.pair_id.iloc[q],
                    "_hidden_query_variant": name,
                    "_hidden_rank": rank + 1,
                    "_hidden_case_row": case_row,
                    "_hidden_same_group": bool(
                        meta.condition_group.iloc[case_row] == pairs.condition_query.iloc[q]),
                })
    out = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out.insert(0, "item_id", [f"B{i + 1:03d}" for i in range(len(out))])
    return out


INSTRUCTIONS = """# Annotation instructions

Two sheets, two annotators. Work **independently** -- do not discuss items while
labelling, or the agreement statistic becomes meaningless. Columns starting with
`_hidden_` are for scoring afterwards: **do not read them while labelling**, and
do not edit them.

If a clinician is available, their copy is the one that validates the lexicon.
A second, non-clinician copy is still useful: it measures how reproducible the
judgments are.

## Sheet A -- answers (`annotation_sheet_A_*.csv`)

For each row, read `patient_question` and `answer_text`, then fill three columns.

1. **`wrong_concepts`** -- of the terms listed in `extractor_concepts`, write the
   ones the answer does **not** actually assert about this patient. Include any
   the answer **denies** ("rash nahi hai") or merely mentions hypothetically.
   Comma-separated, blank if all are correct.
2. **`missed_concepts`** -- clinical findings the answer **does** assert but which
   are missing from `extractor_concepts`. Use the vocabulary in
   `concept_vocabulary.csv`; if the right word is not in that list, write it
   anyway and mark it with a `*`. Blank if nothing was missed.
3. **`appropriateness`** -- how clinically useful the answer is for this patient:
   - `0` wrong, unsafe, or off-topic
   - `1` not wrong but unhelpful (vague, hedged, or refuses without direction)
   - `2` clinically appropriate and responsive

## Sheet B -- retrieved cases (`annotation_sheet_B_*.csv`)

For each row, read the patient's question and the retrieved case, then fill
**`relevance`**:
   - `0` not relevant
   - `1` same clinical area, but would not answer this patient's question
   - `2` directly relevant: a clinician could use this case to answer it

The same question appears several times with different cases, in a shuffled
order. That is intentional. Judge each row on its own.

## When finished

Save each file under its own name, keeping the `_annotatorA` / `_annotatorB`
suffix, then run:

    python -m src.evaluation.annotation_agreement
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build blinded annotation sheets")
    ap.add_argument("--n-answers", type=int, default=120, help="rows in sheet A")
    ap.add_argument("--n-queries", type=int, default=40,
                    help="queries in sheet B (x2 languages x top-3 = ~6 rows each)")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    a = sheet_a(args.n_answers, rng)
    b = sheet_b(args.n_queries, rng)

    for who in ("annotatorA", "annotatorB"):
        a.to_csv(OUT / f"annotation_sheet_A_{who}.csv", index=False, encoding="utf-8-sig")
        if not b.empty:
            b.to_csv(OUT / f"annotation_sheet_B_{who}.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"concept": sorted(POSITIVE_CONCEPTS)}).to_csv(
        OUT / "concept_vocabulary.csv", index=False, encoding="utf-8-sig")
    (OUT / "README.md").write_text(INSTRUCTIONS, encoding="utf-8")

    logger.info("sheet A: %d rows (%s)", len(a), dict(a._hidden_arm.value_counts()))
    logger.info("sheet B: %d rows", len(b))
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
