"""H03: build TOPICALLY MATCHED, EQUAL-SIZED evidence corpora.

THE PROBLEM THIS SOLVES. H03 asks whether authoritative case reports are better
grounding evidence than general biomedical text. Answering it naively -- index each
corpus whole and compare -- does not test that, because the corpora differ enormously
in how much they say about the 18 conditions our patients ask about:

    MultiCaRe case reports      67.9% of documents on-topic
    MMedBench (English subset)  16.4%
    PubMedQA abstracts           2.1%

A naive comparison would find MultiCaRe wins because PubMedQA barely covers these
conditions at all -- measuring corpus TOPICALITY, not evidence PROVENANCE. That is an
uninformative result dressed as a finding.

THE FIX. Filter every corpus to its on-topic documents, then sample each to the SAME
size. The binding constraint is MMedBench English (1,872 usable documents), so all
corpora are capped there. Any remaining difference is then attributable to the kind of
text, which is what H03 actually asks.

FOUR CONDITIONS
    multicare   clinical case narratives   authoritative, case-level
    pubmedqa    research abstracts         general biomedical
    mmedbench   exam / didactic text       instructional
    shuffled    MultiCaRe with sentences shuffled within each document -- a floor that
                holds topic and vocabulary constant while destroying discourse
                structure, so it isolates whether coherent narrative matters

MMedBench is restricted to its English rows (57.6% of the corpus is Chinese);
otherwise provenance would be confounded with language.

Writes data/h3_corpora/{name}.csv  (columns: doc_id, text, condition_group)
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUT = Path("data/h3_corpora")
SEED = 42

#: Surface forms for each MMCQSD condition group. Deliberately generous -- a document
#: only has to be ABOUT the condition to be eligible, and over-inclusion costs less
#: here than wrongly excluding a corpus's on-topic material.
TERMS: dict[str, list[str]] = {
    "skin_rash": ["rash", "exanthem", "eruption"],
    "neck_swelling": ["neck swelling", "neck mass", "cervical lymphadenopathy", "cervical swelling"],
    "mouth_ulcers": ["mouth ulcer", "oral ulcer", "aphthous", "stomatitis"],
    "lip_swelling": ["lip swelling", "cheilitis", "angioedema"],
    "swollen_tonsils": ["tonsil", "tonsillitis", "pharyngitis"],
    "foot_swelling": ["foot swelling", "pedal edema", "ankle swelling"],
    "hand_lump": ["hand mass", "palmar nodule", "hand lump", "ganglion cyst"],
    "swollen_eye": ["eyelid swelling", "periorbital", "swollen eye"],
    "knee_swelling": ["knee swelling", "knee effusion", "knee joint effusion"],
    "edema": ["edema", "oedema"],
    "eye_redness": ["red eye", "conjunctival injection", "eye redness"],
    "skin_growth": ["skin growth", "cutaneous nodule", "skin tumor", "skin lesion"],
    "skin_irritation": ["dermatitis", "skin irritation"],
    "skin_dryness": ["xerosis", "dry skin"],
    "dry_scalp": ["scalp", "seborrheic"],
    "eye_inflammation": ["uveitis", "conjunctivitis", "keratitis"],
    "cyanosis": ["cyanosis", "cyanotic"],
    "itchy_eyelid": ["blepharitis", "itchy eyelid"],
}
PATTERNS = {c: re.compile("|".join(re.escape(t) for t in ts), re.I) for c, ts in TERMS.items()}


def label(text: str) -> str | None:
    """Assign the first matching condition group, or None if off-topic."""
    t = str(text)
    for c, p in PATTERNS.items():
        if p.search(t):
            return c
    return None


def filter_and_label(texts: pd.Series, ids: pd.Series, source: str) -> pd.DataFrame:
    rows = []
    for i, (doc_id, t) in enumerate(zip(ids, texts)):
        c = label(t)
        if c:
            rows.append({"doc_id": f"{source}:{doc_id}", "text": str(t), "condition_group": c})
        if (i + 1) % 50000 == 0:
            logger.info("  %s: scanned %d, kept %d", source, i + 1, len(rows))
    return pd.DataFrame(rows)


def balanced_sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Sample n documents, spread across condition groups as evenly as supply allows."""
    if len(df) <= n:
        return df.reset_index(drop=True)
    groups = df.condition_group.unique()
    per = max(1, n // len(groups))
    rng = np.random.RandomState(seed)
    parts = [g.sample(n=min(len(g), per), random_state=rng)
             for _, g in df.groupby("condition_group")]
    out = pd.concat(parts)
    if len(out) < n:                       # top up from whatever remains
        rest = df[~df.index.isin(out.index)]
        out = pd.concat([out, rest.sample(n=min(len(rest), n - len(out)), random_state=rng)])
    return out.sample(frac=1, random_state=rng).head(n).reset_index(drop=True)


def shuffle_sentences(text: str, rng: np.random.RandomState) -> str:
    """Destroy discourse order while preserving topic and vocabulary exactly."""
    sents = re.split(r"(?<=[.!?])\s+", str(text))
    if len(sents) > 1:
        rng.shuffle(sents)
    return " ".join(sents)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build matched-topic H03 corpora")
    ap.add_argument("--size", type=int, default=0,
                    help="Docs per corpus. 0 = use the smallest corpus as the cap.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    built: dict[str, pd.DataFrame] = {}

    # ---- MultiCaRe: clinical case narratives ----
    logger.info("MultiCaRe ...")
    mc = pd.read_csv("data/processed/multicare_filtered.csv",
                     usecols=["case_id", "case_text"], nrows=80000)
    built["multicare"] = filter_and_label(mc.case_text, mc.case_id, "multicare")
    logger.info("  on-topic: %d of %d (%.1f%%)", len(built["multicare"]), len(mc),
                100 * len(built["multicare"]) / len(mc))

    # ---- PubMedQA: research abstracts ----
    logger.info("PubMedQA ...")
    acc = []
    for chunk in pd.read_csv("data/processed/pubmedqa_records.csv",
                             usecols=["sample_id", "context_text"], chunksize=50000):
        acc.append(filter_and_label(chunk.context_text, chunk.sample_id, "pubmedqa"))
    built["pubmedqa"] = pd.concat(acc, ignore_index=True)
    logger.info("  on-topic: %d", len(built["pubmedqa"]))

    # ---- MMedBench: exam / didactic text, ENGLISH ONLY ----
    logger.info("MMedBench (English only) ...")
    mb = pd.read_csv("data/processed/mmedbench_questions.csv",
                     usecols=["sample_id", "language", "question", "rationale"])
    mb = mb[mb.language == "English"]
    txt = mb.question.fillna("").astype(str) + " " + mb.rationale.fillna("").astype(str)
    built["mmedbench"] = filter_and_label(txt, mb.sample_id, "mmedbench")
    logger.info("  on-topic: %d of %d English rows", len(built["mmedbench"]), len(mb))

    # ---- the cap: smallest corpus decides, so none has a size advantage ----
    cap = args.size or min(len(v) for v in built.values())
    logger.info("Matched size: %d documents per corpus (binding corpus: %s)",
                cap, min(built, key=lambda k: len(built[k])))

    rng = np.random.RandomState(SEED)
    for name, df in list(built.items()):
        s = balanced_sample(df, cap)
        s.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
        logger.info("  %-10s %d docs, %d conditions, mean %d words",
                    name, len(s), s.condition_group.nunique(),
                    s.text.str.split().str.len().mean())

    # ---- shuffled control: same documents, sentence order destroyed ----
    base = pd.read_csv(OUT / "multicare.csv")
    base["text"] = base.text.apply(lambda t: shuffle_sentences(t, rng))
    base["doc_id"] = base.doc_id.str.replace("multicare:", "shuffled:", regex=False)
    base.to_csv(OUT / "shuffled.csv", index=False, encoding="utf-8")
    logger.info("  %-10s %d docs (MultiCaRe, sentences shuffled within document)",
                "shuffled", len(base))

    summary = pd.DataFrame([
        {"corpus": n, "docs": len(pd.read_csv(OUT / f"{n}.csv")),
         "conditions": pd.read_csv(OUT / f"{n}.csv").condition_group.nunique()}
        for n in ["multicare", "pubmedqa", "mmedbench", "shuffled"]])
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
