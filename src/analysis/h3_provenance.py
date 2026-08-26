"""H03: does the KIND of evidence change grounded answer quality?

    H03 (null): using authoritative clinical case evidence does not improve the
    factual correctness of Hinglish explanations relative to general biomedical text.

Four evidence conditions, topically matched and equal-sized (see h3_build_corpora):

    multicare   clinical case narratives    authoritative, case-level
    pubmedqa    research abstracts          general biomedical
    mmedbench   exam / didactic text        instructional
    shuffled    MultiCaRe, sentences shuffled within each document

WHAT IS HELD CONSTANT. Same queries, same prompt, same generator, same scoring, same
retriever, same index size. ONLY the corpus changes. Each corpus is searched with its
own index, so each condition gets the best evidence ITS corpus can offer -- the
comparison is between evidence types, not between retrievers.

WHY THE SHUFFLED CONTROL. It holds topic and vocabulary exactly constant while
destroying discourse order. If shuffled MultiCaRe scores like intact MultiCaRe, then
what grounding extracts is a bag of clinical terms rather than a coherent case, which
would materially qualify any provenance claim. It is the control that makes a positive
H03 result interpretable.

SCORING. Answers are scored against the UNBIASED caption reference, not against the
evidence they were conditioned on. Scoring each arm against its own evidence would
guarantee that whichever corpus is most quotable wins -- the same circularity that
inflates H01 -- and here it would invalidate the comparison outright, because the four
corpora differ in style and therefore in quotability.

No new retrieval index for the queries: each corpus is encoded and searched directly.
Writes results/h3_provenance/.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.h1_real_retrieval import (
    MODEL, SYSTEM_GROUNDED, RotatingGroq, build_prompt, is_refusal, load_keys,
)
from src.encoding.text_encoder import TextEncoder
from src.evaluation.caption_reference import extract_description
from src.evaluation.concept_lexicon import extract_concepts

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CORPORA = Path("data/h3_corpora")
PAIRS = Path("data/processed/mmcqsd_multicare_paired.csv")
OUT = Path("results/h3_provenance")
CACHE = Path("data/h3_corpora/emb")

CONDITIONS = ["multicare", "pubmedqa", "mmedbench", "shuffled"]
MAX_SEQ = 256
SEED = 42


def prf(answer: str, reference: str) -> tuple[float, float, float]:
    o, r = extract_concepts(answer), extract_concepts(reference)
    if not o or not r:
        return np.nan, np.nan, np.nan
    hit = len(o & r)
    p, rc = hit / len(o), hit / len(r)
    f = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0
    return p, rc, f


def encode_corpus(enc: TextEncoder, name: str, texts: list[str]) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npy"
    if f.exists():
        e = np.load(f)
        if e.shape[0] == len(texts):
            logger.info("  cache hit: %s", name)
            return e
    logger.info("  encoding %s (%d docs) ...", name, len(texts))
    e = enc.encode(texts, batch_size=32, show_progress=False)
    np.save(f, e)
    return e


def main() -> None:
    ap = argparse.ArgumentParser(description="H03 evidence provenance")
    ap.add_argument("--n-queries", type=int, default=400)
    ap.add_argument("--model", type=str, default=MODEL)
    args = ap.parse_args()

    import src.analysis.h1_real_retrieval as H
    H.MODEL = args.model

    pairs = pd.read_csv(PAIRS)
    q = (pairs.groupby("condition_query", group_keys=True)
         .apply(lambda g: g.sample(n=max(2, int(args.n_queries * len(g) / len(pairs))),
                                   random_state=SEED), include_groups=False)
         .reset_index(level=0).reset_index(drop=True))
    q["caption_ref"] = q.english_summary.apply(extract_description)
    q = q[q.caption_ref.str.len() > 0].reset_index(drop=True)
    logger.info("queries: %d (all with a caption reference)", len(q))

    enc = TextEncoder(device="cpu")
    enc.load_model()
    enc.model.max_seq_length = MAX_SEQ
    qemb = encode_corpus(enc, "queries", q.hinglish_query.astype(str).tolist())

    # Retrieve the single best document from EACH corpus for every query.
    retrieved: dict[str, list[dict]] = {}
    for name in CONDITIONS:
        f = CORPORA / f"{name}.csv"
        if not f.exists():
            logger.warning("missing corpus %s -- run h3_build_corpora first", f)
            continue
        c = pd.read_csv(f)
        emb = encode_corpus(enc, name, c.text.astype(str).tolist())
        ix = faiss.IndexFlatIP(emb.shape[1])
        ix.add(emb.astype(np.float32))
        _, idx = ix.search(qemb.astype(np.float32), 1)
        retrieved[name] = [
            {"doc_id": c.doc_id.iloc[int(i[0])], "text": str(c.text.iloc[int(i[0])]),
             "condition_group": str(c.condition_group.iloc[int(i[0])])} for i in idx]
        hit = np.mean([r["condition_group"] == g
                       for r, g in zip(retrieved[name], q.condition_query)])
        logger.info("  %-10s retrieval top-1 condition match: %.1f%%", name, 100 * hit)

    OUT.mkdir(parents=True, exist_ok=True)
    partial = OUT / "h3_partial.csv"
    records: list[dict] = []
    start = 0
    if partial.exists():
        records = pd.read_csv(partial).to_dict("records")
        start = len(records)
        logger.info("resuming at %d", start)

    client = RotatingGroq(load_keys())

    for i in range(start, len(q)):
        row = q.iloc[i]
        rec = {"pair_id": row.pair_id, "condition": row.condition_query,
               "caption_ref": row.caption_ref, "model": args.model}
        quota_hit = False
        for name in CONDITIONS:
            if name not in retrieved:
                continue
            ev = retrieved[name][i]
            out = client.chat(SYSTEM_GROUNDED,
                              build_prompt(str(row.hinglish_query), ev["text"]))
            if out == "[QUOTA_EXHAUSTED]":
                quota_hit = True
                break
            p, rc, f1 = prf(out, row.caption_ref)
            rec[f"{name}_output"] = out
            rec[f"{name}_precision"] = p
            rec[f"{name}_recall"] = rc
            rec[f"{name}_f1"] = f1
            rec[f"{name}_refusal"] = is_refusal(out)
            rec[f"{name}_doc"] = ev["doc_id"]
            rec[f"{name}_cond_match"] = ev["condition_group"] == row.condition_query
            # Evidence length is a REAL confound and must be measurable, not hidden:
            # case reports average 657 words against 203 for abstracts, and
            # build_prompt caps at 400, so MultiCaRe alone hits the cap and is handed
            # more text than the others. That is a genuine property of the evidence
            # type, but it must be reported alongside any provenance claim.
            rec[f"{name}_ev_words"] = min(len(ev["text"].split()), 400)
            time.sleep(0.4)
        if quota_hit:
            pd.DataFrame(records).to_csv(partial, index=False, encoding="utf-8")
            logger.error("quota exhausted at query %d -- re-run to resume", i)
            break
        records.append(rec)
        if (i + 1) % 10 == 0:
            pd.DataFrame(records).to_csv(partial, index=False, encoding="utf-8")
            logger.info("[%d/%d]", i + 1, len(q))

    df = pd.DataFrame(records)
    df.to_csv(OUT / "h3_scored.csv", index=False, encoding="utf-8")
    logger.info("wrote %d rows", len(df))
    if len(df) >= 30:
        report(df)


def report(df: pd.DataFrame) -> None:
    present = [c for c in CONDITIONS if f"{c}_f1" in df]
    L = ["# H03: evidence provenance", "",
         f"n = {len(df)} queries x {len(present)} evidence conditions. Corpora are "
         "topically matched and equal-sized; only the evidence type changes.",
         "Answers are scored against the UNBIASED caption reference, never against the "
         "evidence they were given.", "",
         "| Evidence | mean F1 | mean precision | mean recall | refusal | n scoreable |",
         "|---|---:|---:|---:|---:|---:|"]
    for c in present:
        L.append(f"| `{c}` | {df[f'{c}_f1'].mean():.4f} | {df[f'{c}_precision'].mean():.4f} | "
                 f"{df[f'{c}_recall'].mean():.4f} | {df[f'{c}_refusal'].mean():.1%} | "
                 f"{df[f'{c}_f1'].notna().sum()} |")

    wcols = [f"{c}_ev_words" for c in present if f"{c}_ev_words" in df]
    if wcols:
        L += ["", "## Evidence length actually supplied (confound check)", "",
              "| Evidence | mean words in prompt |", "|---|---:|"]
        for c in present:
            if f"{c}_ev_words" in df:
                L.append(f"| `{c}` | {df[f'{c}_ev_words'].mean():.0f} |")
        L += ["", "The prompt caps evidence at 400 words. Case narratives are long enough "
              "to hit that cap while abstracts and exam text are not, so MultiCaRe is "
              "handed more text. Any advantage it shows is therefore an upper bound on a "
              "provenance effect, and partly a length effect.", ""]

    sub = df[[f"{c}_f1" for c in present]].dropna()
    L += ["", f"## Omnibus test (n = {len(sub)} complete cases)", ""]
    if len(sub) >= 10 and len(present) >= 3:
        chi, p = stats.friedmanchisquare(*[sub[f"{c}_f1"] for c in present])
        L.append(f"Friedman chi-square = {chi:.3f}, **p = {p:.4g}**")
        L.append("")
        if p < 0.05:
            L += ["Evidence type significantly affects grounded answer quality; H03 is "
                  "**rejected**.", "", "### Post-hoc (Wilcoxon vs MultiCaRe, Bonferroni)", "",
                  "| Comparison | delta F1 | p (corrected) |", "|---|---:|---:|"]
            k = len(present) - 1
            for c in present:
                if c == "multicare":
                    continue
                d = sub["multicare_f1"] - sub[f"{c}_f1"]
                pv = stats.wilcoxon(d)[1] if d.abs().sum() > 0 else 1.0
                L.append(f"| multicare - {c} | {d.mean():+.4f} | {min(pv*k,1.0):.4g} |")
        else:
            L += ["**H03 is NOT rejected.** With topicality and corpus size held constant, "
                  "evidence type does not significantly change grounded answer quality. "
                  "Note this is a genuine null, not the artefact an unmatched comparison "
                  "would have produced.", ""]

    if "shuffled_f1" in df and "multicare_f1" in df:
        s = df[["multicare_f1", "shuffled_f1"]].dropna()
        if len(s) >= 10:
            d = s.multicare_f1 - s.shuffled_f1
            pv = stats.wilcoxon(d)[1] if d.abs().sum() > 0 else 1.0
            L += ["", "## Does discourse structure matter?", "",
                  f"MultiCaRe minus sentence-shuffled MultiCaRe: **{d.mean():+.4f}** "
                  f"(n = {len(s)}, p = {pv:.4g}).", "",
                  ("Shuffling sentences does not significantly change answer quality, so "
                   "what grounding extracts behaves like a bag of clinical terms rather "
                   "than a coherent narrative. Any provenance claim must be qualified "
                   "accordingly." if pv >= 0.05 else
                   "Intact narrative outperforms shuffled text, so discourse structure "
                   "carries information the generator uses.")]

    (OUT / "h3_report.md").write_text("\n".join(L), encoding="utf-8")
    pd.DataFrame([{"corpus": c, "f1": df[f"{c}_f1"].mean(),
                   "precision": df[f"{c}_precision"].mean(),
                   "recall": df[f"{c}_recall"].mean(),
                   "refusal": df[f"{c}_refusal"].mean()} for c in present]).to_csv(
        OUT / "h3_summary.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    main()
