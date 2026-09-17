"""Transliteration-aware retrieval baseline (ICCSDI review, Reviewer 1).

Reviewer 1 asked for "stronger transliteration-aware baselines". The submitted
paper shows MuRIL retrieving at chance on romanised Hinglish (0.0640 vs a 0.0626
floor) and recovering to 0.1821 on the same content in English, concludes that
the failure is SCRIPT mismatch rather than language mismatch, and then says the
remedy "is a hypothesis the study motivates but does not test".

This script tests it. The same MuRIL index is queried three ways:

    romanised     the deployed Hinglish query, exactly as the user wrote it
    devanagari    the same query transliterated to Devanagari, MuRIL's script
    english       the gold English question (caption-stripped) -- upper bound

If script mismatch is the mechanism, `devanagari` must move away from the random
floor and toward `english`. If it does not, the paper's mechanism claim is not
supported and must be weakened.

Transliteration comes from `src.analysis.hinglish_devanagari`: a hand-verified
lexicon of the most frequent romanised Hindi tokens (~87% of Hindi token
occurrences) with ITRANS rules for the tail. AI4Bharat IndicXlit, the trained
alternative, cannot be installed here -- it requires fairseq, which does not
build on Python 3.12 -- so the mapping is imperfect on the rule-handled tail and
any recovery it produces is a LOWER BOUND on what a trained transliterator would
achieve. A cached `data/processed/queries_devanagari.csv` overrides it if present.

English words inside the code-mixed query ("skin", "doctor") are left in Latin
script: transliterating them would corrupt content MuRIL can already read.

CPU only, no API calls. Writes results/translit_baseline/.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.cmi import build_english_vocab
from src.analysis.h4_retrieval import bootstrap_delta, mcnemar, strip_caption
from src.analysis.hinglish_devanagari import coverage, transliterate_text
from src.evaluation.retrieval_metrics import rank_metrics, success_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PAIRS = Path("data/processed/mmcqsd_multicare_paired.csv")
META = Path("data/faiss_index/evidence_metadata.csv")
XLIT_CSV = Path("data/processed/queries_devanagari.csv")
EMB_CACHE = Path("data/embeddings/translit")
OUT = Path("results/translit_baseline")

MODELS = {"MuRIL": "google/muril-base-cased"}
MAX_DOC_WORDS = 200      # matched to h4_baselines.py, so numbers are comparable
MAX_SEQ = 128
TOP_K = 10
MAX_CHARS = 1500      # queries are clipped before encoding; 128 tokens is far less anyway
ENCODE_CHUNK = 250    # checkpoint interval, so a crash says which chunk failed


def load_transliteration(pairs: pd.DataFrame) -> tuple[list[str], str]:
    """Devanagari forms of the Hinglish queries, plus a description of their provenance."""
    if XLIT_CSV.exists():
        x = pd.read_csv(XLIT_CSV)
        if len(x) == len(pairs) and "devanagari" in x.columns:
            logger.info("using transliteration cached in %s", XLIT_CSV)
            return x.devanagari.astype(str).tolist(), "cached in data/processed/queries_devanagari.csv"
        logger.warning("%s exists but does not match the pairs file; ignoring", XLIT_CSV)

    vocab = build_english_vocab()
    texts = pairs.hinglish_query.astype(str).tolist()
    cov = coverage(texts, vocab)
    logger.info("transliteration coverage: lexicon %.1f%% of Hindi tokens, rules %.1f%% of all tokens",
                100 * cov["lexicon_share_of_hindi"], 100 * cov["rules_share"])
    out = [transliterate_text(t, vocab)[0] for t in texts]
    source = (f"hand-verified lexicon ({100 * cov['lexicon_share_of_hindi']:.0f}% of Hindi tokens) "
              f"+ ITRANS rules for the remainder; English tokens left in Latin script "
              f"({100 * cov['english_share']:.0f}% of tokens)")
    return out, source


def encode(model, texts: list[str], tag: str) -> np.ndarray:
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    path = EMB_CACHE / f"{tag}.npy"
    if path.exists():
        emb = np.load(path)
        if emb.shape[0] == len(texts):
            logger.info("cache hit: %s", tag)
            return emb
    logger.info("encoding %s (%d texts) ...", tag, len(texts))
    # Encode in logged chunks, with each chunk checkpointed. A full-corpus call
    # died twice without a traceback (a native-level crash gives no Python error),
    # which left no way to tell which input caused it. MMCQS queries also run much
    # longer than the 200-word documents, so they are clipped first -- anything
    # past the 128-token limit is discarded by the model regardless.
    part = path.with_suffix(".partial.npy")
    clipped = [str(t)[:MAX_CHARS] if str(t).strip() else "." for t in texts]
    done: list[np.ndarray] = []
    if part.exists():
        prev = np.load(part)
        if prev.shape[0] <= len(clipped):
            done = [prev]
            logger.info("  resuming %s from %d rows", tag, prev.shape[0])
    start = sum(d.shape[0] for d in done)
    for i in range(start, len(clipped), ENCODE_CHUNK):
        batch = clipped[i:i + ENCODE_CHUNK]
        done.append(np.asarray(model.encode(batch, batch_size=8, show_progress_bar=False,
                                            normalize_embeddings=True), dtype=np.float32))
        np.save(part, np.vstack(done))
        logger.info("  %s %d/%d", tag, min(i + ENCODE_CHUNK, len(clipped)), len(clipped))
    emb = np.vstack(done)
    np.save(path, emb)
    part.unlink(missing_ok=True)
    return emb


def main() -> None:
    ap = argparse.ArgumentParser(description="Transliteration-aware retrieval baseline")
    ap.add_argument("--write-transliteration", action="store_true",
                    help="only write the rule-based transliteration CSV and exit")
    args = ap.parse_args()

    pairs = pd.read_csv(PAIRS)
    meta = pd.read_csv(META)
    deva, xlit_source = load_transliteration(pairs)

    if args.write_transliteration:
        XLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"pair_id": pairs.pair_id, "hinglish_query": pairs.hinglish_query,
                      "devanagari": deva}).to_csv(XLIT_CSV, index=False, encoding="utf-8")
        logger.info("wrote %s", XLIT_CSV)
        return

    variants = {
        "romanised": pairs.hinglish_query.astype(str).tolist(),
        "devanagari": deva,
        "english": pairs.english_summary.astype(str).apply(strip_caption).tolist(),
    }
    gold = pairs.condition_query.to_numpy()
    cond = meta.condition_group.to_numpy()
    n_relevant = np.array([(cond == g).sum() for g in gold])
    docs = [" ".join(str(t).split()[:MAX_DOC_WORDS]) for t in meta.case_text]

    import faiss
    from sentence_transformers import SentenceTransformer

    rows, hits = [], {}
    for label, model_id in MODELS.items():
        logger.info("=== %s ===", label)
        model = SentenceTransformer(model_id, device="cpu")
        model.max_seq_length = MAX_SEQ
        demb = encode(model, docs, f"{label}_docs")
        ix = faiss.IndexFlatIP(demb.shape[1])
        ix.add(demb)
        for name, texts in variants.items():
            qemb = encode(model, texts, f"{label}_q_{name}")
            _, idx = ix.search(qemb, TOP_K)
            h = cond[idx] == gold[:, None]
            hits[(label, name)] = h
            rows.append({"system": label, "query": name, **rank_metrics(h, n_relevant)})
            logger.info("  %-11s S@1=%.4f", name, rows[-1]["success@1"])
        del model, ix, demb

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "translit_metrics.csv", index=False)

    tests = []
    for label in MODELS:
        for a_name, b_name in [("romanised", "devanagari"), ("devanagari", "english"),
                               ("romanised", "english")]:
            a = success_at_k(hits[(label, a_name)], 1)
            b = success_at_k(hits[(label, b_name)], 1)
            _, _, p = mcnemar(a, b)
            lo, hi = bootstrap_delta(b, a)
            tests.append({"system": label, "comparison": f"{b_name} - {a_name}",
                          "delta_success@1": b.mean() - a.mean(), "ci_lo": lo, "ci_hi": hi,
                          "mcnemar_p": p, "n": len(pairs)})
    tdf = pd.DataFrame(tests)
    tdf.to_csv(OUT / "translit_tests.csv", index=False)
    write_report(res, tdf, xlit_source, variants, len(pairs))


def write_report(res: pd.DataFrame, tests: pd.DataFrame, xlit_source: str,
                 variants: dict[str, list[str]], n: int) -> None:
    floor = 0.0626
    m = res.set_index(["system", "query"])
    L = ["# Transliteration-aware retrieval baseline", "",
         f"n = {n} queries. CPU only, no API calls.", "",
         f"**Transliteration source:** {xlit_source}", "",
         "The paper reports MuRIL at the random floor on romanised Hinglish and attributes",
         "the failure to script mismatch. This tests that claim directly by giving MuRIL the",
         "same queries in its own script.", "",
         "## Example", "",
         "| Query form | Text |", "|---|---|"]
    for k in ("romanised", "devanagari", "english"):
        L.append(f"| {k} | {str(variants[k][0])[:150]} |")
    L += ["", "## Retrieval quality", "",
          "| System | Query form | S@1 | S@5 | S@10 | MRR@10 | nDCG@10 |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        L.append(f"| `{r.system}` | {r['query']} | {r['success@1']:.4f} | {r['success@5']:.4f} | "
                 f"{r['success@10']:.4f} | {r['MRR@10']:.4f} | {r['nDCG@10']:.4f} |")
    L += [f"| *random floor* | — | *{floor}* | | | | |", "",
          "## Paired tests on Success@1 (exact McNemar, bootstrap CI)", "",
          "| System | Comparison | Delta | 95% CI | p |", "|---|---|---:|---|---:|"]
    for _, r in tests.iterrows():
        L.append(f"| `{r.system}` | {r.comparison} | **{r['delta_success@1']:+.4f}** | "
                 f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {r.mcnemar_p:.3g} |")

    rom = float(m.loc[("MuRIL", "romanised"), "success@1"])
    dev = float(m.loc[("MuRIL", "devanagari"), "success@1"])
    eng = float(m.loc[("MuRIL", "english"), "success@1"])
    gap = eng - rom
    recovered = (dev - rom) / gap if gap > 0 else float("nan")
    p_dev = float(tests.query("comparison == 'devanagari - romanised'").mcnemar_p.iloc[0])
    L += ["", "## Reading", "",
          f"- Romanised {rom:.4f} (floor {floor}) -> Devanagari {dev:.4f} -> English {eng:.4f}.",
          f"- Transliteration recovers **{recovered:.1%}** of the romanised-to-English gap "
          f"(McNemar p = {p_dev:.3g}).", ""]
    if p_dev < 0.05 and dev > rom:
        L += ["**The script-mismatch mechanism is supported.** Giving MuRIL the same content in",
              "its own script significantly improves retrieval, so the paper can report this as a",
              "tested result rather than a motivated hypothesis.", ""]
    else:
        L += ["**The mechanism is NOT supported by this test.** Either the script is not the",
              "binding constraint, or the transliteration is too approximate to carry the",
              "content. With a rule-based transliteration the second cannot be excluded, so",
              "report the result as inconclusive and keep the mechanism claim as a hypothesis.", ""]
    (OUT / "translit_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
