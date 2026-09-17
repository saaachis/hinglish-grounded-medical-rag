"""Translate-then-retrieve baseline (ICCSDI review, Reviewer 2).

Reviewer 2: "include adequate comparisons with relevant existing methods."

The standard existing answer to a code-mixed query is not a cross-lingual
encoder -- it is to TRANSLATE the query into the corpus language first and then
retrieve monolingually. The submitted paper never tests it, so a reader can
reasonably ask why the cross-lingual route is preferable at all. This script
supplies that comparison.

Four query forms are retrieved with the SAME systems and the same relevance
criterion as Table 1:

    hinglish     the deployed query, as the user wrote it
    translated   the same query machine-translated to English by an LLM
    english      the gold human English question -- the upper bound, and what
                 the translation is trying to reproduce

The interesting quantity is how much of the hinglish -> english gap translation
recovers, and at what cost: translation adds an LLM call and its latency to every
query, and inherits that model's availability (the paper's own reproducibility
hazard), while a cross-lingual encoder adds nothing at query time.

Cost control: one short call per query, checkpointed after every batch, so an
interrupted run resumes. Groq enforces 200,000 tokens/day per ORGANISATION per
model, so a stratified subset is the default rather than all 3,015 queries;
paired tests on a subset remain valid, with less power.

Writes results/translation_baseline/.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import h1_real_retrieval as groqmod
from src.analysis.h4_retrieval import bootstrap_delta, mcnemar, strip_caption
from src.evaluation.retrieval_metrics import rank_metrics, success_at_k
from src.retrieval.passage_index import pool_passages_to_cases

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PAIRS = Path("data/processed/mmcqsd_multicare_paired.csv")
META = Path("data/faiss_index/evidence_metadata.csv")
CACHE = Path("data/passage_index")
OUT = Path("results/translation_baseline")
TRANSLATIONS = OUT / "translations.csv"

SYSTEM = ("You are a translator. Translate the user's Hinglish (romanised "
          "Hindi-English) patient question into natural English. Output ONLY the "
          "English translation: no preamble, no explanation, no quotation marks.")
TOP_K = 10
FUSION_DEPTH = 100
PASSAGE_POOL = 400
RRF_K = 60
SEED = 42
CHECKPOINT_EVERY = 25


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(s).lower())


def stratified_sample(pairs: pd.DataFrame, n: int) -> pd.DataFrame:
    """Proportional sample across condition groups, so topic mix is preserved."""
    if n >= len(pairs):
        return pairs.copy()
    frac = n / len(pairs)
    picked = (pairs.groupby("condition_query", group_keys=False)
                   .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=SEED)))
    return picked.sample(min(n, len(picked)), random_state=SEED).sort_index()


def translate(sample: pd.DataFrame, model: str) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    done: dict[int, str] = {}
    if TRANSLATIONS.exists():
        prev = pd.read_csv(TRANSLATIONS)
        # Empty cells read back as NaN. Dropping them here means a query whose
        # translation failed is RETRIED on the next run instead of being cached
        # as the literal string "nan" and silently entering the comparison.
        prev = prev[prev.translated.notna() & (prev.translated.astype(str).str.len() > 3)]
        done = dict(zip(prev.pair_id, prev.translated.astype(str)))
        logger.info("resuming: %d usable translations already cached", len(done))

    todo = [r for r in sample.itertuples() if r.pair_id not in done]
    if todo:
        groqmod.MODEL = model
        client = groqmod.RotatingGroq(groqmod.load_keys())
        logger.info("translating %d queries with %s", len(todo), model)
        for i, r in enumerate(todo, 1):
            # Reasoning tokens count against max_tokens, so a small budget returns
            # empty content on longer queries. 1500 leaves room for both.
            out = clean(client.chat(SYSTEM, str(r.hinglish_query),
                                    max_tokens=1500, reasoning_effort="low"))
            if out.startswith("[QUOTA_EXHAUSTED"):
                logger.error("quota exhausted after %d new translations -- "
                             "re-run later to resume", i - 1)
                break
            if len(out) <= 3 or out.startswith("[API_ERROR"):
                # Some completions come back empty. Do not cache them: leaving the
                # id absent makes the next run retry it.
                logger.warning("empty translation for pair %s -- will retry on re-run", r.pair_id)
                continue
            done[r.pair_id] = out
            if i % CHECKPOINT_EVERY == 0:
                _save(done)
                logger.info("  %d/%d", i, len(todo))
        _save(done)

    sample = sample.copy()
    sample["translated"] = sample.pair_id.map(done)
    missing = sample.translated.isna().sum()
    if missing:
        logger.warning("%d queries untranslated; they are dropped from the comparison", missing)
    return sample[sample.translated.notna()].reset_index(drop=True)


def clean(text: str) -> str:
    """Strip the preambles instruction-tuned models add despite being told not to."""
    t = str(text).strip().strip('"')
    t = re.sub(r"^(here('s| is) the (english )?translation:?|translation:|english:)\s*",
               "", t, flags=re.I).strip()
    return t


def _save(done: dict[int, str]) -> None:
    pd.DataFrame({"pair_id": list(done), "translated": list(done.values())}).to_csv(
        TRANSLATIONS, index=False, encoding="utf-8")


def rrf(*rankings: np.ndarray, top_k: int = TOP_K) -> np.ndarray:
    n = rankings[0].shape[0]
    fused = np.full((n, top_k), -1, dtype=np.int64)
    for q in range(n):
        score: dict[int, float] = {}
        for r in rankings:
            for rank, c in enumerate(r[q]):
                if c >= 0:
                    score[int(c)] = score.get(int(c), 0.0) + 1.0 / (RRF_K + rank + 1)
        for slot, (c, _) in enumerate(sorted(score.items(), key=lambda kv: -kv[1])[:top_k]):
            fused[q, slot] = c
    return fused


def main() -> None:
    ap = argparse.ArgumentParser(description="Translate-then-retrieve baseline")
    ap.add_argument("--n", type=int, default=1000, help="queries to sample (0 = all)")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--translate-only", action="store_true")
    args = ap.parse_args()

    pairs = pd.read_csv(PAIRS)
    meta = pd.read_csv(META)
    sample = stratified_sample(pairs, args.n or len(pairs))
    logger.info("sample: %d queries over %d condition groups",
                len(sample), sample.condition_query.nunique())

    sample = translate(sample, args.model)
    if args.translate_only or sample.empty:
        return

    variants = {
        "hinglish": sample.hinglish_query.astype(str).tolist(),
        "translated": sample.translated.astype(str).tolist(),
        "english": sample.english_summary.astype(str).apply(strip_caption).tolist(),
    }

    import faiss
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    from src.encoding.text_encoder import TextEncoder

    pf = pd.read_parquet(CACHE / "passages.parquet")
    pemb = np.load(CACHE / "passage_emb.npy")
    pix = faiss.IndexFlatIP(pemb.shape[1])
    pix.add(pemb.astype(np.float32))
    case_row = pf.case_row.to_numpy()

    enc = TextEncoder(device="cpu")
    enc.load_model()
    enc.model.max_seq_length = 256

    docs = meta.case_text.astype(str).tolist()
    logger.info("building BM25 / TF-IDF ...")
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    vec = TfidfVectorizer(min_df=2, max_features=200_000, ngram_range=(1, 2))
    D = vec.fit_transform(docs)

    gold = sample.condition_query.to_numpy()
    cond = meta.condition_group.to_numpy()
    n_relevant = np.array([(cond == g).sum() for g in gold])

    rows, hits = [], {}
    for name, texts in variants.items():
        logger.info("=== %s ===", name)
        qemb = enc.encode(texts, batch_size=32, show_progress=False)
        s, p = pix.search(qemb.astype(np.float32), PASSAGE_POOL)
        dense = pool_passages_to_cases(s, p, case_row, FUSION_DEPTH)
        lex = np.vstack([np.argsort(bm25.get_scores(tokenize(t)))[::-1][:FUSION_DEPTH]
                         for t in texts])
        tfidf = np.argsort(-linear_kernel(vec.transform(texts), D), axis=1)[:, :FUSION_DEPTH]
        for sysname, ranking in [("LaBSE-passages", dense[:, :TOP_K]),
                                 ("BM25-full", lex[:, :TOP_K]),
                                 ("TFIDF-full", tfidf[:, :TOP_K]),
                                 ("Hybrid-RRF", rrf(dense, lex))]:
            h = np.where(ranking >= 0, cond[np.clip(ranking, 0, len(cond) - 1)] == gold[:, None], False)
            hits[(sysname, name)] = h
            rows.append({"system": sysname, "query": name, **rank_metrics(h, n_relevant)})
            logger.info("  %-16s S@1=%.4f", sysname, rows[-1]["success@1"])

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "translation_metrics.csv", index=False)

    tests = []
    for sysname in ("Hybrid-RRF", "LaBSE-passages", "BM25-full", "TFIDF-full"):
        for a, b in [("hinglish", "translated"), ("translated", "english"), ("hinglish", "english")]:
            x, y = success_at_k(hits[(sysname, a)], 1), success_at_k(hits[(sysname, b)], 1)
            _, _, p = mcnemar(x, y)
            lo, hi = bootstrap_delta(y, x)
            tests.append({"system": sysname, "comparison": f"{b} - {a}",
                          "delta_success@1": y.mean() - x.mean(), "ci_lo": lo, "ci_hi": hi,
                          "mcnemar_p": p, "n": len(sample)})
    tdf = pd.DataFrame(tests)
    tdf.to_csv(OUT / "translation_tests.csv", index=False)
    write_report(res, tdf, sample, args.model)


def write_report(res: pd.DataFrame, tests: pd.DataFrame, sample: pd.DataFrame, model: str) -> None:
    m = res.set_index(["system", "query"])
    L = ["# Translate-then-retrieve baseline", "",
         f"n = {len(sample)} queries (stratified by condition group). Translator: `{model}`.", "",
         "Reviewer 2 asked for comparison with existing methods. Translating the query into the",
         "corpus language and retrieving monolingually is the standard alternative to a",
         "cross-lingual encoder; this is that comparison, on the same index and criterion as Table 1.", "",
         "## Example", "",
         "| Form | Text |", "|---|---|",
         f"| hinglish | {str(sample.hinglish_query.iloc[0])[:160]} |",
         f"| translated | {str(sample.translated.iloc[0])[:160]} |",
         f"| english (gold) | {strip_caption(str(sample.english_summary.iloc[0]))[:160]} |", "",
         "## Retrieval quality", "",
         "| System | Query form | S@1 | S@5 | S@10 | MRR@10 | nDCG@10 |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        L.append(f"| `{r.system}` | {r['query']} | {r['success@1']:.4f} | {r['success@5']:.4f} | "
                 f"{r['success@10']:.4f} | {r['MRR@10']:.4f} | {r['nDCG@10']:.4f} |")
    L += ["", "## Paired tests on Success@1", "",
          "| System | Comparison | Delta | 95% CI | McNemar p |", "|---|---|---:|---|---:|"]
    for _, r in tests.iterrows():
        L.append(f"| `{r.system}` | {r.comparison} | **{r['delta_success@1']:+.4f}** | "
                 f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {r.mcnemar_p:.3g} |")

    L += ["", "## Recovery of the code-mixing penalty", "",
          "| System | Hinglish | Translated | English | Gap recovered by translation |",
          "|---|---:|---:|---:|---:|"]
    for sysname in res.system.unique():
        hi = float(m.loc[(sysname, "hinglish"), "success@1"])
        tr = float(m.loc[(sysname, "translated"), "success@1"])
        en = float(m.loc[(sysname, "english"), "success@1"])
        rec = (tr - hi) / (en - hi) if en > hi else float("nan")
        L.append(f"| `{sysname}` | {hi:.4f} | {tr:.4f} | {en:.4f} | "
                 f"{'n/a' if not np.isfinite(rec) else f'{rec:.1%}'} |")
    L += ["", "Translation costs one LLM call per query, adds that model's latency to every search,",
          "and inherits its availability -- the same hosted-model dependency this paper reports as a",
          "reproducibility hazard. A cross-lingual encoder costs nothing extra at query time.",
          "Whether the recovered accuracy is worth that is the trade-off practitioners face.", ""]
    (OUT / "translation_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
