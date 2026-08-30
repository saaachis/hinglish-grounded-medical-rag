# Reproducibility check on `retrieval_v2` (internal note, not for the paper)

**Date:** 2026-08-30 · **Branch:** `devikas-hardening`

## What was done

`data/passage_index/` (the passage-embedding cache behind Table 1) was absent from
the repo and from both handoff zips. It was supplied separately and copied in, then
`python -m src.analysis.retrieval_v2` was run end-to-end to confirm it reproduces
Table 1.

## Result: close, but not identical

| System | Table 1 (committed) | This rerun | Same conclusion? |
|---|---|---|---|
| BM25-full | 0.0935 / 0.1847, p = 9.9e-26 | 0.0949 / 0.1828, p = 9.3e-24 | yes |
| LaBSE-passages | 0.1280 / 0.1486, p = 0.0168 | 0.1287 / 0.1486, p = 0.0208 | yes |
| Hybrid-RRF | 0.1751 / 0.1973, **p = 0.0184** | 0.1778 / 0.1947, **p = 0.0743** | **no — crosses 0.05** |
| TFIDF-full | 0.0842 / 0.1529 | 0.0779 / 0.1562 | n/a (no test reported) |

## Cause (diagnosed, not speculated)

`tie_diagnostic.log` in this folder:

- **6.8%** of Hinglish queries and **9.0%** of English queries have a **tie at the top
  TF-IDF score** (204 and 272 queries respectively).
- `np.argsort` (quicksort) breaks those ties arbitrarily, so top-1 — and therefore
  Recall@1 — is not stable across runs or environments.
- `TfidfVectorizer(max_features=200_000)` is a **binding** cap (vocabulary landed on
  exactly 200,000), so the feature cutoff is itself tie-broken and can differ across
  scikit-learn versions.
- Ruled out: stale outputs (rerun timestamps were current) and corpus drift
  (`evidence_metadata.csv` is **MD5-identical** to the March handoff, and
  `passages.parquet` case-IDs align with it perfectly, 0 mismatches / 10,000).

Note the supplied `passage_emb.npy` was never in the handoff zips, so it cannot be
confirmed byte-identical to the cache that produced Table 1. Dense/hybrid drift is
consistent with embeddings generated on different hardware or library versions.

## Decision taken

**Table 1 and Sect. 5.1 left unchanged.** The committed numbers come from a
legitimate run against the committed artifacts and are internally consistent; the
provenance of the replacement cache cannot be established. `results/retrieval_v2/`
was restored from git after the rerun overwrote it.

## Residual risk (for the authors' awareness)

Sect. 5.1 states "largest p = 0.018" and "H04 is rejected for every system tested."
A reviewer who reruns the pipeline in a different environment could obtain p > 0.05
for **Hybrid-RRF** specifically. BM25 (p ~ 1e-24) and LaBSE-passages are far from the
threshold and are unaffected; the paper's headline 4.4x lexical/dense asymmetry is
also unaffected. Only the hybrid system's individual significance claim is fragile.

If this is ever revisited, the correct engineering fix is deterministic tie-breaking
(stable sort, or an explicit secondary key on document index) in
`src/analysis/retrieval_v2.py`, followed by regenerating Table 1 and Fig. 1.

## Files here

- `h4_v2_tests_RERUN.csv`, `retrieval_v2_metrics_RERUN.csv` — the rerun's output
- `tie_diagnostic.log` — the tie-frequency measurement
