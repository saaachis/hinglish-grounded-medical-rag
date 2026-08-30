# Zenodo Upload — Data Artifacts

> ## ✅ DEPOSITED — 30 August 2026
> **DOI:** https://doi.org/10.5281/zenodo.22166651
> Published, public, CC-BY 4.0. Verified live and correctly attributed.
> `data.zip` (266.1 MB on Zenodo) contains all 8 files listed below;
> a local control zip of the same 8 files measured 253.8 MB, consistent.
>
> Cite as: Devika Jonjale, & Saachi Shinde. (2026). *Data artifacts for
> "The Code-Mixing Penalty in Clinical Retrieval-Augmented Generation:
> A Measurement Study for Hinglish Clinical Decision Support"* [Dataset].
> Zenodo. https://doi.org/10.5281/zenodo.22166651
>
> ⚠️ The record carries the authors' names, so the **anonymised** manuscript
> deliberately does NOT cite this DOI — it says the DOI is withheld for review.

**The `data/` payload is excluded from git** via `.gitignore`; only this README is
tracked. The upload is already complete — this file documents what was deposited.

## What's inside (preserve this directory structure)

```
data/
├── processed/
│   ├── mmcqsd_multicare_paired.csv      11 MB   3,015 query-evidence pairs
│   └── multicare_filtered.csv          261 MB   61,316 filtered MultiCaRe cases
├── faiss_index/
│   ├── evidence.index                   29 MB   FAISS IndexFlatIP (10,000 cases)
│   └── evidence_metadata.csv            42 MB   index metadata + case text
└── passage_index/
    ├── passages.parquet                 22 MB   41,746 passage chunks
    ├── passage_emb.npy                 122 MB   LaBSE embeddings, (41746, 768) float32
    ├── q_Q1_hinglish_256.npy             9 MB   cached Hinglish query embeddings
    └── q_Q2_english_question_256.npy     9 MB   cached English query embeddings
```

**Total: ~504 MB.** Well within Zenodo's 50 GB free-tier limit.

## Verification performed before staging

- All four items' presence and dimensions checked against what
  `src/analysis/retrieval_v2.py` expects (`CACHE = Path("data/passage_index")`,
  filenames `passages.parquet` / `passage_emb.npy` / `q_{variant}_256.npy`).
- `passage_emb.npy`: shape (41746, 768) — matches the paper's stated 41,746 passages
  and LaBSE's 768-dim output.
- `q_Q1_hinglish_256.npy` / `q_Q2_english_question_256.npy`: shape (3015, 768) each —
  matches the 3,015 evaluation pairs.
- **End-to-end reproduction:** ran `python -m src.analysis.retrieval_v2` against this
  exact cache. It reproduces Table 1 closely but not bit-identically; see
  `results/_reproducibility_check/NOTE.md` for the measured differences and the
  diagnosed cause (arbitrary tie-breaking in the lexical rankers).

## How to upload

1. Zip the `data/` folder (keep the `processed/`, `faiss_index/`, `passage_index/`
   subfolder structure exactly as-is — the pipeline code reads these paths literally).
2. Go to [zenodo.org](https://zenodo.org), log in via GitHub.
3. **New upload** → drag in the zip.
4. Fill in the metadata below (§Zenodo metadata).
5. Publish → copy the resulting DOI.
6. Send the DOI back so it can replace the `[AUTHORS: insert Zenodo DOI]` placeholder
   in the paper's Data Availability statement.

## Zenodo metadata

**Upload type:** Dataset

**Title:**
```
Data artifacts for "The Code-Mixing Penalty in Clinical Retrieval-Augmented Generation: A Measurement Study for Hinglish Clinical Decision Support"
```

**Authors:** Devika Jonjale (NMIMS Nilkamal School of Mathematics, Applied Statistics
& Analytics); Saachi Shinde (same affiliation)

**Description:**
```
Derived data artifacts supporting the paper "The Code-Mixing Penalty in Clinical
Retrieval-Augmented Generation: A Measurement Study for Hinglish Clinical Decision
Support" (Jonjale & Shinde, ICCSDI 2026).

Contents:
- data/processed/mmcqsd_multicare_paired.csv: 3,015 query-evidence pairs linking
  Hinglish patient questions from MMCQS to English clinical case reports from
  MultiCaRe, with similarity scores and condition-group labels.
- data/processed/multicare_filtered.csv: 61,316 filtered MultiCaRe case reports
  (39,652 unique cases) balanced across 18 condition groups, used to build the
  retrieval index.
- data/faiss_index/: the FAISS IndexFlatIP evidence index (10,000 LaBSE-encoded
  cases) and its metadata, as used by the deployed retrieval system.
- data/passage_index/: LaBSE embeddings for 41,746 passage-chunked documents plus
  cached query embeddings, used for the matched-content retrieval comparison
  reported in Table 1.

These files are gitignored in the code repository (they exceed practical GitHub
size limits) and are required to reproduce Table 1 and the retrieval-stage results
without re-running the full pipeline, which otherwise takes several hours of CPU
encoding. Source datasets (MMCQS, MultiCaRe) are separately publicly available;
see the paper's Data Availability statement.

Code: https://github.com/saaachis/hinglish-grounded-medical-rag
```

**Keywords:** Hinglish, code-mixing, retrieval-augmented generation, clinical
question answering, cross-lingual retrieval, MultiCaRe, MMCQS

**License:** CC-BY 4.0

**Related identifiers:** "Is supplement to" → your GitHub repo URL
