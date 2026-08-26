# Grounded RAG for Hinglish Clinical Queries

Evidence-grounded retrieval-augmented generation for **Hinglish** (Hindi–English
code-switched) patient questions, evaluated against English clinical case reports.

**Research Discourse I · Group 5**

---

## What this repository contains

A retrieve-then-generate pipeline and — more importantly — **a careful measurement
study of it**. Several results here are negative or corrective, and they are
reported as such.

| Stage | Implementation |
|---|---|
| Query encoding | `sentence-transformers/LaBSE`, 768-d, CPU, `max_seq_length=256` |
| Index | FAISS `IndexFlatIP` over 10,000 MultiCaRe cases, passage-chunked (41,746 passages) |
| Retrieval | Dense, BM25, TF-IDF, and hybrid reciprocal-rank fusion |
| Generation | Groq `openai/gpt-oss-120b`, grounded vs zero-shot prompts |
| Scoring | Concept precision / recall / F1 against two references, with degenerate baselines |
| Statistics | Wilcoxon, McNemar, bootstrap CIs, Benjamini–Hochberg across the test family |

> **This is a text-only system.** It does not use LLaVA, BioMedCLIP, QLoRA or DPO.
> Earlier versions of this README advertised those; they were never implemented and
> the claims have been removed.

---

## Headline results

**Retrieval — code-mixing damages lexical retrieval far more than dense retrieval.**
Recall@1 over 3,015 queries, every system reading the same full case text:

| System | Hinglish query | English question | Penalty | McNemar *p* |
|---|---:|---:|---:|---:|
| Hybrid (RRF) | **0.1751** | 0.1973 | +0.0222 | 0.018 |
| LaBSE (passages) | 0.1280 | 0.1486 | +0.0206 | 0.017 |
| BM25 | 0.0935 | 0.1847 | **+0.0912** | 9.9e-26 |
| TF-IDF | 0.0842 | 0.1529 | — | — |
| *random floor* | *0.0626* | *0.0626* | — | — |

BM25 wins on English and collapses on Hinglish; dense retrieval barely moves. The
penalty is **4.4× larger for lexical retrieval**, which is the empirical argument
for cross-lingual embedding in this setting.

**MuRIL sits at the random floor on Hinglish** (0.0640 vs 0.0626) yet recovers to
0.1821 on the same content in English. It is trained on *Devanagari* Hindi while
MMCQSD is *romanised*: **script mismatch, not language mismatch**, is what breaks it.

**Generation — the apparent grounding benefit depends on what you score against.**
The same generations, scored twice (concept F1, grounded − zero-shot):

| Source / arm | Circular reference | Unbiased reference |
|---|---:|---:|
| llama-3.1-8b, oracle evidence | +0.203 | +0.062 |
| gpt-oss-120b, oracle evidence | +0.093 | −0.021 (n.s.) |
| gpt-oss-120b, real retrieval | −0.032 (n.s.) | **−0.047** |

The *circular* reference is the evidence the grounded arm was conditioned on and the
zero-shot arm never saw. The *unbiased* one is the MMCQSD image description, which
neither arm saw. Grounding trades recall for precision, and that trade is only
sometimes profitable.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY=gsk_...

python build_index.py         # build the FAISS index
streamlit run app.py          # demo
pytest -q                     # 44 tests
```

Reproduce every reported number from cached artifacts:

```bash
python -m src.analysis.reproduce_all
```

---

## Layout

```
src/
  encoding/text_encoder.py       LaBSE wrapper
  retrieval/
    indexer.py                   FAISS wrapper
    retriever.py                 top-k + (disabled) adaptive truncation
    passage_index.py             passage chunking, max-pool to cases
    passage_retriever.py         case retrieval over the passage index
  generation/generator.py        Groq grounded / zero-shot generation
  evaluation/
    concept_lexicon.py           canonical lexicon (replaced 5 divergent copies)
    caption_reference.py         M4' unbiased reference + cluster statistics
    baselines.py                 degenerate baselines every table must carry
    relevance.py                 multi-label relevance criterion
    hypothesis.py                statistical test helpers
  analysis/                      every experiment and figure script
results/                         all outputs, version-controlled
plans/next-phase/                audits, handoffs, and the execution plan
```

---

## Known limitations

- **The original generator no longer exists.** `llama-3.1-8b-instant` produced the
  n=1,165 results and was decommissioned by Groq mid-project; it now returns 404.
  Results are replicated on `openai/gpt-oss-120b`.
- **Relevance labels are coarse** — 18 condition groups, so a same-group case can
  still be clinically irrelevant.
- **The concept lexicon is not clinician-validated**, and covers 26 concepts.
- **`precision` (formerly `factual_support`) has no recall term**, so its optimum is
  a one-word answer. Report paired deltas and F1, never an absolute level.
- **Adaptive truncation does not work** in this setting and is disabled; see
  `results/truncation_sweep/`.
- **H₀₃ (evidence provenance) is not implemented.** PubMedQA is only 2.1% on-topic
  for these conditions and MMedBench is 57.6% Chinese.

## Data availability

Corpora are public: MMCQSD, MultiCaRe. Derived artifacts (paired file, FAISS index,
passage index) are gitignored for size; all `results/` CSVs and figures are tracked.
