# Handover — Code Work Complete

**To:** Devika Jonjale
**From:** Saachi's machine · 2026-08-28 · branch `saachi-hardening` (pushed)
**Status:** **Implementation is finished.** No further code is required for the paper.

---

## 1. What this document is

The engineering phase is closed. Every hypothesis that can be answered with the
resources available has been answered, every result is committed and reproducible,
and the manuscript is drafted in the official template. What remains is writing and
administration, listed in §6.

Read this file first, then `HARDENING-REPORT.md` for the fuller technical history.

---

## 2. Results, final

| Hypothesis | Outcome | Key numbers |
|---|---|---|
| **H₀₁** Grounding effect | **Rejected** | +0.235, d = 0.576, p = 3.1×10⁻⁶⁴ (n=1,165); replicates at +0.224, d = 0.616 on a second generator |
| **H₀₂** Code-mixing robustness | **Rejected, per arm** | grounded flat (ρ = −0.001, p = 0.98); zero-shot degrades (ρ = −0.116, BH p = 0.0003) |
| **H₀₃** Evidence provenance | **Partially answered** | refusal differs by evidence type (Cochran's Q = 9.09, p = 0.028); answer quality undetermined |
| **H₀₄** Retrieval penalty | **Rejected, every system** | BM25 −0.0912 (p = 9.9×10⁻²⁶) vs dense −0.0206 (p = 0.017) |

### Table 1 — retrieval, n = 3,015

| System | Hinglish | English | Penalty | McNemar p |
|---|---:|---:|---:|---:|
| Hybrid (RRF) | **0.1751** | 0.1973 | +0.0222 | 0.018 |
| LaBSE (passages) | 0.1280 | 0.1486 | +0.0206 | 0.017 |
| BM25 | 0.0935 | 0.1847 | **+0.0912** | 9.9×10⁻²⁶ |
| TF-IDF | 0.0842 | 0.1529 | — | — |
| *random floor* | *0.0626* | *0.0626* | — | — |

**Lexical retrieval is ~4.4× more damaged by code-mixing than dense.** BM25 wins on
English and collapses on Hinglish. This crossover is the paper's empirical argument
for cross-lingual embedding.

**MuRIL sits at the random floor on romanised Hinglish** (0.0640 vs 0.0626) and
recovers to 0.1821 on the same content in English — *script* mismatch, not language.

### The measurement caution

The same generations scored against two references give different answers:

| Generator / evidence | Circular ref. | Unbiased ref. |
|---|---:|---:|
| llama-3.1-8b, oracle | +0.203 | +0.062 |
| gpt-oss-120b, oracle | +0.093 | −0.021 (n.s.) |
| gpt-oss-120b, real retrieval | −0.032 (n.s.) | −0.047 |

And a constant one-word answer (`"swelling"`) scores **0.7132** against the unbiased
reference while the real system scores **0.1528**. Concept precision has no recall
term, so its optimum is terseness. **Never quote an absolute level of this metric;
only paired deltas between arms scored identically are interpretable.**

---

## 3. Why H₀₃ stops where it does

H₀₃ needed a low-refusal prompt to answer its answer-quality arm: at 76–88% refusal,
a four-way test needs every arm to answer simultaneously, and that probability is
multiplicative — the omnibus ran on **13 of 160 rows**.

The fix works (refusal 67% → 0% on a probe) and is implemented behind
`--prompt direct`. It was **not** run to completion because of a hard external limit:

> Groq enforces **200,000 tokens/day per organisation, per model** — not per key.
> Adding keys to the same account adds no capacity. The run needs ~1.5M tokens,
> i.e. 7+ days of one organisation's quota.

So H₀₃ is reported as partially answered, with the reason stated in the paper. That
is honest and defensible. **Do not spend further days chasing it.**

If you ever want to finish it, the command is ready and resumes from checkpoint:

```
python -m src.analysis.h3_provenance --prompt direct --model openai/gpt-oss-120b \
       --out-dir results/h3_direct --n-queries 400
```

It needs Groq keys from **genuinely different accounts** (separate organisations) to
be feasible.

---

## 4. What changed in the code, and why it matters

Five defects were found and fixed. Each changed a reported number, so they are worth
knowing before you read older documents.

1. **Negation was English-only.** Hindi negation is post-posed, so every Hinglish
   denial scored as an assertion — `extract_concepts("rash nahi hai")` returned
   `{"rash"}`. Now scoped three ways (pre-posed, post-posed, epistemic hedges).
   The test that "covered" this was vacuous: its right disjunct was a constant
   `True`, so the suite passed while the bug was live.

2. **`factual_support` is precision-only.** See §2. `src/evaluation/baselines.py`
   ships degenerate baselines for every table.

3. **`hallucination` is exactly `1 − precision`** (verified min = max = 1.0). The
   published "+73.5% factuality **and** −44% hallucination" pair double-counts one
   result. Report one or the other, not both as independent.

4. **The index encoded ~15% of each case.** `build_index.py` capped text at 200
   words then truncated to 128 tokens, against a 554-word median case — while BM25
   read the full document. Passage chunking fixed it and **reversed** the
   BM25-beats-LaBSE finding.

5. **Five divergent concept lexicons** coexisted (18/7/24/26/24 concepts), so H₁ and
   the Phase-6 ablation were scored with different instruments. Unified into
   `src/evaluation/concept_lexicon.py`.

---

## 5. Repository state

- **Branch** `saachi-hardening`, pushed — **this is the branch to work from.** It is
  ahead of `main` and holds every fix, result, figure, the draft and the
  bibliography. Start from it rather than from `main`:
  `git fetch origin && git checkout saachi-hardening`
- **Tests**: 44 passing, 0 errors (two had errored since the first commit — pytest
  was collecting `hypothesis.py`'s `test_*` helpers as tests).
- **Reproducibility**: `python -m src.analysis.reproduce_all` regenerates every
  reported number and figure from cached artefacts; `--list` shows the 13 stages and
  an explicit list of what is *not* reproducible and why.
- **Repo hygiene**: README, `config.yaml` and `requirements.txt` now describe the
  system that exists. Five dead stub files removed (475 lines, 14
  `NotImplementedError`), including the `app/streamlit_app.py` the README pointed at.
  `requirements.txt` previously omitted `groq`, which is imported at module load.

### Where things live

| Path | Contents |
|---|---|
| `src/evaluation/concept_lexicon.py` | canonical lexicon (replaced five copies) |
| `src/evaluation/caption_reference.py` | the unbiased reference + cluster statistics |
| `src/evaluation/baselines.py` | degenerate baselines every table must carry |
| `src/analysis/retrieval_v2.py` | Table 1: chunking, matched content, hybrid RRF |
| `src/analysis/h3_provenance.py` | H₀₃, matched-topic corpora |
| `src/analysis/reproduce_all.py` | single entry point |
| `results/` | every result, version-controlled |
| `research-paper/make_paper.py` | builds the manuscript from the official template |

---

## 6. What is left — all of it writing, none of it code

| # | Task | Owner | Notes |
|---|---|---|---|
| 1 | **Related Work prose** | — | Citations are now supplied — see below. Only the prose is left |
| 2 | Library versions and compute in §4.3 | — | `pip freeze` for torch, sentence-transformers, faiss-cpu, scipy |
| 3 | Zenodo deposit → DOI | — | Required by the template's Data availability. ~380 MB |
| 4 | Repository URL + release tag | — | For Code availability |
| 5 | Acknowledgements | — | Supervisors, institution |
| 6 | Read the draft end to end | **both** | Every `[TO COMPLETE]` marker is a decision for you |

### The bibliography is done

`research-paper/references.bib` holds nine entries read straight out of the PDFs in
`research-work/papers/` — authors, venue, year and page ranges taken from the papers
themselves rather than recalled. Seven standard method citations (LaBSE, MuRIL, BM25,
RRF, FAISS, multilingual-E5, MultiCaRe) sit in a separate section marked `VERIFY`,
because those were written from general knowledge and their page ranges and author
lists have not been checked.

Three things the filenames get wrong, so check these before citing:

- The dataset is **MMCQS**, not "MMCQSD" as this repo calls it throughout, and it is
  introduced by **MedSumm** (Ghosh et al., 2024, arXiv:2401.01596). That is the most
  important citation in the paper — it is the source of every Hinglish query.
- **LLaVA-Med is NeurIPS 2023** (Datasets and Benchmarks track), not 2024.
- **Fact-Aware Multimodal Retrieval Augmentation is NAACL 2025**, not EMNLP 2024.

---

### One decision only the two of you can make

**The `hallucination` results.** They appear in the poster and the review-2 reports
as an independent finding. Since they are arithmetically `1 − precision`, they must
either be withdrawn or relabelled. The paper currently omits them. Decide together
and apply it consistently across poster, reports and paper.

---

## 7. The manuscript

`research-paper/draft/Hinglish_RAG_ICCSDI2026.docx` — built by filling the official
ICCSDI Word template in place, so every style, the Times New Roman body, the A4 page
and the title block are the template's own. Margins reduced slightly as requested;
nothing else about the layout was changed.

It contains only what the research established. Template scaffolding the study does
not use (worked equation, pseudocode block, optional appendix) was removed rather
than filled with placeholder text.

Rebuild after any results change:

```
python research-paper/make_paper.py
```

A LaTeX version of an earlier draft is at
`research-paper/draft/hinglish_rag_iccsdi2026.tex` if you prefer that route; the
Word version is the current one.

---

## 8. Honest assessment

The paper is **defensible rather than flattering**, and that is deliberate. Several
original claims did not survive scrutiny — the headline factuality gain shrinks
under an unbiased reference, adaptive truncation never worked, and a configuration
defect had reversed the main retrieval comparison. All are reported.

What replaced them is stronger: a measured code-mixing penalty with a mechanism
(script, not language), a demonstration that evaluation design changes the apparent
size of a RAG benefit, and two hypothesis rejections that replicate across generator
families.

Every number in the paper is produced by a committed script from committed data.
