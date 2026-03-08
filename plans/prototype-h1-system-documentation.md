# H1 Prototype System Documentation

> End-to-end technical documentation for the current MVP prototype that evaluates **H1**:
> grounded RAG outputs are more factual and less hallucinated than zero-shot outputs.

---

## 1) Prototype Objective

The prototype compares two response modes for the same Hinglish medical query:

1. **Zero-shot** (no retrieved evidence in response construction)
2. **Grounded** (response constructed using retrieved report evidence)

Then it evaluates factuality and hallucination differences across all samples.

---

## 2) Data Used (Current Prototype Run)

## 2.1 Is the data real or synthetic?

For the current run, data is **synthetic radiology-style seed data** (MVP mode), not full real Open-i downloads.

- Report texts are from a curated mini pool in `src/prototype/build_hmg_mini.py` (`SEED_REPORTS`).
- Hinglish queries are generated from those reports using template logic + anchor terms.
- This produces `HMG-mini` for fast prototype validation.

This is intentional for 7-day execution speed and reproducibility.

## 2.2 Dataset used

### Primary runtime dataset
- **HMG-mini** (`data/processed/hmg_mini.csv`)
- Current size: **150** samples (deterministic seed run)
- Columns:
  - `sample_id`
  - `query_hinglish`
  - `report_id`
  - `report_text`
  - `cmi_bucket`

### Dataset lineage
- Designed to mirror your planned pipeline:
  - report text source role -> Open-i style report evidence
  - Hinglish query style role -> MMCQSD-style phrasing
- Current stage is MVP simulation; later this can be swapped to real Open-i input.

## 2.3 Can this be switched to real reports?

Yes. `build_hmg_mini.py` supports:
- `--input-reports <path-to-openi-csv>`

If provided, it builds from real report rows instead of seed records.

---

## 3) RAG System Description (Current MVP)

This is a **lightweight script-based RAG prototype**, not full LLM fine-tuning.

## 3.1 Retriever

- Embedding method: **TF-IDF** vectorization (fast MVP)
- Vector index: **FAISS IndexFlatIP**
- Metadata features:
  - concept tags (`report_concepts`)
  - side tags (`report_side`: left/right/bilateral/unspecified)
  - cleaned text (`clean_report_text`)

### Retrieval reranking
After FAISS candidates are fetched, reranking combines:

- concept overlap score
- lexical token overlap score
- side match bonus (query side vs report side)

This improves top-1/top-k realism for small-data setup.

## 3.2 Generator behavior

Two deterministic response modes in `run_baselines.py`:

- `generate_zero_shot(query)`
  - no report evidence inserted
  - generic condition-level reasoning statement

- `generate_grounded(query, evidence_texts)`
  - inserts top retrieved report evidence sentence
  - explicitly marks explanation as report-grounded

Important: This MVP uses **rule/template generation**, not a full LLM decoder.  
Purpose: validate retrieval-grounding effect quickly before heavy model integration.

---

## 4) Full Execution Flow (Top-to-Bottom)

1. Build HMG-mini dataset
   - script: `src/prototype/build_hmg_mini.py`
   - output: `data/processed/hmg_mini.csv`

2. Build retrieval index
   - script: `src/prototype/build_index.py`
   - outputs:
     - `indices/openi_reports.index`
     - `indices/openi_reports_metadata.csv`
     - `indices/tfidf_vectorizer.pkl`

3. Run both baselines
   - script: `src/prototype/run_baselines.py`
   - output: `results/h1_outputs.csv`

4. Evaluate H1
   - script: `src/prototype/evaluate_h1.py`
   - outputs:
     - `results/h1_scored.csv`
     - `results/h1_mvp_summary.md`

---

## 5) Commands to Reproduce

From repo root:

```bash
python src/prototype/build_hmg_mini.py --target-size 150 --seed 42
python src/prototype/build_index.py
python src/prototype/run_baselines.py --top-k 5
python src/prototype/evaluate_h1.py
```

---

## 6) Metrics — Meaning and Interpretation

## Retrieval metrics

- **Retrieval top-1 hit rate**
  - Fraction of samples where top retrieved report ID equals expected report ID.
  - Higher means stronger first-rank retrieval precision.

- **Retrieval top-k hit rate**
  - Fraction of samples where expected report ID appears anywhere in top-k retrieved IDs.
  - Higher means evidence recall is good even if top-1 is imperfect.

## Generation quality metrics

- **Factual score (0 to 1)**
  - Based on concept-level support between output and evidence.
  - `1.0`: claims supported by evidence concepts
  - lower values indicate missing or unsupported support

- **Hallucination flag/rate**
  - Flags outputs with unsupported medical concepts or contradiction patterns.
  - Rate is average of flags across samples.
  - Lower is better.

## Statistical test for H1

- **Wilcoxon signed-rank** used for paired non-parametric comparison.
- Inputs: per-sample grounded factual score vs zero-shot factual score.
- Reported:
  - p-value
  - effect size (Cohen's d on paired differences)
  - 95% confidence interval of mean difference

Interpretation:
- If p-value < 0.05 and mean grounded factual > mean zero-shot factual,
  early support exists for H1.

---

## 7) Current Result Snapshot (Deterministic Run)

From `results/h1_mvp_summary.md`:

- Samples: **150**
- Retrieval top-1 hit rate: **0.627**
- Retrieval top-k hit rate: **0.893**
- Mean factual score:
  - zero-shot: **0.740**
  - grounded: **0.786**
- Hallucination rate:
  - zero-shot: **0.240**
  - grounded: **0.167**
- Wilcoxon p-value: **0.015976**
- Effect size (d): **0.223**
- 95% CI (grounded - zero-shot): **[0.013, 0.079]**

Conclusion (MVP stage): grounded approach shows statistically significant factual improvement.

---

## 8) Example Input and Output (Real Prototype Row)

## Example input
- Query:
  - `Chest ke side me fluid jama hai kya please batao? right side me Report me 'pleural effusion' mention hai kya?`

## Retrieved evidence (top-1)
- `Moderate right pleural effusion with compressive atelectasis.`

## Zero-shot output
- `Query: ... Symptoms ke basis par pleural effusion ka possibility lag sakta hai. clinical correlation advised, lekin bina report evidence final confirmation possible nahi hai.`

## Grounded output
- `Query: ... Available report evidence ke hisaab se: Moderate right pleural effusion with compressive atelectasis. Isliye explanation report-grounded hai.`

## Scored outcome for this sample
- `zero_factual = 1.0`
- `grounded_factual = 1.0`
- `zero_hallucination = 0`
- `grounded_hallucination = 0`

---

## 9) What this prototype is NOT (yet)

- Not full Open-i raw ingestion at scale
- Not LaBSE/BioMedCLIP full production embedding stack
- Not QLoRA/DPO fine-tuned LLM generation
- Not clinician-validated clinical safety benchmark

This is a **defensible MVP** proving working flow + early H1 evidence under constrained time.

---

## 10) Next Upgrade Path (Post-Prototype)

1. Replace seed reports with real Open-i subset pipeline
2. Add better Hinglish synthesis from MMCQSD examples
3. Move from template generation to LLM-based zero-shot and grounded decoding
4. Expand evaluation set and include stricter human review
5. Add H2 setup (CMI-stratified robustness analysis)

