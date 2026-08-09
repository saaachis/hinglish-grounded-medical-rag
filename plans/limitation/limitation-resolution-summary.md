# Limitation Resolution: From 11 Pairs to 3,015 — Full Development Summary

## 1. The Original Limitation

The initial prototype paired two incompatible datasets:

| Dataset | Content | Medical Scope |
|---|---|---|
| **Open-i** | 6,687 English chest X-ray radiology reports | Chest only (lungs, heart, pleural space) |
| **MMCQSD** | 3,015 Hinglish patient queries + English summaries | Multi-specialty (skin, eye, ENT, oral, musculoskeletal, etc.) |

**Result:** Only **11 usable pairs** after aggressive TF-IDF matching. This was a fundamental **domain mismatch** — 95.8% of MMCQSD queries (skin rash, neck swelling, mouth ulcers, etc.) had zero topical overlap with chest radiology.

This blocked all downstream work:
- Statistical hypothesis testing (n=11 is too fragile)
- Training or fine-tuning any model
- Producing publishable research results
- Building a meaningful demo

---

## 2. Solution: MultiCaRe as Primary Evidence Corpus

### Why MultiCaRe

**MultiCaRe** (Multi-Modal Clinical Dataset) contains 75,000+ open-access clinical case reports from PubMed Central (1990-2023), spanning oncology, cardiology, dermatology, ophthalmology, surgery, ENT, oral pathology, orthopedics, and more. Unlike Open-i's narrow chest focus, MultiCaRe's specialties naturally align with all 18 MMCQSD condition categories.

### Updated Project Scope

From: *"Evidence-grounded RAG for Hinglish clinical queries, grounded in chest radiology reports"*

To: *"Evidence-grounded RAG for Hinglish clinical queries, grounded in multi-specialty clinical evidence"*

The core hypothesis (grounded > zero-shot) and architecture (LaBSE encoding, FAISS retrieval, grounded generation) remain unchanged. Only the evidence corpus broadened.

---

## 3. Data Pipeline: Filtering, Encoding, and Matching

### Step 1 — MultiCaRe Filtering

Used the `multiversity` Python library to filter MultiCaRe down to conditions relevant to MMCQSD's 18 categories.

| Metric | Value |
|---|---|
| Raw MultiCaRe cases | 93,000+ |
| Filtered to relevant specialties | **61,316 cases** |
| Minimum case text length | 50 words |

### Step 2 — LaBSE Cross-Lingual Encoding

Replaced the initial TF-IDF approach with **LaBSE** (Language-Agnostic BERT Sentence Embedding) for cross-lingual semantic matching between English evidence and Hinglish queries.

| Design Decision | Rationale |
|---|---|
| Model: `sentence-transformers/LaBSE` | Supports 109 languages including Hindi and English; designed for cross-lingual similarity |
| L2-normalized embeddings | Enables cosine similarity via FAISS inner product |
| `max_seq_length=128` | Memory optimization for low-RAM system (7.8 GB total) |
| `batch_size=32` | Prevents OOM crashes during encoding |
| Evidence sampling (~8,570 cases) | ~3x MMCQSD queries per condition to keep encoding feasible on constrained hardware |
| Text truncation (200 words) | Keeps embedding input manageable while retaining key clinical content |

### Step 3 — FAISS Matching with Condition Filtering

Built a FAISS index on the encoded evidence and matched each MMCQSD query to the most semantically similar MultiCaRe case, with intelligent condition-based filtering to prevent medically nonsensical pairings.

**Condition compatibility groups** ensured that, for example, a "skin rash" query would only match against dermatology/dermatitis-relevant evidence, not cardiology or ENT cases.

### Matching Results

| Metric | Value |
|---|---|
| Total matched pairs | **3,015** (100% of MMCQSD covered) |
| High-quality matches (sim > 0.50) | 1,547 (51.3%) |
| Medium-quality matches (0.40-0.50) | 1,443 (47.9%) |
| Low-quality matches (< 0.40) | 25 (0.8%) |
| Mean similarity score | 0.500 |
| Max similarity score | 0.710 |
| Conditions covered | All 18 |

**Improvement:** From **11 weakly-supervised pairs** (Open-i + TF-IDF) to **3,015 semantically-matched pairs** (MultiCaRe + LaBSE) — a **274x increase**.

---

## 4. Prototype Development Phases

### Phase 5a — Template-Based Prototype (All 3,015 Pairs)

Ran all 3,015 pairs through a template-based generator (no LLM) as an initial validation:
- Grounded response = first 500 words of evidence text prefixed with "Based on the clinical report:"
- Zero-shot response = generic template without evidence

| Metric | Zero-Shot | Grounded | Delta |
|---|---|---|---|
| Factual support | 0.352 | 0.995 | +0.642 |
| Hallucination score | 0.284 | 0.000 | +0.284 |

These results were **artificially perfect** because the template directly copied evidence text. While confirming the pipeline was functional, it demonstrated nothing about real LLM behavior. This motivated the move to real LLM generation.

### Phase 5b — Real LLM Generation (Groq API, Initial Run)

Used **Llama-3.1-8B-Instant** via Groq API to generate actual Hinglish medical responses in two modes:
- **Zero-shot:** Query only, no evidence provided
- **Grounded:** Query + retrieved MultiCaRe evidence

The initial run attempted 299 pairs (proportionally sampled across conditions). Due to daily token limit exhaustion (500K TPD on Groq free tier), only **73 pairs** received both valid zero-shot and grounded outputs.

### Day 1 Run — CMI-Stratified LLM Generation (399 New Pairs)

To enable H2 testing alongside H1, designed a **CMI-stratified sampling strategy**:

1. Computed Code-Mixing Index (CMI) for all remaining unevaluated pairs
2. Split into tertiles using percentile thresholds
3. Sampled 133 pairs from each tertile (low, medium, high code-mixing)
4. Ran all 399 through Groq API with a fresh key

| Metric | Value |
|---|---|
| Total pairs attempted | 399 |
| Clean pairs (both outputs valid) | **374** |
| API errors (daily limit exhaustion) | 25 |
| Runtime | ~90 minutes |
| Tokens consumed | ~500K (full daily budget) |

---

## 5. Combined Results (447 Clean Pairs)

Merged the 73 clean pairs from Phase 5b with the 374 clean pairs from Day 1 = **447 total clean pairs**.

### H1: Grounded RAG vs Zero-Shot Generation

| Metric | Zero-Shot | Grounded | Delta |
|---|---|---|---|
| **Factual support** | 0.314 | 0.549 | **+0.235 (+74.8%)** |
| **Hallucination score** | 0.518 | 0.293 | **-0.225 (-43.4%)** |

**Statistical Tests:**

| Test | Metric | Statistic | p-value | Significance |
|---|---|---|---|---|
| Wilcoxon signed-rank | Factual support | 6356.5 | **1.21e-24** | Highly significant |
| Wilcoxon signed-rank | Hallucination reduction | 4934.0 | **1.14e-19** | Highly significant |

| Effect Measure | Value | Interpretation |
|---|---|---|
| Cohen's d (factual) | **0.562** | Medium effect |
| Cohen's d (hallucination) | **0.482** | Small-to-medium effect |
| 95% CI (factual gain) | [0.196, 0.274] | Excludes zero |

**H1 Verdict: STRONGLY SUPPORTED.** Grounding with retrieved clinical evidence significantly improves factual support (+75%) and reduces hallucination (-43%) compared to zero-shot generation.

### H2: Effect of Code-Mixing Intensity on RAG Performance

Queries were split into three CMI levels using percentile-based tertiles:

| CMI Level | N | Mean CMI | Grounded Factual | Factual Gain | Halluc Reduction |
|---|---|---|---|---|---|
| Low (more English) | 148 | 0.345 | 0.570 | +0.206 | +0.211 |
| Medium | 147 | 0.428 | 0.493 | +0.212 | +0.207 |
| High (more Hindi) | 152 | 0.492 | 0.583 | +0.286 | +0.256 |

**Statistical Tests:**

| Test | Metric | Statistic | p-value | Result |
|---|---|---|---|---|
| Kruskal-Wallis | Factual gain across levels | H=2.875 | 0.238 | Not significant |
| Kruskal-Wallis | Halluc reduction across levels | H=1.310 | 0.519 | Not significant |
| Spearman correlation | CMI vs factual gain | rho=0.080 | 0.091 | Trending (not significant) |
| Spearman correlation | CMI vs halluc reduction | rho=0.017 | 0.725 | Not significant |

**Pairwise Comparisons (Mann-Whitney U, Bonferroni corrected):** No significant differences between any pair of CMI levels for either metric.

**H2 Verdict: NOT SUPPORTED statistically, but the result is academically meaningful.** The grounding benefit is **robust across all code-mixing levels** — higher Hindi content does not degrade RAG performance. The slight trend showing *better* performance at higher CMI (factual gain of +0.286 vs +0.206) may reach significance with more data.

---

## 6. Evaluation Methodology

### Concept-Based Metrics

Rather than surface-level token overlap, the evaluation extracts **medical concepts** from both output and evidence using pattern matching across 24 clinical concept categories (rash, infection, fever, pain, swelling, etc.), with negation detection for terms like "no," "without," and "nahi."

| Metric | Definition |
|---|---|
| **Factual support score** | Proportion of medical concepts in the output that are also present in the evidence (0.0 to 1.0) |
| **Hallucination score** | Proportion of medical concepts in the output that are NOT present in the evidence (0.0 to 1.0) |

### CMI (Code-Mixing Index)

Ratio of Hindi-origin tokens (from a curated dictionary of ~100 common Hinglish words) to total Latin-alphabet tokens in each query. This captures the degree of Hindi-English code-mixing on a continuous scale (0.0 = pure English, 1.0 = fully romanized Hindi).

For statistical testing, the continuous CMI was split into tertiles (percentile-based) to create three balanced groups.

### Statistical Tests

| Test | Purpose |
|---|---|
| Wilcoxon signed-rank | Paired non-parametric test for H1 (grounded vs zero-shot on same queries) |
| Kruskal-Wallis H-test | 3-group non-parametric test for H2 (across CMI levels) |
| Mann-Whitney U | Pairwise comparisons between CMI levels (with Bonferroni correction) |
| Spearman's rho | Continuous correlation between CMI and performance metrics |
| Cohen's d | Effect size for practical significance |

---

## 7. Infrastructure and Technical Decisions

### Hardware Constraints

The development was done on a system with:
- **CPU:** Intel (integrated GPU only — Intel UHD Graphics 730)
- **RAM:** 7.8 GB total (frequently at 94% utilization)
- **No dedicated GPU**

This drove several architectural decisions:
- LaBSE encoding with `max_seq_length=128` and `batch_size=32`
- Evidence corpus sampling (8,570 cases instead of full 61K)
- Text truncation to 200 words for encoding, 400 words for generation
- Cloud API inference (Groq) instead of local Ollama

### LLM Inference: Groq API

| Parameter | Value |
|---|---|
| Model | `llama-3.1-8b-instant` |
| Temperature | 0.3 |
| Max output tokens | 300 |
| Rate limit | 500K tokens/day (free tier) |
| Retry strategy | Built-in Groq SDK exponential backoff |
| Delay between calls | 2 seconds |

Two system prompts controlled generation mode:
- **Grounded:** Instructed to respond strictly based on provided clinical evidence, in Hinglish
- **Zero-shot:** Instructed to respond based on general medical knowledge only, in Hinglish

### LLM Evidence Extraction (Phase 6 — Partial)

Extracted structured evidence from 242 unique MultiCaRe cases using the same LLM. Each raw case narrative was transformed into a structured format:

```
Primary Finding: [diagnosis]
Location: [body area]
Symptoms: [patient-reported symptoms]
Clinical Signs: [examination findings]
Severity: [mild/moderate/severe]
Duration: [timeframe]
Key Evidence: [most important clinical sentence, quoted]
```

This extraction was completed for the 242 cases used in Phase 5b. The ablation study comparing raw vs structured evidence (Phase 6 Step 2) was paused at 10 pairs due to token limit exhaustion and is pending resumption.

---

## 8. Files and Artifacts

### Data Files

| File | Description |
|---|---|
| `data/processed/multicare_filtered.csv` | 61,316 MultiCaRe cases filtered to MMCQSD-relevant specialties |
| `data/processed/mmcqsd_multicare_paired.csv` | 3,015 matched pairs (MMCQSD query + MultiCaRe evidence) |
| `data/processed/cmi_sample_day1.csv` | 399 CMI-stratified pairs selected for Day 1 evaluation |
| `data/processed/extracted_evidence.csv` | Structured evidence for 242 unique MultiCaRe cases |

### Result Files

| File | Description |
|---|---|
| `results/multicare_h1/h1_multicare_scored.csv` | Template-based prototype results (3,015 pairs) |
| `results/multicare_h1_llm/h1_llm_scored.csv` | Phase 5b LLM results (299 pairs, 73 clean) |
| `results/multicare_h1h2_day1/h1_llm_scored.csv` | Day 1 CMI-stratified LLM results (399 pairs, 374 clean) |
| `results/combined_h1h2/combined_scored.csv` | Merged clean results (447 pairs) |
| `results/combined_h1h2/h1_h2_combined_report.md` | Combined H1 + H2 statistical analysis report |
| `results/multicare_h1_ablation/ablation_scored_partial.csv` | Phase 6 ablation (10 pairs, partial) |

### Source Code

| File | Purpose |
|---|---|
| `src/encoding/text_encoder.py` | LaBSE model loading and text encoding |
| `src/matching/pair_builder.py` | FAISS matching pipeline with condition filtering |
| `run_matching.py` | Memory-efficient matching orchestrator |
| `src/prototype/run_multicare_prototype.py` | Template-based H1 prototype |
| `src/prototype/run_llm_prototype.py` | Real LLM generation prototype (Groq API) |
| `src/prototype/run_phase6_ablation.py` | Phase 6: extraction + ablation |
| `src/prototype/build_cmi_sample.py` | CMI-stratified sampling for H2 |
| `src/prototype/run_h1h2_analysis.py` | Combined H1 + H2 statistical analysis |

---

## 9. What Remains

### Immediate Next Steps (Days 2-4)

| Task | Token Cost | Purpose |
|---|---|---|
| Day 2: 400 more CMI-stratified pairs | ~480K | Push total to ~850 clean pairs, strengthen H2 |
| Day 3: 400 more pairs | ~480K | Push total to ~1,250 clean pairs |
| Day 4: Phase 6 ablation (289 pairs) | ~350K | Compare raw vs structured evidence |

### Semester 3 Scope

| Task | Description |
|---|---|
| H3 testing | Compare authoritative vs general evidence quality |
| BioMedCLIP integration | Image encoder for multimodal retrieval |
| QLoRA fine-tuning | Adapt generator to medical Hinglish domain |
| Full-scale evaluation | All 3,015 pairs through LLM |
| Streamlit demo | Interactive prototype interface |

---

## 10. Key Takeaways

1. **The limitation was a domain mismatch, not a code problem.** Replacing Open-i with MultiCaRe solved the fundamental issue of having no topically relevant evidence for 96% of queries.

2. **LaBSE + FAISS produces high-quality cross-lingual matches.** 99.2% of pairs achieved medium or high similarity scores, despite the query language (Hinglish) being different from the evidence language (English).

3. **H1 is strongly supported.** Evidence grounding improves factual support by 75% and reduces hallucination by 43%, with p-values below 1e-19 and medium effect sizes.

4. **H2 shows robustness, not degradation.** Code-mixing intensity does not significantly affect RAG performance. The system works equally well for lightly code-mixed and heavily code-mixed Hinglish queries.

5. **Hardware constraints shaped but did not block the work.** Every step was adapted to run on a system with no GPU and limited RAM, using memory-efficient encoding, cloud API inference, and incremental processing with resume capability.
