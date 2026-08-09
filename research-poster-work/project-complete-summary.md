# Complete Project Summary: Grounded Multimodal RAG for Hinglish Clinical Decision Support

> This document is a comprehensive reference for creating research poster content. It covers the full project arc — from proposal through development, limitation resolution, evaluation, and the final real-time RAG system.

---

## 1. Project Identity

- **Title**: Grounded Multimodal RAG for Code-Switching (Hinglish) in Clinical Decision Support Systems
- **Course**: Research Discourse - I
- **Team**: Devika Jonjale (B045), Saachi Shinde (B048), Manjiri Apshinge (B061)

---

## 2. Problem Statement

Modern clinical AI systems assume medical reasoning begins with clean, formal language. In India, patient-doctor interactions occur in **Hinglish** (Hindi-English code-switching), which:

- Lacks standardized grammar
- Varies by region, education, and context
- Encodes medical meaning indirectly (e.g. "pani bhar gaya" for pleural effusion)

Existing AI models treat Hinglish as noisy or broken English, leading to incorrect retrieval, loss of semantic intent, or **clinically unsafe hallucinated explanations**. This is not a translation problem — it is a **reasoning alignment problem**.

---

## 3. Research Hypotheses

| ID | Hypothesis | Statistical Test |
|---|---|---|
| **H1** | Evidence-grounded RAG significantly outperforms zero-shot LLMs in factual consistency and hallucination reduction | Wilcoxon signed-rank (paired, non-parametric) |
| **H2** | Increasing code-mixing level does not significantly degrade grounded model performance | Kruskal-Wallis H-test + Mann-Whitney U pairwise + Spearman correlation |

*(H3 — authoritative vs general evidence quality — deferred to future work due to scope.)*

---

## 4. Architecture Evolution

### 4.1 Original Proposal Architecture

```
Hinglish Query → LaBSE Encode → FAISS Retrieve → LLaVA-v1.5 (QLoRA + DPO) → Hinglish Explanation
                                                    ↑
                                           Open-i X-ray Reports
                                           + BioMedCLIP (images)
```

**Proposed techniques**: LaBSE encoding, FAISS + adaptive truncation, QLoRA fine-tuning, DPO hallucination control, BioMedCLIP multimodal, MMFCM evaluation metric.

### 4.2 Implemented Architecture (Final)

```
Hinglish Query → LaBSE Encode → FAISS Search (10K vectors) → Adaptive Truncation → Groq LLM (Llama-3.1-8B) → Hinglish Response
                                        ↑
                               MultiCaRe Clinical Cases
                              (61,316 filtered, 10K indexed)
```

**What changed and why**:

| Proposed | Implemented | Reason |
|---|---|---|
| Open-i (6,687 chest X-ray reports) | MultiCaRe (61,316 multi-specialty cases) | Domain mismatch — Open-i only covers chest, MMCQSD spans 18 specialties |
| LLaVA-v1.5 + QLoRA + DPO (local) | Llama-3.1-8B-Instant via Groq API (cloud) | No dedicated GPU (Intel UHD 730 only, 7.8 GB RAM) |
| BioMedCLIP image encoding | Text-only pipeline | Pivoted to multi-specialty text evidence; image encoding deferred |
| MMFCM metric | Concept-based factual support + hallucination scoring | Pragmatic evaluation suited to available data |
| HMG synthetic triplet dataset | MMCQSD + MultiCaRe paired via LaBSE + FAISS | Direct cross-lingual matching replaced synthetic generation |

**What remained identical to proposal**:

- LaBSE cross-lingual encoding (Hinglish ↔ English)
- FAISS inner-product search for retrieval
- Adaptive context truncation (MMed-RAG style)
- Evidence-grounded generation with evidence injection
- Zero-shot vs grounded comparison (H1)
- Code-mixing intensity analysis (H2)
- Streamlit interactive demo

---

## 5. The Critical Limitation and Its Resolution

### 5.1 The Problem

The initial prototype paired Open-i (chest X-ray reports) with MMCQSD (multi-specialty Hinglish queries). This produced only **11 usable pairs** after TF-IDF matching — a **domain mismatch**, not a code problem. 95.8% of MMCQSD queries (skin rash, neck swelling, mouth ulcers, etc.) had zero topical overlap with chest radiology.

### 5.2 The Solution

Replaced Open-i with **MultiCaRe** (75,000+ open-access clinical case reports from PubMed Central, 1990-2023), spanning all 18 MMCQSD condition categories.

### 5.3 Improvement Quantified

| Metric | Before (Open-i) | After (MultiCaRe) |
|---|---|---|
| Usable pairs | 11 | **3,015** (100% coverage) |
| Improvement | — | **274x increase** |
| Conditions covered | 1 (chest) | **18** (all MMCQSD categories) |
| Matching method | TF-IDF | **LaBSE + FAISS** (cross-lingual semantic) |
| High-quality matches (sim > 0.50) | ~0 | **1,547 (51.3%)** |
| Medium-quality matches (0.40-0.50) | — | 1,443 (47.9%) |
| Mean similarity score | — | 0.500 |

---

## 6. Data Pipeline

### 6.1 Datasets Used

| Dataset | Size | Role |
|---|---|---|
| **MMCQSD** | 3,015 samples | Hinglish patient queries (query source) |
| **MultiCaRe** | 61,316 filtered cases | English clinical evidence (evidence corpus) |

### 6.2 Processing Steps

1. **MultiCaRe filtering**: 93K+ raw cases → 61,316 cases filtered to MMCQSD-relevant specialties (dermatology, ophthalmology, ENT, oral pathology, orthopedics, oncology, etc.)
2. **LaBSE encoding**: Cross-lingual embeddings (dim=768) for both Hinglish queries and English evidence, with `max_seq_length=128` and `batch_size=32` for memory efficiency
3. **FAISS matching**: Inner-product search with condition-group filtering to prevent medically nonsensical pairings (e.g., skin rash query only matches dermatology evidence)
4. **Evidence sampling**: ~8,570 cases (3x queries per condition) for encoding feasibility on constrained hardware
5. **Text truncation**: 200 words for encoding, 400 words for LLM generation

### 6.3 The 18 Condition Groups

dry_scalp, edema, eye_inflammation, eye_redness, foot_swelling, hand_lump, itchy_eyelid, knee_swelling, lip_swelling, mouth_ulcers, neck_swelling, skin_dryness, skin_growth, skin_irritation, skin_rash, swollen_eye, swollen_tonsils, cyanosis

---

## 7. Evaluation Methodology

### 7.1 Concept-Based Metrics

The evaluation extracts **medical concepts** from both LLM output and evidence using pattern matching across **24 clinical concept categories** (rash, infection, fever, pain, swelling, inflammation, lesion, ulcer, erythema, pruritus, mass, fracture, effusion, cyanosis, necrosis, malignancy, allergy, etc.), with **negation detection** for terms like "no", "without", "not", "nahi".

| Metric | Definition | Range |
|---|---|---|
| **Factual support score** | Proportion of medical concepts in output that are also in evidence | 0.0 – 1.0 (higher = better) |
| **Hallucination score** | Proportion of medical concepts in output that are NOT in evidence | 0.0 – 1.0 (lower = better) |
| **Factual gain** | Grounded factual support − Zero-shot factual support | Positive = grounding helps |
| **Hallucination reduction** | Zero-shot hallucination − Grounded hallucination | Positive = grounding helps |

### 7.2 Code-Mixing Index (CMI)

Ratio of Hindi-origin tokens (from a curated dictionary of ~100 common Hinglish words) to total Latin-alphabet tokens. Continuous scale (0.0 = pure English, 1.0 = fully romanized Hindi). Split into **percentile-based tertiles** for group comparison.

### 7.3 Statistical Tests Used

| Test | Purpose |
|---|---|
| Wilcoxon signed-rank | Paired non-parametric test for H1 (same queries, two modes) |
| Cohen's d | Effect size for practical significance |
| 95% confidence interval | Precision of the estimated gain |
| Kruskal-Wallis H-test | 3-group comparison for H2 (across CMI levels) |
| Mann-Whitney U | Pairwise CMI-level comparisons (Bonferroni corrected) |
| Spearman's rho | Continuous correlation: CMI vs performance |

---

## 8. Results

### 8.1 Scale of Evaluation

| Run | Pairs Attempted | Clean Pairs | Key |
|---|---|---|---|
| Phase 5b (initial LLM) | 299 | 73 | Saachi's key |
| Day 1 (CMI-stratified) | 399 | 374 | Udit's key 1 |
| Day 2 | 362 | 362 | Udit's key 2 |
| Day 3 | 356 | 356 | Udit's key 3 |
| **Combined total** | — | **1,165** | — |

All evaluated using Llama-3.1-8B-Instant via Groq API (some early pairs used Llama-3.3-70B).

### 8.2 H1 Results: Grounded RAG vs Zero-Shot (n = 1,165)

| Metric | Zero-Shot | Grounded | Delta |
|---|---:|---:|---:|
| **Factual support** | 0.319 | 0.554 | **+0.235 (+73.5%)** |
| **Hallucination score** | 0.500 | 0.280 | **−0.220 (−44.0%)** |

**Statistical significance**:

| Test | Metric | Statistic | p-value | Effect Size |
|---|---|---|---|---|
| Wilcoxon signed-rank | Factual support | 38,951.5 | **3.09 × 10⁻⁶⁴** | Cohen's d = 0.576 (medium) |
| Wilcoxon signed-rank | Hallucination reduction | 28,429.5 | **5.33 × 10⁻⁵¹** | Cohen's d = 0.492 (small-to-medium) |

- 95% CI for factual gain: **[0.211, 0.258]** (excludes zero)

**H1 Verdict: STRONGLY SUPPORTED.** Evidence grounding improves factual support by ~74% and reduces hallucination by ~44%, with p-values far below any conventional threshold and medium effect sizes.

### 8.3 H2 Results: Code-Mixing Intensity Effect (n = 1,165)

| CMI Level | N | Mean CMI | Grounded Factual | Factual Gain | Halluc Reduction |
|---|---:|---:|---:|---:|---:|
| Low (more English) | 385 | 0.351 | 0.554 | +0.202 | +0.206 |
| Medium | 384 | 0.428 | 0.544 | +0.241 | +0.208 |
| High (more Hindi) | 396 | 0.493 | 0.563 | +0.260 | +0.245 |

**Statistical tests**:

| Test | Metric | Statistic | p-value |
|---|---|---|---|
| Kruskal-Wallis | Factual gain across levels | H = 3.879 | 0.144 (n.s.) |
| Kruskal-Wallis | Hallucination reduction across levels | H = 2.528 | 0.283 (n.s.) |
| Spearman correlation | CMI vs Factual gain | rho = 0.070 | 0.016* (marginal) |
| Spearman correlation | CMI vs Halluc reduction | rho = 0.033 | 0.260 (n.s.) |

**Pairwise comparisons (Mann-Whitney U, Bonferroni corrected)**: No significant differences between any pair.

**H2 Verdict: NOT SUPPORTED statistically, but academically meaningful.** The grounding benefit is **robust across all code-mixing levels** — higher Hindi content does not degrade RAG performance. The slight trend of *better* performance at higher CMI is intriguing but not significant.

### 8.4 Phase 6 Ablation: Raw vs Structured Evidence (n = 401)

| Metric | Raw Evidence | Structured Evidence | Delta |
|---|---:|---:|---:|
| Grounded factual support | 0.571 | 0.639 | **+0.069** |
| Grounded hallucination | 0.240 | 0.196 | **−0.044** |
| Factual gain | 0.230 | 0.294 | **+0.065** |
| Cohen's d | 0.555 | 0.677 | **+0.122** |
| Wilcoxon p-value | 8.38 × 10⁻²² | 4.06 × 10⁻²⁸ | Both significant |

**Takeaway**: LLM-extracted structured evidence improves grounding quality (+6.9% factual support, −4.4% hallucination), but the core RAG pipeline works well even with raw narrative evidence.

---

## 9. Real-Time RAG Pipeline (Final Implementation)

### 9.1 Components

| Component | Implementation | File |
|---|---|---|
| **Text Encoder** | LaBSE (`sentence-transformers/LaBSE`), dim=768 | `src/encoding/text_encoder.py` |
| **Vector Index** | FAISS IndexFlatIP, 10,000 vectors | `src/retrieval/indexer.py` |
| **Retriever** | Adaptive context truncation (score-drop based) | `src/retrieval/retriever.py` |
| **Generator** | Llama-3.1-8B-Instant via Groq API | `src/generation/generator.py` |
| **Pipeline** | End-to-end orchestrator with scoring | `src/pipeline.py` |
| **Index Builder** | One-time script, ~17 min on CPU | `build_index.py` |
| **Demo UI** | Streamlit app, side-by-side comparison | `app.py` |

### 9.2 Pipeline Flow

```
User types Hinglish query
    → LaBSE encodes to 768-dim vector
    → FAISS searches 10,000 pre-indexed MultiCaRe cases
    → Adaptive truncation filters by score drop
    → Top-k evidence texts injected into grounded prompt
    → Groq API (Llama-3.1-8B) generates Hinglish response
    → Concept-based factual support & hallucination scores computed
    → Side-by-side display: Grounded vs Zero-Shot
```

### 9.3 Streamlit Demo Features

- Example query buttons (6 pre-loaded Hinglish medical queries)
- Custom query text input
- Sidebar: architecture info, top-k slider, zero-shot toggle
- Retrieved evidence display (case_id, condition, similarity score, excerpt)
- Two-column response comparison (grounded vs zero-shot)
- Real-time factual support and hallucination metrics

---

## 10. Hardware Constraints and Adaptations

| Constraint | Impact | Adaptation |
|---|---|---|
| No dedicated GPU (Intel UHD 730) | Cannot run local LLM inference or fine-tuning | Used Groq API for cloud inference |
| 7.8 GB RAM | OOM during encoding/matching | batch_size=32, max_seq_length=128, evidence sampling, text truncation |
| Groq free tier (500K tokens/day) | Cannot evaluate all 3,015 pairs in one run | Multi-day evaluation with key rotation, resume capability |
| No CUDA | QLoRA/DPO fine-tuning impossible | Accepted pre-trained LLM with prompt engineering |

---

## 11. Key Contributions

1. **Cross-lingual RAG for Hinglish**: First evidence-grounded RAG system specifically designed for Hindi-English code-switched clinical queries
2. **LaBSE + FAISS pipeline**: Demonstrated effective cross-lingual retrieval between Hinglish queries and English clinical evidence (mean similarity 0.50 across 3,015 pairs)
3. **Statistically validated grounding benefit**: H1 supported with n=1,165, p=3.09×10⁻⁶⁴, Cohen's d=0.576 — grounding improves factual support by 74% and reduces hallucination by 44%
4. **Code-mixing robustness**: H2 shows RAG performance is stable across all code-mixing levels — the system works equally well for lightly and heavily code-mixed Hinglish
5. **Evidence format ablation**: Structured evidence provides incremental improvement but raw evidence is sufficient for effective grounding
6. **Working demo**: Real-time Streamlit-based RAG pipeline for live demonstration

---

## 12. Limitations and Future Work

### Current Limitations

| Limitation | Description |
|---|---|
| No fine-tuning | LLM used as-is via API; no domain-specific QLoRA adaptation |
| No DPO | No preference-based hallucination control training |
| Text-only | No multimodal (image/X-ray) evidence integration |
| Concept-based evaluation | Not a clinician-validated metric; pattern-matching based |
| Cloud-dependent generation | Requires Groq API key and internet connection |
| H3 not tested | Authoritative vs general evidence comparison deferred |

### Future Work (Semester 3 Scope)

| Task | Description |
|---|---|
| QLoRA fine-tuning | Adapt LLM to medical Hinglish domain |
| DPO training | Preference optimization to reduce hallucination |
| BioMedCLIP integration | Add medical image retrieval (X-rays, dermatology photos) |
| H3 hypothesis testing | Compare authoritative vs general biomedical evidence |
| Clinician evaluation | Expert validation of generated explanations |
| Full-scale evaluation | All 3,015 pairs through LLM |
| MMFCM metric | Implement formal multimodal fact-capturing metric |

---

## 13. Visualizations Available (for poster)

All plots are in `research-poster-work/` as high-resolution PNGs:

| File | Content |
|---|---|
| `01_h1_bar_comparison.png` | H1 bar chart: zero-shot vs grounded (factual & hallucination means) |
| `02_h1_distributions.png` | Violin plots of factual & hallucination score distributions |
| `03_factual_gain_distribution.png` | Histogram of per-pair factual gain |
| `04_h2_cmi_levels.png` | H2 grouped bars by CMI tertile, with Kruskal-Wallis p annotation |
| `05_per_condition_gain.png` | Horizontal bars: mean factual gain by condition |
| `06_cmi_scatter.png` | CMI vs performance scatter with trend lines and Spearman rho |
| `07_limitation_improvement.png` | Before/after: 11 → 3,015 pairs; condition coverage pie charts |
| `08_pipeline_architecture.png` | System architecture diagram |
| `09_match_quality.png` | Similarity score histogram + match quality distribution pie |
| `10_significance_summary.png` | Bar chart of −log₁₀(p) for all statistical tests |
| `11_ablation_comparison.png` | Raw vs structured evidence comparison bars |

---

## 14. Key Numbers for Quick Reference

| Metric | Value |
|---|---|
| Total MMCQSD queries | 3,015 |
| MultiCaRe cases (filtered) | 61,316 |
| Matched pairs | 3,015 (100% coverage) |
| FAISS indexed vectors (demo) | 10,000 |
| LLM-evaluated pairs (combined) | 1,165 |
| Ablation pairs | 401 |
| Factual support improvement (H1) | +73.5% (0.319 → 0.554) |
| Hallucination reduction (H1) | −44.0% (0.500 → 0.280) |
| H1 p-value (factual) | 3.09 × 10⁻⁶⁴ |
| H1 effect size (Cohen's d) | 0.576 (medium) |
| H2 Kruskal-Wallis p (factual gain) | 0.144 (n.s. = robust) |
| Condition groups | 18 |
| CMI tertile levels | 3 (low / medium / high) |
| LaBSE embedding dimension | 768 |
| LLM model | Llama-3.1-8B-Instant |
| Pipeline latency (per query) | ~10-15 seconds (encoding + retrieval + generation) |

---

## 15. Sample RAG Outputs (for poster examples)

### Example 1: Skin Rash (Grounding helps dramatically)

**Query**: "Mujhe ek samasya hai mere skin me jo mere legs par develop hui aur ab mere arms aur wrists par hai, woh bahut khujali kar sakti hai, woh lal dots jaise hain"

**Zero-shot** (no evidence): "Bhai, mujhe lagta hai ki aapko ek skin problem hai, jiska naam 'urticaria' ho sakta hai..."
- Factual: 0.00 | Hallucination: 1.00

**Grounded** (with MultiCaRe evidence): "Didi, aapki samasya ka sabse pehle kya hai, yehi dekhna hai. Aapki blood report mein neutrophil count 9.9 hai, jo thoda high hai... Lekin aapki skin par lal dots aur khujali hai..."
- Factual: 1.00 | Hallucination: 0.00

### Example 2: Eye Redness

**Query**: "Doctor mujhe eye ke aas paas bahut dard ho raha hai jab mujhe toy gun ki goli se unexpected hit laga. Cornea mein thodi si redness hai..."

**Zero-shot**: Generic advice about eye pain.
- Factual: 0.25 | Hallucination: 0.00

**Grounded**: References clinical evidence about traumatic iritis, mentions photophobia and dull ache from the case report.
- Factual: 1.00 | Hallucination: 0.00

### Example 3: Lip Swelling

**Query**: "Mere gaal sujan aati hai (ya toh right ho jata hai ya left, kabhi dono nahi, aur sirf cheek bone par)"

**Zero-shot**: Mentions "Trigeminal Neuralgia" without evidence — hallucinated diagnosis.
- Factual: 0.00 | Hallucination: 1.00

**Grounded**: References CBCT findings from the matched case report, stays factual.
- Factual: 0.25 | Hallucination: 0.00

---

## 16. Project Timeline (Actual)

| Phase | What | Duration |
|---|---|---|
| Phase 1 | Project proposal, literature review, dataset selection | Weeks 1-3 |
| Phase 2 | Initial prototype with Open-i + MMCQSD + TF-IDF matching | Weeks 4-6 |
| Phase 3 | Identified limitation (11 pairs), dataset comparison study | Weeks 7-8 |
| Phase 4 | MultiCaRe integration, LaBSE + FAISS matching, 3,015 pairs | Weeks 9-10 |
| Phase 5 | Template-based prototype → LLM prototype → Multi-day evaluation | Weeks 11-14 |
| Phase 6 | Structured evidence extraction + ablation study | Week 15 |
| Phase 7 | Real-time RAG pipeline + Streamlit demo | Week 16 |
| Phase 8 | Research poster preparation | Current |

---

## 17. Repository Structure (Key Files)

```
hinglish-grounded-medical-rag/
├── src/
│   ├── encoding/text_encoder.py       # LaBSE model
│   ├── retrieval/indexer.py           # FAISS indexer
│   ├── retrieval/retriever.py         # Adaptive context retriever
│   ├── generation/generator.py        # Groq-based Hinglish generator
│   ├── pipeline.py                    # RAG pipeline orchestrator
│   ├── matching/pair_builder.py       # LaBSE+FAISS matching pipeline
│   └── prototype/
│       ├── run_llm_prototype.py       # LLM evaluation script
│       ├── run_h1h2_analysis.py       # Statistical analysis
│       └── run_phase6_ablation.py     # Evidence format ablation
├── build_index.py                     # One-time FAISS index builder
├── app.py                             # Streamlit demo
├── data/
│   ├── processed/multicare_filtered.csv
│   └── faiss_index/                   # Pre-built index + metadata
├── research-poster-work/
│   ├── generate_plots.py              # All visualization generation
│   └── *.png                          # 11 publication-ready plots
└── results/                           # Evaluation outputs (gitignored)
    ├── combined_h1h2/combined_scored.csv
    └── multicare_h1_ablation/ablation_report.md
```
