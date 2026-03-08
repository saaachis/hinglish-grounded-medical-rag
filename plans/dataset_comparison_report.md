# Multi-Dataset Synthetic Comparison Report

## Purpose
Compare potential datasets (separately) under the same prototype pipeline and techniques to estimate which dataset family is most useful for the current Hinglish-grounded RAG objective.

## Important Validity Note
- This is a **synthetic proxy comparison** (not full real dataset benchmarking).
- It is useful for early directional decisions, not final research claims.

## Experiment Setup
- Samples per dataset profile: **800**
- Same retrieval/indexing and generation pipeline across all profiles
- Added cross-profile distractor reports in retrieval index to better mimic real-world confusion
- Added noisy/code-switched query variants to better mimic real input conditions
- Same evaluator and paired statistical test

## Dataset Profiles Compared
- Open-i Proxy (radiology-style, closest to target objective)
- MMCQSD Proxy (code-mixed clinical language style)
- PubMedQA Proxy (abstract/evidence QA style)
- MMed-Bench Proxy (mixed multimodal QA style)

## Results Table

### Context: what each metric means
- **Top-1 Hit / Top-k Hit**: retrieval quality (higher is better).
- **Factual Gain**: `(Grounded Factual - Zero Factual)` (higher is better).
- **Hallucination Drop**: `(Zero Hall. - Grounded Hall.)` (higher is better).
- **p-value**: significance of grounded vs zero-shot difference (`< 0.05` means meaningful difference).

### Quick legend
- ✅ strong signal
- ⚠️ mixed signal
- ❌ weak / no signal

| Profile | Retrieval Signal | Factual Gain Signal | Hallucination Signal | p-value Signal | Decision Context |
|---|---|---|---|---|---|
| **MMed-Bench Proxy (Mixed multimodal QA)** | Top-1: 0.281, Top-k: 0.666 ⚠️ | Gain: **+0.413 ✅** | Drop: **0.000 ❌** | 0.0000 ✅ | Strong factual gain in synthetic setup, but no hallucination improvement and lower objective fit for Hinglish-radiology task. |
| **PubMedQA Proxy (Abstract QA style)** | Top-1: 0.166 ❌, Top-k: 0.834 ⚠️ | Gain: **+0.250 ✅** | Drop: **0.000 ❌** | 0.0000 ✅ | Useful for abstract reasoning behavior, but weak retrieval precision and low domain/language alignment for final objective. |
| **MMCQSD Proxy (Code-mixed clinical)** | Top-1: 0.334 ⚠️, Top-k: 1.000 ✅ | Gain: **+0.167 ⚠️** | Drop: **+0.166 ✅** | 0.0000 ✅ | Best Hinglish-style behavior signal; important for query realism and code-mixing robustness. |
| **Open-i Proxy (Radiology aligned)** | Top-1: 0.330 ⚠️, Top-k: 0.711 ⚠️ | Gain: **+0.162 ⚠️** | Drop: **+0.369 ✅** | 0.0000 ✅ | Best radiology-grounding relevance to project objective; now shows clear hallucination reduction and positive factual gain. |

### Bottom-line interpretation
- If you rank by **synthetic factual gain only**, MMed-Bench appears highest.
- If you rank by **project objective fit** (Hinglish + radiology grounding), **Open-i + MMCQSD** remains the correct foundation.
- This is why final selection still prioritizes:
  1) Open-i real subset  
  2) MMCQSD-guided Hinglish generation  
  3) HMG construction on top of both.

## Ranking Summary (Most Useful First)
- `MMed-Bench Proxy (Mixed multimodal QA)`: factual gain=0.413, hallucination drop=0.000, top-k hit=0.666, p=0.0000
- `PubMedQA Proxy (Abstract QA style)`: factual gain=0.250, hallucination drop=0.000, top-k hit=0.834, p=0.0000
- `MMCQSD Proxy (Code-mixed clinical)`: factual gain=0.167, hallucination drop=0.166, top-k hit=1.000, p=0.0000
- `Open-i Proxy (Radiology aligned)`: factual gain=0.162, hallucination drop=0.369, top-k hit=0.711, p=0.0000

## Recommendation
- Best profile **under this synthetic proxy setup**: **MMed-Bench Proxy (Mixed multimodal QA)**
- Do **not** treat synthetic ranking as final real-dataset selection.
- For your actual project objective (Hinglish grounded radiology support), still prioritize:
  1) Open-i real subset
  2) MMCQSD-guided Hinglish generation
  3) HMG construction on top of (1)+(2)

## Next Step to Increase Validity
- Replace synthetic proxy inputs with real subset CSVs and re-run the same scripts.
