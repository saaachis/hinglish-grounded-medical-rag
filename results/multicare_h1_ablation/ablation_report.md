# Phase 6 Ablation: Raw Evidence vs Structured Evidence

## Setup
- 401 evaluated pairs (common between raw and structured runs)
- Same MMCQSD Hinglish patient queries
- Same LLM generator (Llama-3.1-8B-Instant via Groq)
- **Raw run**: Unstructured MultiCaRe case narratives as evidence
- **Structured run**: LLM-extracted structured clinical findings (Phase 6)

## Results Comparison

| Metric | Raw Evidence | Structured Evidence | Delta |
|---|---:|---:|---:|
| Grounded factual support | 0.5707 | 0.6394 | +0.0687 |
| Grounded hallucination | 0.2399 | 0.1961 | -0.0439 |
| Zero-shot factual support | 0.3408 | 0.3451 | +0.0042 |
| Zero-shot hallucination | 0.4262 | 0.4286 | +0.0024 |
| Factual gain (grounded - zero) | 0.2298 | 0.2943 | +0.0645 |
| Halluc reduction (zero - grounded) | 0.1863 | 0.2326 | +0.0463 |
| Effect size (Cohen's d) | 0.5551 | 0.6769 | +0.1218 |
| Wilcoxon p-value | 8.38e-22 | 4.06e-28 | - |

## Interpretation

- Structured evidence **improved** the factual gain by +0.0645 over raw evidence.
- Structured evidence **reduced** grounded hallucination by 0.0439.
- Both runs show statistically significant grounding benefit (raw p=8.38e-22, structured p=4.06e-28).

## Key Takeaway

The grounding approach (RAG) provides consistent factual support improvement regardless of whether
evidence is raw narrative or LLM-extracted structured format. This validates that the core RAG pipeline
is robust and not dependent on a specific evidence preprocessing step.

---
*Phase 6 ablation study on 401 pairs, Llama-3.1-8B-Instant via Groq*
