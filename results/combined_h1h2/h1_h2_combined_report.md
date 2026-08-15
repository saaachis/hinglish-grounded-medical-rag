# Combined H1 + H2 Results

**Total clean pairs evaluated: 1165**

---

## H1: Grounded RAG vs Zero-Shot Generation

### Factual Support

| Metric | Zero-Shot | Grounded | Delta |
|---|---:|---:|---:|
| Factual support | 0.3190 | 0.5535 | **+0.2345** |
| Hallucination score | 0.5001 | 0.2804 | **+0.2197** |

### Statistical Tests

- **Factual support**: wilcoxon_signed_rank, stat=38951.5000, p=3.09e-64 ***
  - Effect size (Cohen's d): Medium (0.576)
  - 95% CI: [0.2111, 0.2579]
- **Hallucination reduction**: wilcoxon_signed_rank, stat=28429.5000, p=5.33e-51 ***
  - Effect size (Cohen's d): Small (0.492)

### H1 Verdict

**SUPPORTED**: Grounded RAG significantly improves factual support over zero-shot (p=3.09e-64, d=0.576).
**SUPPORTED**: Grounded RAG significantly reduces hallucination (p=5.33e-51, d=0.492).

---

## H2: Effect of Code-Mixing Intensity on RAG Performance

### Per-Level Metrics

| CMI Level | N | Mean CMI | Zero Factual | Grounded Factual | Factual Gain | Halluc Reduction |
|---|---:|---:|---:|---:|---:|---:|
| low_cm | 385 | 0.351 | 0.3517 | 0.5535 | +0.2018 | +0.2057 |
| medium_cm | 384 | 0.428 | 0.3029 | 0.5437 | +0.2409 | +0.2077 |
| high_cm | 396 | 0.493 | 0.3028 | 0.5631 | +0.2602 | +0.2451 |

### Kruskal-Wallis Test (3-group comparison)

- **Factual gain across CMI levels**: H=3.8792, p=0.1438 n.s.
- **Halluc reduction across CMI levels**: H=2.5280, p=0.2825 n.s.

### Pairwise Comparisons (Mann-Whitney U, Bonferroni corrected)

| Comparison | Factual Gain p (Bonf.) | Halluc Reduction p (Bonf.) |
|---|---:|---:|
| low_cm vs medium_cm | 0.6080 n.s. | 1.0000 n.s. |
| low_cm vs high_cm | 0.1536 n.s. | 0.4771 n.s. |
| medium_cm vs high_cm | 1.0000 n.s. | 0.5497 n.s. |

### Spearman Correlation (continuous CMI vs performance)

- CMI vs Factual Gain: rho=0.0704, p=0.0163 *
- CMI vs Halluc Reduction: rho=0.0330, p=0.2598 n.s.

### H2 Verdict

**NOT SUPPORTED**: No significant effect of code-mixing intensity on factual support (p=0.1438).
Code-mixing intensity does not significantly affect hallucination rate (p=0.2825).

---
*Analysis on 1165 clean pairs across 3 CMI tertile levels*
