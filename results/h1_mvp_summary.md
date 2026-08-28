# H1 MVP Summary

- Samples evaluated: **150**
- Retrieval top-1 hit rate: **0.627**
- Retrieval top-k hit rate: **0.893**
- Mean factual score (zero-shot): **0.740**
- Mean factual score (grounded): **0.786**
- Hallucination rate (zero-shot): **0.240**
- Hallucination rate (grounded): **0.167**

## Paired Comparison (H1)
- Test: **wilcoxon_signed_rank**
- Statistic: **117.0000**
- p-value: **0.015976**
- Effect size (Cohen's d): **0.223**
- 95% CI (grounded - zero-shot): **[0.013, 0.079]**

## Preliminary Interpretation
- Early results support H1: grounded outputs outperform zero-shot on factual support.
- This is an MVP result and should be interpreted with small-sample caution.
