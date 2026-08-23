# H04 - Retrieval-Stage Code-Mixing Penalty

n = 3015 pairs (of 3,015 available), index = 10000 MultiCaRe cases, encoder = LaBSE (max_seq_length=128), top-k = 10.
Relevance: retrieved case `condition_group` == query `condition_query`.
No API calls.

## Query conditions

| Variant | Meaning |
|---|---|
| `Q1_hinglish` | The deployed path |
| `Q2_english_question` | Translation ceiling (caption stripped) |
| `Q3_english_plus_caption` | Multimodal ceiling (caption retained) |

> Q3's caption contains the underscore-joined condition-group label, which is
> the relevance label itself. Q3 is an upper bound, **not** an English baseline.
> The unconfounded code-mixing penalty is Q1 vs Q2.

## Retrieval quality

| Variant | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| `Q1_hinglish` | 0.1144 | 0.2915 | 0.4136 | 0.6083 | 0.2432 | 0.4098 |
| `Q2_english_question` | 0.1602 | 0.3512 | 0.4799 | 0.6886 | 0.2985 | 0.4863 |
| `Q3_english_plus_caption` | 0.2143 | 0.4232 | 0.5579 | 0.7522 | 0.3592 | 0.5558 |
| `random_floor(analytic)` | 0.0626 | 0.1761 | 0.2755 | 0.4739 | -- | -- |

## Paired comparisons

| Comparison | dR@1 | 95% CI | McNemar n01/n10 | McNemar p | Wilcoxon(RR) p | dMRR |
|---|---:|---|---|---:|---:|---:|
| Q2_english_question - Q1_hinglish | +0.0458 | [+0.0292, +0.0627] | 400/262 | 9.13e-08 | 2.09e-12 | +0.0553 |
| Q3_english_plus_caption - Q2_english_question | +0.0541 | [+0.0375, +0.0710] | 412/249 | 2.41e-10 | 1.23e-17 | +0.0607 |

## Reading

- Deployed Hinglish R@1 = **0.1144**, against an analytic random floor of **0.0626** (1.83x the floor).
- English question R@1 = **0.1602**; the unconfounded code-mixing penalty is **+0.0458** absolute.
- The Q2->Q3 increment is the headroom a perfect image reader would add.

Per-condition breakdown: `h4_per_condition.csv`.