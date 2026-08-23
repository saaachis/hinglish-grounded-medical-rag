# M4' -- reference-based factuality vs the circular evidence-based metric

n = 1165 cached generations. No API calls. Both metrics use the unified word-boundary lexicon with no 0.25 default.

## Headline

| Metric | Reference | Zero-shot | Grounded | Delta |
|---|---|---:|---:|---:|
| Evidence-based (circular) | retrieved evidence | 0.2658 | 0.5971 | **+0.3313** |
| M4' caption (unbiased) | image description | 0.1066 | 0.1528 | **+0.0461** |

## Paired tests

| Metric | n paired | Cohen's d | Wilcoxon p |
|---|---:|---:|---:|
| Evidence-based (circular) | 707 | 0.678 | 2.475e-50 |
| M4' caption (unbiased) | 701 | 0.181 | 8.567e-07 |

## Grounding effect with CLUSTER-bootstrap CIs on the paired delta

Resamples descriptions, not rows -- one description covers 22% of the corpus, so a row-level bootstrap would badly understate these intervals.

| Metric | delta | 95% cluster CI | excludes 0? |
|---|---:|---|---|
| Evidence-based (circular) | +0.2749 | [+0.2471, +0.3035] | yes |
| M4' caption (unbiased) | +0.0462 | [+0.0152, +0.0646] | yes |

### Per-arm marginal CIs (for reference only -- not the test)

| Arm | mean | 95% CI |
|---|---:|---|
| grounded | 0.1528 | [0.1294, 0.1901] |
| zero | 0.1066 | [0.0747, 0.1554] |

> The reference has only 325 distinct values across 1154 rows, so a row-level bootstrap would badly understate these intervals. The CIs above resample descriptions.

## Metric coverage (how often anything is measured at all)

| Arm | evidence-based | M4' |
|---|---:|---:|
| grounded | 74.2% | 73.6% |
| zero | 74.0% | 73.4% |

Per-condition table: `m4_per_condition.csv` -- **do not quote the aggregate without it** (see module docstring).
