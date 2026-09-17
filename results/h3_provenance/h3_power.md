# H03: what a powered provenance experiment would require

Computed from the cached n = 160 run (gpt-oss-120b, four matched corpora). No API calls.

## Why the usable sample collapsed

| Evidence corpus | Refusal | Scoreable (asserts >= 1 concept) | of which ALSO flagged refusal | Answered *and* scoreable |
|---|---:|---:|---:|---:|
| MultiCaRe (case reports) | 88.1% | 63 (39.4%) | 59 | 4 (2.5%) |
| Shuffled MultiCaRe control | 82.5% | 61 (38.1%) | 53 | 8 (5.0%) |
| PubMedQA (abstracts) | 79.4% | 70 (43.8%) | 53 | 17 (10.6%) |
| MMedBench (exam text) | 76.2% | 66 (41.2%) | 56 | 10 (6.2%) |

> **A validity problem the reviewers did not raise.** Most scoreable answers are also
> flagged as refusals: hedged replies that decline to answer yet still name a concept
> (MultiCaRe: 59 of 63).
> The paper's Table 5 mean F1 is therefore computed largely on refusals, not on answers.
> Requiring a genuine answer leaves only 0 queries usable
> under all four conditions. Report the F1 column with this caveat or drop it.

- Jointly scoreable under **all four** conditions: **13 of 160** (8.1%).
- If the four conditions were independent, the expected joint rate would be 2.7%; the observed 8.1% is higher, so refusals are correlated across conditions (some queries are simply harder).

## Queries needed for a four-way omnibus

| Target usable queries | Queries to run | Generation calls | Tokens (approx.) | Days at one Groq organisation's quota |
|---:|---:|---:|---:|---:|
| 50 | 616 | 2,464 | 2,833,600 | 14.2 |
| 100 | 1,231 | 4,924 | 5,662,600 | 28.3 |
| 200 | 2,462 | 9,848 | 11,325,200 | 56.6 |

At the observed joint rate, the submitted design would need **616 queries** for a modest 50 usable ones.

The refusal-suppressing prompt (`--prompt direct`, 67% -> 0% refusal on a probe) is the
lever that changes this: at a 90% per-condition scoreable rate the joint rate would be
~66%, needing only ~77 queries for 50 usable ones.
That is the design a follow-up should use.

## Effect sizes and the sample each would need

Pairwise F1 differences on the queries scoreable in BOTH conditions of each pair,
with the n required to detect that effect at 80% power, alpha = 0.05 (two-sided).

| Comparison | n available | Mean F1 difference | Cohen's dz | p | n for 80% power |
|---|---:|---:|---:|---:|---:|
| mmedbench vs pubmedqa | 34 | -0.0847 | -0.219 | 0.21 | 164 |
| multicare vs pubmedqa | 38 | -0.0553 | -0.177 | 0.238 | 252 |
| pubmedqa vs shuffled | 38 | +0.0693 | +0.163 | 0.36 | 296 |
| mmedbench vs shuffled | 34 | -0.0259 | -0.070 | 0.55 | 1,624 |
| multicare vs shuffled | 32 | +0.0094 | +0.024 | 1 | 13,625 |
| mmedbench vs multicare | 34 | +0.0045 | +0.012 | 0.958 | 58,758 |

## What the paper should say

- H03's primary outcome (answer quality) is **not adequately tested**: neither rejected
  nor supported. Report it as exploratory and provisional.
- The refusal effect (Cochran's Q) is a **secondary outcome**, on a different dependent
  variable from the one H03 names.
- A properly powered replication needs the low-refusal prompt and roughly 77-305 queries. At the refusal rates
  observed here the same design needs 616-2,462 queries (2.8-11.3M tokens, 14-57 days of one
  organisation's quota), which is why it was not run.
- Detecting the largest observed pairwise effect (dz = 0.22) at 80% power needs about 164 jointly scoreable pairs.
