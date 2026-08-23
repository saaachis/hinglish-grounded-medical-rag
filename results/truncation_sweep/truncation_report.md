# Adaptive truncation sweep

n = 3015 queries, max_k = 10, real FAISS index, no condition filter.

## The shipped rule cannot fire

- top-1 similarity, mean: **0.4966**
- threshold the rule requires at ratio=0.5: **0.2483**
- largest adjacent gap anywhere in the data: **0.10923**
- mean adjacent gap: **0.00403**

The rule waits for a drop roughly **2.3x larger** than the biggest gap that exists.

## Sweep

| System | fires | mean kept | precision of kept | recall (>=1 correct kept) |
|---|---:|---:|---:|---:|
| `adaptive(ratio=0.5)` | 0.0% | 10.00 | 0.1135 | 0.6083 |
| `adaptive(ratio=0.2)` | 0.0% | 10.00 | 0.1135 | 0.6083 |
| `adaptive(ratio=0.1)` | 1.0% | 9.91 | 0.1139 | 0.6033 |
| `adaptive(ratio=0.05)` | 15.0% | 8.70 | 0.1148 | 0.5413 |
| `adaptive(ratio=0.02)` | 66.7% | 4.54 | 0.1124 | 0.3234 |
| `adaptive(ratio=0.01)` | 95.6% | 2.10 | 0.1127 | 0.1934 |
| `fixed_k=1` | -- | 1.00 | 0.1141 | 0.1141 |
| `fixed_k=3` | -- | 3.00 | 0.1150 | 0.2915 |
| `fixed_k=5` | -- | 5.00 | 0.1138 | 0.4133 |
| `fixed_k=10` | -- | 10.00 | 0.1135 | 0.6083 |

## Reading

- At the shipped `ratio=0.5` the rule is inert: it returns all 10 cases on every query, so the system is a plain fixed-k=10 retriever and the adaptive-selection claim is unsupported.
- Precision and recall trade off monotonically with `mean_kept`. The honest test is whether an adaptive setting beats the **fixed k with the same mean_kept**.

## Verdict: adaptive vs the nearest fixed k

| Adaptive | mean kept | vs | fixed k | precision | recall | wins? |
|---|---:|---|---|---|---|---|
| `adaptive(ratio=0.5)` | 10.00 | vs | `fixed_k=10` (k=10) | +0.0000 | +0.0000 | no |
| `adaptive(ratio=0.2)` | 10.00 | vs | `fixed_k=10` (k=10) | +0.0000 | +0.0000 | no |
| `adaptive(ratio=0.1)` | 9.91 | vs | `fixed_k=10` (k=10) | +0.0004 | -0.0050 | no |
| `adaptive(ratio=0.05)` | 8.70 | vs | `fixed_k=10` (k=10) | +0.0013 | -0.0670 | no |
| `adaptive(ratio=0.02)` | 4.54 | vs | `fixed_k=5` (k=5) | -0.0013 | -0.0899 | no |
| `adaptive(ratio=0.01)` | 2.10 | vs | `fixed_k=3` (k=3) | -0.0023 | -0.0982 | no |

**Adaptive truncation wins in 0 of 6 settings.**

Precision is essentially flat (~0.113-0.115) across every setting, adaptive and fixed alike, while recall falls monotonically as fewer cases are kept. That means **the similarity gap carries no information about relevance** -- cutting on it discards correct evidence at the same rate as incorrect evidence.

**Recommendation:** report this as a negative result and drop the "MMed-RAG-style adaptive context selection" claim from `README.md`, `config/config.yaml` and the poster. A fixed k is simpler and strictly better at every budget. The honest finding -- *that a published adaptive-selection heuristic does not transfer to this setting* -- is worth more than the claim was.
