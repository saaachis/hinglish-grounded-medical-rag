# H1 under oracle vs real retrieval

n = 256 pairs · generator `openai/gpt-oss-120b` · top-k = 1 · scored with the unified word-boundary lexicon (no 0.25 default).

> The original generator (`llama-3.1-8b-instant`) was decommissioned by Groq mid-project and returns 404 on every key. All three arms were therefore regenerated on a current model so that oracle-vs-real is not confounded with a model change. The cached llama outputs are retained per row.

---

## Q3 first: what happens when retrieval fails

| Arm | evidence | refusal rate | scoreable (has concepts) |
|---|---|---:|---:|
| zero | none | 0.0% | 92.6% |
| oracle | condition-filtered (ceiling) | 30.5% | 50.0% |
| real | FAISS top-k (deployed) | 33.2% | 52.3% |

Oracle vs real refusal, McNemar: n01=53, n10=46, p=0.5467

**The grounded arm declines rather than confabulates.** Refusals carry no clinical concept, so they are `nan` under the concept metric and disappear from a naive mean -- the system would look healthiest exactly where it fails. Report refusal rate beside every factuality number.

---

## Q1/Q2: grounding benefit, ceiling vs deployed

| Contrast | n paired | zero-shot | grounded | delta | Cohen's d | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| Oracle evidence (ceiling) | 124 | 0.2710 | 0.5560 | **+0.2849** | 0.694 | 1.015e-10 |
| Real retrieval (deployed) | 131 | 0.2649 | 0.4996 | **+0.2347** | 0.661 | 1.003e-10 |

Oracle − real grounded factuality: **+0.0748** (n=76 both-scoreable, p=0.226). This is the inflation the condition filter bought.

---

## Q4: is factuality conditional on retrieval being correct?

| Retrieval top-1 | n | grounded factual (real) | refusal rate |
|---|---:|---:|---:|
| wrong | 235 | 0.5070 | 33.2% |
| correct | 21 | 0.4167 | 33.3% |

Mann-Whitney U = 589, p = 0.4639. **Retrieval correctness does NOT predict factuality.** Either the condition-group label is too coarse to capture usefulness, or the generator is leaning on the prompt framing rather than the evidence. This is an important negative result -- investigate before writing.

Retrieval top-1 correct: **8.2%** · any of top-k correct: **8.2%**
