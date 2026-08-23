# H1 under oracle vs real retrieval

n = 268 pairs · generator `openai/gpt-oss-20b` · top-k = 1 · scored with the unified word-boundary lexicon (no 0.25 default).

> The original generator (`llama-3.1-8b-instant`) was decommissioned by Groq mid-project and returns 404 on every key. All three arms were therefore regenerated on a current model so that oracle-vs-real is not confounded with a model change. The cached llama outputs are retained per row.

---

## Q3 first: what happens when retrieval fails

| Arm | evidence | refusal rate | scoreable (has concepts) |
|---|---|---:|---:|
| zero | none | 0.0% | 90.7% |
| oracle | condition-filtered (ceiling) | 82.8% | 62.7% |
| real | FAISS top-k (deployed) | 84.0% | 61.9% |

Oracle vs real refusal, McNemar: n01=31, n10=28, p=0.7948

**The grounded arm declines rather than confabulates.** Refusals carry no clinical concept, so they are `nan` under the concept metric and disappear from a naive mean -- the system would look healthiest exactly where it fails. Report refusal rate beside every factuality number.

---

## Q1/Q2: grounding benefit, ceiling vs deployed

| Contrast | n paired | zero-shot | grounded | delta | Cohen's d | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| Oracle evidence (ceiling) | 161 | 0.3142 | 0.5045 | **+0.1902** | 0.500 | 7.237e-09 |
| Real retrieval (deployed) | 158 | 0.2720 | 0.4557 | **+0.1837** | 0.492 | 1.058e-08 |

Oracle − real grounded factuality: **+0.0755** (n=125 both-scoreable, p=0.106). This is the inflation the condition filter bought.

---

## Q4: is factuality conditional on retrieval being correct?

| Retrieval top-1 | n | grounded factual (real) | refusal rate |
|---|---:|---:|---:|
| wrong | 230 | 0.4471 | 85.2% |
| correct | 38 | 0.5079 | 76.3% |

Mann-Whitney U = 1644, p = 0.5307. **Retrieval correctness does NOT predict factuality.** Either the condition-group label is too coarse to capture usefulness, or the generator is leaning on the prompt framing rather than the evidence. This is an important negative result -- investigate before writing.

Retrieval top-1 correct: **14.2%** · any of top-k correct: **14.2%**
