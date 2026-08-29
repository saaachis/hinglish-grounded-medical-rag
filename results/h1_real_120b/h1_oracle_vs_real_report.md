# H1 under oracle vs real retrieval

n = 467 pairs · generator `openai/gpt-oss-120b` · top-k = 1 · scored with the unified word-boundary lexicon (no 0.25 default).

> The original generator (`llama-3.1-8b-instant`) was decommissioned by Groq mid-project and returns 404 on every key. All three arms were therefore regenerated on a current model so that oracle-vs-real is not confounded with a model change. The cached llama outputs are retained per row.

---

## Q3 first: what happens when retrieval fails

| Arm | evidence | refusal rate | scoreable (has concepts) |
|---|---|---:|---:|
| zero | none | 0.2% | 92.9% |
| oracle | condition-filtered (ceiling) | 29.8% | 49.3% |
| real | FAISS top-k (deployed) | 33.8% | 52.0% |

Oracle vs real refusal, McNemar: n01=97, n10=78, p=0.1734

**The grounded arm declines rather than confabulates.** Refusals carry no clinical concept, so they are `nan` under the concept metric and disappear from a naive mean -- the system would look healthiest exactly where it fails. Report refusal rate beside every factuality number.

---

## Q1/Q2: grounding benefit, ceiling vs deployed

| Contrast | n paired | zero-shot | grounded | delta | Cohen's d | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| Oracle evidence (ceiling) | 223 | 0.2816 | 0.5475 | **+0.2659** | 0.662 | 5.547e-17 |
| Real retrieval (deployed) | 237 | 0.2762 | 0.5001 | **+0.2238** | 0.616 | 6.533e-17 |

Oracle − real grounded factuality: **+0.0875** (n=138 both-scoreable, p=0.0411). This is the inflation the condition filter bought.

---

## Q4: is factuality conditional on retrieval being correct?

| Retrieval top-1 | n | grounded factual (real) | refusal rate |
|---|---:|---:|---:|
| wrong | 417 | 0.4977 | 34.3% |
| correct | 50 | 0.5351 | 30.0% |

Mann-Whitney U = 3280, p = 0.6052. **Retrieval correctness does NOT predict factuality.** Either the condition-group label is too coarse to capture usefulness, or the generator is leaning on the prompt framing rather than the evidence. This is an important negative result -- investigate before writing.

Retrieval top-1 correct: **10.7%** · any of top-k correct: **10.7%**
