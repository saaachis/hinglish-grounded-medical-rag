# Random-evidence control -- the decisive test of the echo thesis

n = 268 paired rows · generator `openai/gpt-oss-20b` · same queries, same prompt, same lexicon as `h1_real_retrieval`.

## Refusal rates

| Arm | evidence | refusal |
|---|---|---:|
| oracle | condition-filtered | 82.8% |
| real | FAISS top-1 | 84.0% |
| random | uniform random case | 89.9% |

## The decisive contrast

| Scored against | mean | n scoreable |
|---|---:|---:|
| random evidence it was GIVEN (echo) | 0.3917 | 171 |
| oracle evidence for the query (correctness) | 0.3605 | 171 |
| real-retrieval arm (reference) | 0.4548 | 166 |
| oracle-evidence arm (reference) | 0.5023 | 168 |

real − random(echo-scored) = **+0.0680**, n=124, Wilcoxon p = 0.1427

## ⚠️ Selection gate -- read before quoting anything above

- random-arm refusal rate: **89.9%**
- refusals that STILL carry clinical concepts: **165** (68% of refusals)
- rows where BOTH arms actually answered: **n = 1**

A refusal here is not silence -- the model declines while *quoting the evidence back* ("evidence mein sirf X ka zikr hai"). That quoting is scored as concept overlap, so the means above are computed largely on REFUSAL TEXT rather than on answers.

### Reading

**INCONCLUSIVE.** Only 1 row(s) have a genuine answer from both arms, so the paired comparison has no power. This control cannot decide the echo thesis on this generator.

What it DOES establish: `openai/gpt-oss-20b` refuses 90% of the time on random evidence and ~83% even on condition-matched evidence, versus **18.1%** for the original `llama-3.1-8b-instant`. The two generators are not behaving like the same system, so every gpt-oss contrast (oracle-vs-real, retrieval-correctness-vs-factuality) is a comparison of refusal texts and must not be read as evidence about grounding.

**To settle the echo thesis, re-run on a generator that answers** -- either a current model with a lower refusal rate, or the same model with the "say you cannot confirm it" clause removed from the system prompt so refusal is not instruction-driven. Then repeat this control.

Random cases that coincidentally matched the query's condition: **6.7%** (chance level ~5.6% over 18 groups).
