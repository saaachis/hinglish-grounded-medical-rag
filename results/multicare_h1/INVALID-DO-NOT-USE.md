# INVALID RESULT -- DO NOT CITE

`h1_multicare_scored.csv` in this directory was produced by
`src/prototype/run_multicare_prototype.py`, which **does not call a language
model at all.** Both of its arms are constructed strings, so the file is an
arithmetic identity rather than an experiment.

## What the script actually does

| Arm | Construction | Consequence |
|---|---|---|
| Zero-shot | `"Patient query: " + the query itself` (`:113-118`) | Echoes the question |
| Grounded | `"Based on the clinical report: " + evidence[:500 words]` (`:127-128`) | Concepts are a **subset of the evidence by definition** |

## Measured signature (verified)

| Metric | Value | Why it is impossible |
|---|---:|---|
| grounded rows starting `"Based on the clinical report:"` | **100.0%** | It is a literal string concatenation |
| `grounded_factual` mean | **0.9945** | 99.3% of rows are exactly 1.0 |
| `grounded_hallucination` mean | **0.0000** | SD is exactly 0.0 -- no row ever differs |

A "grounded factual support" of 0.99 with a hallucination rate of *identically*
zero is not a model result. It is the scorer comparing the evidence against
itself.

## Provenance audit outcome -- no published number is affected

This file did **not** feed any reported result. The headline H1/H2 numbers come
from `results/combined_h1h2/combined_scored.csv` (1,165 real Groq generations:
zero-shot 0.3190 / grounded 0.5535), and the Phase-6 ablation comes from
`results/multicare_h1_ablation/` (grounded 0.6394). Both were verified to
contain genuine, non-echoed model output. **Nothing needs to be withdrawn.**

The file is retained only so this audit is reproducible. Do not merge it into
any analysis, and do not use `run_multicare_prototype.py` for new experiments.

See `plans/next-phase/unified-research-plan.md` section 1.1-B.
