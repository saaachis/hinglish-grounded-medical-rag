# Paper Readiness — What Is Done, What Is Missing, What To Do Next

**Date:** 2026-08-24 · assessed against `saachi-hardening` (pushed)
**Target:** ICCSDI 2026 / SN Computer Science template

---

## 1. Asset inventory — what you can already write up

Nine results now exist, are reproducible from a clone, and are defensible.

| # | Result | n | Where | Strength |
|---|---|---:|---|---|
| R1 | **H₀₁ grounding effect, dual-metric** — circular +0.275 vs unbiased +0.046 | 1,165 | `results/m4_caption/` | ⭐ Headline |
| R2 | **H₀₄ retrieval code-mixing penalty** — +0.0458 R@1, p=9×10⁻⁸ | 3,015 | `results/h4_retrieval/` | ⭐ Strongest |
| R3 | **Oracle vs real retrieval** — Δ=+0.076, p=0.106 n.s. | 268 | `results/h1_real_retrieval/` | Strong |
| R4 | **Refusal behaviour** — 84% grounded vs 0% zero-shot | 268 | same | ⭐ Safety result |
| R5 | **Retrieval correctness ↛ factuality** — p=0.53 | 268 | same | ⭐ Key negative |
| R6 | **Retrieval baselines** — 4 systems × 3 query conditions | 3,015 | `results/h4_retrieval/h4_baselines.csv` | Table 1 |
| R7 | **H₀₂ code-mixing × generation** — per-arm, corrected CMI | 1,165 | `results/h2_per_arm/` | Solid |
| R8 | **Adaptive truncation** — 0 of 6 wins, negative result | 3,015 | `results/truncation_sweep/` | Ablation |
| R9 | **Pairing contribution** — 11 → 3,015 pairs, 274× | — | `plans/limitation/` | Methods |

That is enough material for a full paper. **The gap is not quantity — it is one missing
control, one coherence problem, and the write-up itself.**

---

## 2. 🔴 The three things that actually block submission

### BLOCKER 1 — The random-evidence control (~300 calls, half a day)

Five independent results now point at the same explanation: **the measured grounding
benefit is largely an *echo* effect** — the metric rewards restating whatever evidence was
supplied, relevant or not.

| Evidence | Result |
|---|---|
| Oracle vs real retrieval | p = 0.106 — the condition filter buys nothing |
| Retrieval correctness → factuality | p = 0.53 — relevance does not predict quality |
| Circular vs unbiased metric | effect shrinks **6×** |
| BM25 vs LaBSE on Hinglish | lexical **beats** the cross-lingual encoder |
| Refusal rate identical oracle vs real | p = 0.79 — evidence quality changes nothing |

All five are **indirect**. A random-evidence arm tests it head-on: ground the model on a
case drawn uniformly at random and re-score.

- If factuality **barely drops** → the echo thesis is proven, and it becomes the paper's
  central, novel contribution.
- If it **drops sharply** → retrieval genuinely matters, and R3/R5 need another explanation.

Either outcome is publishable; not knowing is not. **A reviewer will ask for this control
by name.** It is the cheapest high-value experiment left and it should be run first.

### BLOCKER 2 — Results now span two generators, one of which is dead

`llama-3.1-8b-instant` was decommissioned mid-project and 404s on every key. So:

- R1, R7 (n=1,165) are **llama** — cannot be extended, cannot be reproduced by a reviewer.
- R3, R4, R5 (n=268) are **gpt-oss-20b**.

The paper cannot silently mix them. Three options:

| Option | Cost | Verdict |
|---|---|---|
| **A. Re-run everything on gpt-oss-20b** | ~3,500 calls ≈ 3–4 days of quota | Cleanest. Recommended if time allows |
| **B. Report gpt-oss as primary, llama as a replication appendix** | 0 extra | **Recommended fallback.** Honest, and the model-transfer contrast becomes a feature |
| **C. Keep llama primary** | 0 | ❌ Reject — the headline model would be irreproducible |

Whichever you pick, the decommissioning **must** appear in Limitations. It is also a real
finding about doing reproducible research on free LLM tiers.

### BLOCKER 3 — The repo still advertises a system that does not exist

`README.md` and `config/config.yaml` describe LLaVA-1.5, BioMedCLIP, QLoRA, DPO, and
metrics (`mmfcm`, BLEU, ROUGE) that nothing computes. `config.yaml` still lists
`adaptive_truncation: true`, now known to be a no-op. `requirements.txt` omits `groq`
(imported at module load) while pinning `peft`/`trl`/`bitsandbytes`/`open_clip_torch` for
code that does not exist.

A reviewer who opens the repository — and SN Computer Science requires a code-availability
statement, so they will — finds an architecture diagram that does not match the code.
**~2 hours of work, disproportionate credibility cost if skipped.**

---

## 3. 🟠 Should-have, in priority order

| # | Item | Cost | Why |
|---|---|---|---|
| 4 | **H₀₃ provenance** | ~1,800 calls | The proposal commits to three hypotheses; delivering two invites "why not H₃?". Use matched-topic, equal-size (~1,800 doc) indexes — PubMedQA is only 2.1% on-topic and MMedBench is 57.6% Chinese |
| 5 | **Swap LaBSE → multilingual-e5-base** | Free (CPU) | e5 beats LaBSE on every query condition, +53% R@1 on English. Either adopt it, or report honestly that the deployed encoder is not the best available |
| 6 | **Finish H1 to n≈400** | Resumes on quota | 268 is ample (p<10⁻⁸); this only narrows CIs |
| 7 | **Figures for R1–R6** | Free | Only the H2 figures are regenerated. Need: H₀₄ recall bars, baseline table, metric-shrinkage comparison, refusal rates, truncation sweep. Reuse `h2_figures.py` conventions (300 DPI + vector PDF) |
| 8 | **Re-run the Phase-6 ablation** under the unified lexicon | Free | The +0.069 structured-vs-raw result is currently partly a lexicon artefact |
| 9 | **Zenodo deposit + DOI** | Free | SN Computer Science **requires** a Data Availability statement in Declarations. ~380 MB |

---

## 4. 🟡 Nice-to-have (cut these first)

- Gemini LLM-judge (M3) — a third metric; the dual-metric contrast already carries the argument
- Controlled CMI ladder — extends H₀₂; the corrected continuous analysis already works
- 100-sample human validation — point it at the ~30% of rows where the lexicon fires nothing
- Generator-scale ablation — now moot, the model landscape changed under you

---

## 5. Recommended sequence

**Week 1 — close the science**
1. Random-evidence control (Blocker 1) ← *start here*
2. Decide the generator question (Blocker 2); if Option A, launch the re-run immediately — it is quota-bound, so it must run in the background from day one
3. H₀₃ provenance, matched-topic

**Week 2 — close the artefact**
4. README / config / requirements (Blocker 3)
5. Figures for R1–R6
6. Encoder swap decision; re-run Phase-6 ablation under the unified lexicon
7. Zenodo deposit, tagged GitHub release

**Week 3 — write**
8. Draft against the template. One notebook must regenerate every table and figure from
   cached CSVs. If a number is in the paper and not produced by that notebook, it does not
   go in the paper.

---

## 6. The paper this now supports

The original framing — *"RAG improves factuality for Hinglish clinical queries"* — is no
longer the strongest available claim, and would not survive review: the effect is 6×
smaller under unbiased measurement and does not depend on retrieval being correct.

What the data now supports, and what is genuinely novel:

> **Standard evidence-based factuality metrics substantially overstate the benefit of
> retrieval-augmented generation for code-mixed clinical queries, because they reward
> echoing the supplied evidence rather than answering correctly. Under a reference the
> model never saw, the benefit is real but roughly six times smaller — and it does not
> depend on the retrieved evidence being topically relevant. Meanwhile code-mixing imposes
> a large, separately measurable penalty at the *retrieval* stage, where a lexical baseline
> matches a cross-lingual encoder.**

Three contributions fall out of that, each already evidenced:

- **C1** A dual-metric evaluation protocol (circular vs unbiased) with the shrinkage
  quantified — R1, plus R5 and the random-evidence control as mechanism.
- **C2** The retrieval-stage code-mixing penalty, measured against gold human translations
  with a leakage gate — R2, R6.
- **C3** Safety behaviour under retrieval failure: grounded models decline rather than
  confabulate — R4.

Plus the pairing contribution (R9) as a methods finding and the truncation negative result
(R8) as an ablation.

**This is a better paper than the original plan, and most of it is already measured.**
