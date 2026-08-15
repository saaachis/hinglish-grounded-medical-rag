# Implementation Report — What Changed, Why, and What It Cost

**Session date:** 2026-08-16
**Branch:** `devikas-updates` (commits `a302d18`, `441087b`, `859e32f`)
**Scope:** Tier 0 (unblock), Tier 1 (integrity), Tier 2.2 (H2 re-analysis). H₀₄ started.
**API spend:** ₹0 — every result below re-scores cached outputs. No new model calls.

---

## Executive summary

| | Outcome |
|---|---|
| **Claims strengthened** | 1 — the H2 factual-support finding (p 0.042 → **0.0003**) |
| **Claims withdrawn** | 2 — both hallucination×code-mixing effects were lexicon artefacts |
| **Fabrication risk** | **Closed.** The degenerate script never fed a published number |
| **Bugs found in new code** | 2 — both in code written this session, caught before results were used |
| **Reproducibility** | Restored. Tier 0 gate passes; results are now version-controlled |
| **Still outstanding** | The oracle-retrieval problem (needs Groq quota) and H₀₄ (needs a GPU) |

**Net judgement: worth it.** Three of the five things that changed were *removals* of results. That is the correct direction — a claim that dies to a repaired instrument would have died to a reviewer instead, and more expensively.

---

## Part 1 — Did the updates help? (Yes, specifically)

### 1.1 The blocker was never real

`results/`, the pairs CSV and the FAISS index were believed to exist only on Saachi's machine. In fact `handoff-tier1-essential.zip` was already committed to the repo, **unextracted**. Extracting it passes the Tier 0 gate exactly:

- 3,015 pairs, mean `similarity_score` **0.49966** (expected 0.4996636) ✅
- 1,165 cached generations with both arms, all four scores, CMI ✅

Devika's independently regenerated `multicare_filtered.csv` matches Saachi's **exactly** (61,316 / 39,652 / 18, byte-identical per-condition breakdown), closing the cross-machine mismatch risk.

### 1.2 The fabrication risk is closed — nothing needs withdrawing

`run_multicare_prototype.py` is degenerate by construction, and this was confirmed empirically:

| Signature | Measured |
|---|---:|
| Grounded rows that are a literal `"Based on the clinical report: " + evidence` | **100.0%** |
| `grounded_factual` mean | **0.9945** (99.3% exactly 1.0) |
| `grounded_hallucination` mean | **0.0000** (SD exactly 0.0) |

**But it did not feed any reported number.** The headline comes from `combined_h1h2/combined_scored.csv` — 1,165 genuine Groq generations, zero-shot 0.3190 / grounded 0.5535, matching the published 0.319 / 0.554. The ablation (0.6394) is likewise genuine. Quarantined at `results/multicare_h1/INVALID-DO-NOT-USE.md`.

### 1.3 The merged dataset is statistically sound

`combined_scored.csv` merges four separate runs, which raised a batch-effect concern. It is unfounded:

- Kruskal–Wallis across sources, `grounded_factual`: **H = 0.347, p = 0.951**
- Per-run means: 0.548 / 0.564 / 0.548 / 0.556
- Composition reconciles exactly: 374 + 362 + 356 + 73 = **1,165**

The first run contributes only 73 rows because 226 of its 299 were `[API_ERROR]`.

### 1.4 The H2 factual finding survived repair and got stronger

The legacy CMI was doubly broken — it **over-counted** (`doctor` fires in 71.0% of queries, `please` 38.0%) *and* **under-counted** (32.9% of tokens unknown to it, the OOV list almost entirely romanised Hindi: `mein` 5,340, `mere` 3,945, `hoon` 2,572). Both compress the score, which is why its SD was only 0.075.

After repair, holding the construct constant:

| Arm | Legacy lexicon | **Repaired lexicon** |
|---|---|---|
| Grounded factual | flat (ρ +0.015, p 0.612) | **flat (ρ −0.0006, p 0.983)** |
| Zero-shot factual | declines (ρ −0.068, p 0.042) | **declines (ρ −0.116, p 0.0003)** |

The effect roughly doubled and the p-value improved by two orders of magnitude. Stable across all three ambiguous-token policies.

### 1.5 Two bugs caught in this session's own code

- **Encoding mismatch (correctness).** `build_index.py` encodes documents at `max_seq_length = 128`. The first H₀₄ script did not set it, so queries were encoded at LaBSE's default (256) against documents at 128 — incomparable vectors and silently wrong similarities. Now pinned and asserted.
- **Over-strict leakage gate (would have false-failed).** The plan specified "assert no Q2 string contains any `condition_group` token." That fails on **79.6%** of rows, because a patient question legitimately names its own symptom ("a lump in my neck"). Only the templated underscore-joined label is leakage. Corrected.

---

## Part 2 — What the bad effects were (read this part)

### 2.1 🔴 Two findings were destroyed

Under the legacy lexicon, *both* arms' hallucination rates appeared to rise significantly with code-mixing (grounded ρ +0.061, BH p 0.0496; zero-shot ρ +0.081, BH p 0.022). Under the repaired lexicon **both go flat** (p 0.593 and 0.303).

Those were artefacts of counting `doctor` and `please` as Hindi. They must not be reported. **This is a genuine net loss of two results.**

### 2.2 🔴 The H2 claim is now narrower

| | Before | After |
|---|---|---|
| Claim | "Grounding is robust to code-mixing" | "Grounding protects **factual support only**" |
| Evidence | factual + hallucination | factual only |

The defensible sentence is now: *increasing Hindi content significantly degrades zero-shot factual support while leaving grounded factual support unchanged; hallucination is unaffected in either arm.* Weaker than the original framing, but true.

### 2.3 🟠 Every existing CMI figure is now invalid

The corrected measure is on a different scale (0–50 vs 0–1) **and a different rank order**. Legacy vs Das & Gambäck CMI correlate at **ρ = −0.53**. Consequences:

- All H2 charts in `research-poster-work/` must be regenerated.
- The **CMI tertile framing must be dropped**. Zero-shot factual is significant on the continuous measure (Spearman) but *not* across tertiles (Kruskal–Wallis p = 0.127) — bucketing discards information and costs power. The poster's tertile table is underpowered.

### 2.4 🟠 A methodological trap was created, and must be explained in the paper

Das & Gambäck CMI measures mixing **balance** (maximal at 50/50, **zero for monolingual text in either language**). The legacy measure was a Hindi **proportion**. They are near-perfect inverses here (**ρ = −0.94**) because these queries are Hindi-dominant (mean proportion 0.696).

This nearly produced a false headline. The corrected CMI first appeared to **reverse** the H2 result — zero-shot factual "rises" — when in fact "rises with balance" and "declines with proportion" are *the same finding on inverted scales*. Only adding `hindi_proportion()` as a diagnostic separated the two changes.

**Cost:** the paper must now explain two constructs, three ambiguity policies, and a diagnostic measure. That is real space and real reviewer patience.

### 2.5 🔴 This laptop cannot run H₀₄

Benchmarked LaBSE encoding on the target machine (4 cores, 12 GB):

| batch size | throughput | projected 9,045 texts |
|---:|---:|---:|
| 16 | 0.46 texts/s | **326 min** |
| 32 | 0.39 texts/s | 390 min |
| 64 | 0.38 texts/s | 399 min |

Larger batches are *worse* — the machine is memory-bound, not compute-bound (only ~1.9 GB free during the first attempt). Disabling the stray TensorFlow import did not help.

**Consequence:** H₀₄ — the paper's highest-value, zero-API-cost experiment — is a multi-hour job here. Mitigations applied: embeddings are cached to `data/embeddings/h4/`, and `--limit N` takes a condition-stratified subsample. **The definitive full-3,015 run should go to a Kaggle GPU session.**

### 2.6 🟠 The biggest problem is still untouched

**The oracle-retrieval issue remains unfixed.** Every headline number is still measured with evidence selected using a ground-truth condition label. Tier 1.2 (the real-retrieval H1 runner) requires ~1,200 Groq calls and has not been run. Until it is, the headline is not a deployed-system number.

---

## Part 3 — Honest scorecard

| Change | Helped? | Cost |
|---|---|---|
| Extract artifacts | ✅ Unblocked everything | None |
| `.gitignore` fix | ✅ Results now versioned | None |
| Provenance audit | ✅ Closed a fabrication risk | ~3 h; found nothing to withdraw |
| Batch-effect check | ✅ Validated the merge | ~1 h |
| Lexicon repair | ✅ Doubled the effect size | **Killed 2 hallucination findings** |
| Das & Gambäck CMI | ⚠️ Standard metric, but wrong construct for this question | **Invalidated all CMI figures**; needs explaining |
| `hindi_proportion` diagnostic | ✅ Prevented a false "reversal" headline | Extra methodology to describe |
| Bootstrap CIs + BH-FDR | ✅ Rigour; effects survive correction | Slow (~4 min/run) |
| H₀₄ implementation | ⚠️ Correct and gated, but not yet runnable here | Revealed a hard hardware limit |

---

## Part 4 — What to do next, in order

1. **Run H₀₄ on Kaggle** (30 GPU-hrs/week, free). Full 3,015 × 3 variants, minutes on GPU. This is the paper's headline experiment.
2. **Run the real-retrieval H1** (Tier 1.2, ~1,200 Groq calls). Fixes the oracle problem — the single biggest outstanding threat.
3. **Regenerate every CMI figure** on `hindi_prop_v2`, and drop the tertile framing for continuous regression.
4. **Unify the 5 concept lexicons** (Tier 1.4) — still outstanding, and the last known metric-validity issue.
5. Only then: H₀₃ provenance, the LLM-judge, the CMI ladder.

---

## Files added

| Path | Purpose |
|---|---|
| `src/analysis/cmi.py` | Das & Gambäck CMI + `hindi_proportion` + repaired lexicon |
| `src/analysis/h2_per_arm.py` | Per-arm H2 with bootstrap CIs and BH-FDR |
| `src/analysis/h2_recompute.py` | H2 under corrected CMI, 3-policy sensitivity |
| `src/analysis/h4_retrieval.py` | H₀₄ Q1/Q2/Q3 with leakage gate, caching, `--limit` |
| `results/h2_per_arm/*` | Reports + stats CSVs |
| `results/multicare_h1/INVALID-DO-NOT-USE.md` | Quarantine notice |
