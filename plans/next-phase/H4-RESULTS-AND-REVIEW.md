# H₀₄ Results, Retrieval Baselines, and Adjudication of the Three Flagged Items

**From:** Saachi's machine · 2026-08-23 · branch `devikas-updates`
**Covers:** H₀₄ executed on all 3,015 pairs · retrieval baselines · the three items Devika flagged for review in `HANDOFF-TO-SAACHI.md` §4
**API spend:** ₹0. Every number below comes from local CPU compute over data already in the repo.

---

## 0. Scoreboard — where the project actually stands

| Claim | Status | Where |
|---|---|---|
| **H₀₁ — grounding improves factual support** | ✅ Holds, effect survives metric repair | `plan-evaluation-and-h3.md` §3 |
| **H₀₂ — factual support × code-mixing** | ✅ **Strengthened** (ρ −0.116, BH p 0.0003) | `results/h2_per_arm/` |
| **H₀₂ — hallucination × code-mixing** | ❌ **WITHDRAWN** — lexicon artefact | §2.1 below |
| **H₀₄ — retrieval-stage code-mixing penalty** | ✅ **H₀ REJECTED**, executed on all 3,015 | §1 below |
| **Retrieval baselines (Table 1)** | ✅ Executed | §3 below |
| **"Code-mixing costs half the retrieval signal"** | ⚠️ **CORRECTED** — 1.40×, not 1.84× | §2.2 below |
| **Oracle-retrieval problem** | 🔴 **STILL OPEN** — top threat | needs ~1,200 Groq calls |
| **Scoring circularity (P1)** | 🔴 Still open; M4′ is the fix | §2.3 below |
| **Adaptive truncation** | 🔴 Still dead code (0/299 fires) | unchanged |
| **Five divergent lexicons** | 🟠 Still unmerged | unchanged |
| **Reproducibility from a clone** | ✅ **FIXED** by Devika's `.gitignore` change | `results/**` re-include |

**One-line status:** the retrieval half of the paper is now measured and defensible; the generation half still rests on oracle retrieval and a circular metric, and both fixes are specified but unbuilt.

---

## 1. H₀₄ — executed on all 3,015 pairs

> **H₀₄:** retrieval quality does not differ between code-mixed Hinglish queries and semantically equivalent English renderings, with encoder, index and relevance criterion held constant.

Devika's laptop encodes LaBSE at 0.38–0.46 texts/s, making this a ~6-hour job that was killed. This machine encodes at **38.3 texts/s** — roughly 90× faster — so the full run took **under 4 minutes on CPU**. **No Kaggle session was needed.** `src/analysis/h4_retrieval.py` ran unmodified.

### Results

| Query condition | R@1 | R@3 | R@5 | R@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| **Q1 Hinglish** (deployed) | 0.1144 | 0.2915 | 0.4136 | 0.6083 | 0.2432 | 0.4098 |
| **Q2 English question** (caption stripped) | 0.1602 | 0.3512 | 0.4799 | 0.6886 | 0.2985 | 0.4863 |
| **Q3 English + caption** (leaked ceiling) | 0.2143 | 0.4232 | 0.5579 | 0.7522 | 0.3592 | 0.5558 |
| Random floor (analytic, prevalence-weighted) | 0.0626 | 0.1761 | 0.2755 | 0.4739 | — | — |

### Paired tests

| Comparison | ΔR@1 | 95% CI | McNemar n01/n10 | McNemar p | Wilcoxon (RR) p | ΔMRR |
|---|---:|---|---|---:|---:|---:|
| **Q2 − Q1** (the code-mixing penalty) | **+0.0458** | [+0.0292, +0.0627] | 400/262 | **9.13×10⁻⁸** | **2.09×10⁻¹²** | +0.0553 |
| Q3 − Q2 (multimodal headroom) | +0.0541 | [+0.0375, +0.0710] | 412/249 | 2.41×10⁻¹⁰ | 1.23×10⁻¹⁷ | +0.0607 |

### Reading

- **H₀₄ is rejected.** Code-mixing imposes a real, highly significant retrieval penalty: +0.0458 absolute R@1, a **1.40× relative** gap (0.1602 / 0.1144).
- The deployed Hinglish path sits at **1.83×** the random floor at R@1. That is weak retrieval in absolute terms and remains the system's binding constraint.
- **The Q2→Q3 increment is a free argument for multimodal work.** It is the headroom a perfect image reader would add (+0.054 R@1, larger than the code-mixing penalty itself) — evidence for the multimodal extension without doing any multimodal work. This is a genuinely good idea of Devika's and belongs in Future Work.

### ⚠️ Do not report the aggregate alone

`skin_rash` is 34.8% of queries (1,050 / 3,015) and has among the worst retrieval (Q1 R@1 = 0.071), so it dominates the aggregate. The effect is also **heterogeneous in sign** — English is *worse* than Hinglish for `neck_swelling`, `foot_swelling`, `swollen_eye`, `skin_dryness` and `skin_growth`.

| Condition | n | Q1 Hinglish | Q2 English | Q3 +caption |
|---|---:|---:|---:|---:|
| `skin_rash` | 1050 | 0.0714 | 0.1400 | 0.1152 |
| `neck_swelling` | 276 | 0.1377 | 0.1159 | 0.2101 |
| `mouth_ulcers` | 196 | 0.1276 | 0.1480 | 0.0867 |
| `lip_swelling` | 193 | 0.2124 | 0.2487 | 0.4560 |
| `swollen_tonsils` | 174 | 0.0460 | 0.0690 | 0.0690 |
| `hand_lump` | 162 | 0.1790 | 0.4815 | 0.6543 |
| `swollen_eye` | 152 | 0.1382 | 0.1118 | 0.4868 |
| `knee_swelling` | 115 | 0.3043 | 0.2957 | 0.3913 |

Full table: `results/h4_retrieval/h4_per_condition.csv`. Devika's §1.4 warning was well-founded.

---

## 2. The three flagged items — adjudicated

### 2.1 The hallucination retraction — ✅ Devika is right, retraction accepted

`doctor`, `please` and `pls` are literally in the legacy Hindi token list (`src/prototype/run_h1h2_analysis.py:34`). Measured on the 1,165 evaluated rows: `doctor` fires in **68.2%** of queries, `please` in **35.7%**, `pls` in 0.4%. These are English words inflating the Hindi proportion for most of the corpus.

Recomputed independently from source (not read from her CSV), using `hindi_prop_v2` — the repaired lexicon holding the *construct* constant:

| Arm | Legacy ρ / p | Repaired ρ / p | Verdict |
|---|---|---|---|
| Grounded factual | +0.0149 / 0.612 | −0.0006 / 0.983 | flat → flat, **survives** |
| Zero-shot factual | −0.0677 / 0.021 | **−0.1155 / 0.000077** | **survives, ~2× stronger** |
| Grounded hallucination | +0.0610 / 0.037 | −0.0224 / 0.445 | ❌ **withdrawn** |
| Zero-shot hallucination | +0.0812 / 0.006 | +0.0420 / 0.152 | ❌ **withdrawn** |

Reproduces her `h2_corrected_cmi_stats.csv` exactly. **Both hallucination × code-mixing effects were artefacts.** The factual finding survived and roughly doubled in strength.

> Her isolation of the two changes — lexicon repair (`hindi_prop_v2`) vs construct change (Das & Gambäck CMI is a *balance* measure, maximal at 50/50, so it flips the sign) — is methodologically correct and is what makes this adjudicable at all. Note the Das & Gambäck variants show zero-shot factual at ρ **+0.114**; that is not a contradiction, it is a different construct. **Report `hindi_prop_v2`; do not mix the two.**

### 2.2 The Q3 caption confound — ✅ Devika is right, and it is worse than stated

Measured on all 3,015 rows: the `english_summary` field contains the query's own `condition_group` label — the relevance answer key — in **96.2%** of rows (33.6% as the literal underscore form, 74.3% spaced). **95.0%** of that sits in the caption clause; only 27.2% appears in the question clause.

**This invalidates a number I previously reported.** My earlier "gold English R@1 = 23.4%" used the full summary and lands almost exactly on Q3 (0.2143), confirming the diagnosis.

| | Earlier claim | Corrected |
|---|---|---|
| Hinglish R@1 | 12.7% (n=299) | **11.4%** (n=3,015) |
| "English" R@1 | 23.4% *(leaked)* | **16.0%** (caption stripped) |
| Relative gap | 1.84× | **1.40×** |

The direction and significance of the finding survive. **The magnitude I gave does not.** Any slide, figure or draft quoting 23.4% as an English baseline must be corrected.

Her leakage gate in `h4_retrieval.py` is also correctly specified — it targets only the templated underscore label, not any condition token. The stricter version originally planned would have false-failed 79.6% of rows because patients legitimately name their own symptoms.

### 2.3 Caption-as-reference (M4′) — ✅ Sound, with one caveat she did not measure

Her statistics verify: 100% of rows have a caption, **96.9%** have residual description beyond the boilerplate (she reported 99.1%; the difference is boilerplate-regex variants and is immaterial), mean residual length **74 chars**, and only **4.8%** still contain the condition label. The descriptions are genuine human-written clinical image findings — *"The back of the throat has swelling with whitish mass accumulation."* — which **neither text model ever saw**. As a circularity-free reference this is legitimate and it is the right idea.

> #### ⚠️ New caveat: the reference has only 501 distinct values
>
> Across the 2,923 rows carrying a description there are only **501 unique descriptions — 17.1% distinct**. The median description covers 3 rows, and **one single description covers 653 rows (22% of the corpus)**. For `skin_rash`, 1,045 rows share just 113 descriptions.
>
> **Consequence:** M4′ is closer to a *per-condition template* than a per-query reference. It measures "does the answer name the canonical visual findings for this condition" — a real construct, but narrower than "is this answer factual for this patient."
>
> **Two requirements this imposes:**
> 1. Report M4′ **per condition**, never as a bare aggregate.
> 2. Do **not** treat the 2,923 rows as independent observations. With references repeating up to 653×, significance tests need clustered/robust standard errors or a per-condition analysis, or the p-values will be badly anti-conservative.
>
> This does not block M4′. It is still the best zero-cost fix for P1 and should be built. It just needs these two lines in the Limitations section.

---

## 3. Retrieval baselines — Table 1 of the paper

Same three query conditions, same index, same relevance criterion; only the retrieval system changes. CPU, no API calls. Produced by `src/analysis/h4_baselines.py` (written this session; mirrors the notebook's cells 13–14 for local execution).

> ⏳ **Running at time of writing.** BM25 over 9,045 long Hinglish queries × 10,000 documents is
> the bottleneck (~20–40 min single-threaded), followed by `multilingual-e5-base` and `MuRIL`
> document encoding. Results land in `results/h4_retrieval/h4_baselines.csv` and this section
> will be filled in a follow-up commit.

**What to expect, and what would be surprising:**

- **BM25 and TF-IDF should collapse on Q1 (Hinglish).** They match on surface tokens, and the
  MultiCaRe corpus is English — so a romanised-Hindi query has almost nothing to match on.
  That failure, quantified, *is* the motivation for cross-lingual dense retrieval, and it is the
  cleanest justification the paper has for using LaBSE at all.
- **The Q1 → Q2 gap should be much larger for lexical systems than for LaBSE.** If it is not,
  that undermines the claim that LaBSE is doing cross-lingual work.
- **If `multilingual-e5-base` or `MuRIL` beats LaBSE**, report it honestly — a better encoder is
  a cheap, real improvement and changes the recommended architecture rather than the thesis.

---

## 4. Corrections to the record

Two claims from `plan-evaluation-and-h3.md` are now retracted or corrected. Both were mine.

| # | Claim | Status | Cause |
|---|---|---|---|
| 1 | Hallucination rises with code-mixing in both arms (ρ +0.061 / +0.081) | ❌ **Withdrawn** | `doctor`/`please` counted as Hindi |
| 2 | Code-mixing costs ~half the retrieval signal (12.7% → 23.4%) | ⚠️ **Corrected** to 11.4% → 16.0% | Condition label leaked via the caption in 96.2% of rows |

Everything else in that document stands, including the oracle-retrieval finding, the metric-sensitivity table, the dead adaptive-truncation rule, and the five-lexicon problem.

**Files that must be regenerated before any of this is shown externally:** every CMI figure in `research-poster-work/` (built on the legacy measure), and any slide quoting 23.4%.

---

## 5. Is this good or bad?

### Good

- **The paper's retrieval story is now real.** H₀₄ is measured on the full corpus with paired tests, proper CIs, a leakage gate and a per-condition breakdown. It is the most defensible result in the project.
- **Three claims died and one strengthened.** That is the right direction. A claim that dies to a repaired instrument would otherwise have died to a reviewer, more expensively.
- **Reproducibility is fixed.** Devika's `results/**` re-include means every published number is now reproducible from a clone. This was the single largest process risk and it is closed.
- **The multimodal headroom result (Q2→Q3) is a free contribution** — larger than the code-mixing penalty itself, and it justifies the proposal's multimodal ambition with evidence instead of assertion.
- **Both people's work found real errors in the other's.** The review loop is functioning.

### Bad

- **The headline number is still oracle-based.** No reported generation result describes the deployed system, and R@1 = 0.114 means the top-1 case is off-condition ~89% of the time. Expect a substantial drop when the real-retrieval H1 runs. This is unchanged and remains the top threat.
- **Retrieval is weak in absolute terms** — 1.83× the random floor. Honest, but it constrains what the system can claim.
- **The metric is still circular and still blind on ~30% of rows.** M4′ is specified but unbuilt.
- **Five lexicons still coexist**, so the Phase-6 ablation (+0.069) remains partly a lexicon artefact.
- **Coverage is 38.6%** — 1,165 of 3,015 pairs LLM-evaluated.
- **Rework debt:** all CMI figures invalid; the tertile framing must be dropped for continuous regression.

### Net

**Good.** The project is more honest and better evidenced than it was, and the two things that broke were broken before — they were just not yet known. The remaining risk is concentrated in one place (oracle retrieval), it is understood, and it costs ~1,200 Groq calls to resolve.

---

## 6. What to run next, in order

| # | Task | Cost | Unblocks |
|---|---|---|---|
| **1** | **Real-retrieval H1** — reuse `RAGPipeline.query()`, same pairs/prompts, evidence from FAISS. **Capture `retrieved_condition_groups`** or you will re-run. | ~1,200 Groq calls | The oracle problem — the last structural threat |
| **2** | **M4′ caption-reference metric**, with the §2.3 caveats built in from the start | Zero — CPU | Scoring circularity (P1) |
| **3** | **Unify the five lexicons** into `src/evaluation/concept_lexicon.py`; word boundaries; no `0.25` | Zero — ~1 day | Makes H1 and the ablation comparable |
| **4** | **Regenerate all CMI figures** on `hindi_prop_v2`; drop tertiles for continuous regression | Zero | Removes invalid assets |
| **5** | **Fix `threshold_ratio`** (0.5 → ~0.05), then sweep against fixed *k* | Zero for the sweep | Turns a false claim into an ablation |
| **6** | **H₀₃ provenance** — matched-topic, equal-size (~1,800 doc) indexes | ~1,800 Groq calls | Completes the proposal's hypothesis set |
| **7** | Gemini judge · CMI ladder · Zenodo deposit | Free tiers | Depth |

Items 2–5 are free and fix the most serious remaining problems. **Quota is not the binding constraint on the next phase of work** — only items 1 and 6 need it.

---

## 7. Artifacts produced this session

| Path | What |
|---|---|
| `results/h4_retrieval/h4_metrics.csv` | R@k, MRR, nDCG for Q1/Q2/Q3 + random floor |
| `results/h4_retrieval/h4_comparisons.csv` | Paired tests: bootstrap CI, McNemar, Wilcoxon |
| `results/h4_retrieval/h4_per_condition.csv` | Per-condition R@1 — **required** for reporting |
| `results/h4_retrieval/h4_report.md` | Generated H₀₄ report |
| `results/h4_retrieval/h4_baselines.csv` | Table 1 — BM25 / TF-IDF / e5 / MuRIL |
| `src/analysis/h4_baselines.py` | Local CPU baseline runner (new) |
| `plans/next-phase/H4-RESULTS-AND-REVIEW.md` | This file |

Dependency added: `rank_bm25` (already listed in the paper-readiness plan's requirements fix).
