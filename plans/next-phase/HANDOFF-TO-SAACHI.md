# Handoff to Saachi — Critical Review, What's Broken, and What to Run

**From:** Devika's machine, 2026-08-16 · branch `devikas-updates`
**Read order:** this file → `IMPLEMENTATION-REPORT.md` → `unified-research-plan.md`
**All local processes have been stopped.** Nothing is running here.

---

## 0. Why this is being handed over

Devika's laptop (4 cores / 12 GB) encodes LaBSE at **0.38–0.46 texts/s**. It is memory-bound, so larger batches are *slower*, not faster. H₀₄ needs 9,045 encodings ≈ **6 hours**, and it was killed after confirming the rate. Everything GPU-bound or quota-bound now needs to run elsewhere.

**Everything that could be done for free on CPU has been done.** What remains is listed in §3.

---

## 1. 🔴 Critical review — what is genuinely not working

### 1.1 The oracle problem is still unfixed — this is the top threat

Every headline number still comes from evidence selected with a **ground-truth condition label**. No evaluation script calls a retriever; they read a precomputed `evidence_text` column built by `run_matching.py`, which parses the condition from `Multimodal_images/<condition>/` and uses it **twice** — as a corpus pre-filter (`:69-71`) *and* a candidate filter (`:234-242`).

**The evaluated pipeline and the demonstrated pipeline are different pipelines.** Until the real-retrieval H1 runs, no number describes the deployed system. Expect the headline to drop: preliminary R@1 ≈ 12.7% means the top-1 case is off-condition ~87% of the time.

### 1.2 🔴 Circularity is NOT fixed by fixing the oracle — and this is under-appreciated

Real retrieval changes **which** evidence is selected. It does **not** change the scoring target. The grounded arm is still scored against the very text it was conditioned on, while zero-shot is scored against text it never saw.

**Fixing retrieval does not fix P1.** These are two separate defects and only one has a plan.

### 1.3 🟢 But there IS a fix for circularity, and it costs nothing — verified this session

Devika's original M4 (score against MMCQSD's `english_summary`) was correctly rejected: that field is a restated *question* plus an image caption.

**However, the caption itself is a legitimate independent reference,** and the numbers support using it:

| Property | Measured |
|---|---:|
| Rows with a caption | 3,015 / 3,015 (**100%**) |
| Rows with description **beyond** the boilerplate clause | 2,988 (**99.1%**) |
| Mean residual description length | 73 chars |
| Descriptions containing ≥1 scoreable concept | 2,731 (**90.6%**) |
| Mean concepts per description | 1.51 (vs 3.89 in `evidence_text`) |

Examples after stripping `"The image (here) shows ..."`:
- *"The back of the throat has swelling with whitish mass accumulation."*
- *"There is swelling in the knee and red bruises."*
- *"Round and swollen spots on the skin, red in color and multiple in the affected area."*

These are human-written clinical descriptions of the image. **Neither text model ever saw them.** Scoring both arms against this description is a genuine reference-based metric with no circularity.

**Recommendation:** implement `M4′ = concept-F1 against the caption description`. It is narrower than evidence-based scoring (1.51 vs 3.89 concepts), so report it as the *unbiased* metric alongside the evidence-based one as the *generous* metric. **This is the single highest-value thing left that needs no GPU and no API quota.**

### 1.4 🟠 The relevance label is coarse, and the index/query distributions do not match

Relevance = `condition_group` match across only 18 groups, so a same-group case can still be clinically irrelevant.

Worse, the index is condition-**balanced** while the queries are not:

| Condition | Share of queries | Share of index |
|---|---:|---:|
| `skin_rash` | **34.8%** | 7.2% |
| `neck_swelling` | 9.2% | 5.9% |
| `mouth_ulcers` | 6.5% | 6.0% |

Prevalence-weighted random R@1 = **0.0626**. Because one condition is a third of all queries, the aggregate R@1 is dominated by it. **Never report aggregate R@1 without the per-condition table** — the notebook produces it.

### 1.5 🟠 The metric is blind on ~30% of rows

Using the repaired word-boundary lexicon, outputs containing **no scoreable concept at all**:

- `zero_shot_output`: **29.8%**
- `grounded_output`: **27.9%**

On those rows the metric is not measuring — it is guessing (previously via the hard-coded `0.25`). Any human-validation effort should target exactly these rows.

### 1.6 🟠 Five lexicons still coexist (Tier 1.4 not done)

`src/pipeline.py` (18 concepts), `evaluate_h1.py` (8, chest-X-ray-specific), `run_llm_prototype.py` (28), `run_phase6_ablation.py` (28), `run_multicare_prototype.py` (35). **All use naive substring matching.** Consequence: H1 and the Phase-6 ablation were scored with *different instruments*, so the +0.069 structured-vs-raw result is partly a lexicon artefact. `src/evaluation/metrics.py` is entirely `NotImplementedError` stubs.

### 1.7 🟠 Coverage is 38.6%

Only **1,165 of 3,015** pairs have been LLM-evaluated.

### 1.8 🟠 Rework created by the CMI repair

The corrected measure has a different scale *and* a different rank order (ρ = −0.53 vs legacy). **Every CMI figure in `research-poster-work/` is invalid** and must be regenerated on `hindi_prop_v2`. The **tertile framing must be dropped** — zero-shot factual is significant continuously (p = 0.0003) but not across tertiles (p = 0.127); bucketing costs power.

---

## 2. ✅ What actually works and is ready to write up

| Result | Status | Where |
|---|---|---|
| **H₀₂ per-arm, corrected CMI** | ✅ Solid, 3-policy sensitivity, BH-FDR | `results/h2_per_arm/` |
| **Integrity/provenance audit** | ✅ No published number is fabricated | `results/multicare_h1/INVALID-DO-NOT-USE.md` |
| **Batch-effect check** | ✅ 4 runs exchangeable (p = 0.951) | §1.3 of implementation report |
| **Metric sensitivity (your 3 variants)** | ✅ Effect survives repair | your `plan-evaluation-and-h3.md` §3 |
| **Pairing contribution (11 → 3,015)** | ✅ Genuine methods finding | `plans/limitation/` |

**The defensible H₀₂ claim now:** *increasing Hindi content significantly degrades zero-shot factual support (ρ = −0.116, BH p = 0.0003) while grounded factual support is unchanged (ρ = −0.0006); hallucination is unaffected in both arms.*

⚠️ Your original hallucination finding **did not survive** the lexicon repair — both arms go flat. Those effects were driven by `doctor` (71% of queries) and `please` (38%) being counted as Hindi. Please sanity-check this yourself; it is the one place my work removed a result of yours.

---

## 3. What to run, in priority order

### 3.1 🥇 H₀₄ on Kaggle GPU — free, minutes, headline experiment

**Notebook:** `notebooks/h4_retrieval_kaggle.ipynb`

1. Create a Kaggle Dataset with `mmcqsd_multicare_paired.csv`, `evidence_metadata.csv`, `evidence.index`.
2. New Notebook → upload the `.ipynb` → attach the dataset → **Accelerator: GPU** → Run All.
3. Download `/kaggle/working/h4_results/` → commit to `results/h4_retrieval/`.

Also produces the **retrieval baselines** in the same session (BM25, TF-IDF, multilingual-e5, MuRIL) — the paper's Table 1.

> ⚠️ **The one thing to check before trusting the output:** your preliminary "gold English R@1 = 23.4%" almost certainly used the **full** summary, whose caption contains the literal `condition_group` label — i.e. the relevance answer key. That number is inflated. The notebook splits it into Q2 (caption stripped) and Q3 (caption kept) and hard-fails if any label leaks into Q2. **Q1 vs Q2 is H₀₄; Q3 is a ceiling, not an English baseline.**

### 3.2 🥈 Real-retrieval H1 — ~1,200 Groq calls, fixes §1.1

Not yet written. Reuse `RAGPipeline.query()` in `src/pipeline.py` — do **not** write a new retriever. Same pairs, same prompts; evidence from FAISS instead of the `evidence_text` column. Report oracle-vs-real as a designed contrast (ceiling vs deployed), not as a correction.

Capture `retrieved_condition_groups` in the output — it makes the retrieval→generation coupling analysis free. Omit it and you will re-run.

### 3.3 🥉 Caption-reference metric (M4′) — free, CPU, fixes §1.2

Highest value-per-hour item that needs nothing but a laptop. Implement per §1.3, re-score the cached 1,165, report alongside evidence-based scores.

### 3.4 Then
4. **Unify the 5 lexicons** (§1.6) into `src/evaluation/concept_lexicon.py`, word boundaries, no `0.25`.
5. **Regenerate all CMI figures** on `hindi_prop_v2`; drop tertiles for continuous regression.
6. **H₀₃ provenance** — matched-topic, equal-sized (~1,800 doc) indexes. PubMedQA is only 2.1% on-topic and MMedBench is 57.6% Chinese (use its `language` column).
7. Gemini LLM-judge · CMI ladder · Zenodo deposit.

---

## 4. Files to share / review

### Written this session (all committed)

| File | What it is |
|---|---|
| `plans/next-phase/HANDOFF-TO-SAACHI.md` | **This file** |
| `plans/next-phase/IMPLEMENTATION-REPORT.md` | What changed, what it cost, what it broke |
| `plans/next-phase/unified-research-plan.md` | The merged plan (yours + Devika's) |
| `notebooks/h4_retrieval_kaggle.ipynb` | **Run this first** |
| `src/analysis/cmi.py` | Das & Gambäck CMI + `hindi_proportion` + repaired lexicon |
| `src/analysis/h2_per_arm.py` | Per-arm H2, bootstrap CIs, BH-FDR |
| `src/analysis/h2_recompute.py` | H2 under corrected CMI, 3-policy sensitivity |
| `src/analysis/h4_retrieval.py` | CPU version (same logic as the notebook; `--limit N`) |
| `results/h2_per_arm/*` | Reports + stats |
| `results/multicare_h1/INVALID-DO-NOT-USE.md` | Quarantine notice |

### Data you must supply to Kaggle (gitignored, not in the repo)

`mmcqsd_multicare_paired.csv` · `evidence_metadata.csv` · `evidence.index`

### 🔴 Please review these three specifically

1. **The hallucination retraction** (§2) — I removed a finding of yours. Verify you agree.
2. **The Q3 caption confound** (§3.1) — it invalidates the 23.4% figure as an English baseline.
3. **The caption-as-reference proposal** (§1.3) — this is a new idea and it changes the metric story.

---

## 5. Repo state

Branch `devikas-updates`, **5 commits ahead of origin, nothing pushed.**

```
47fbc10  H04: pin max_seq_length, cache embeddings, --limit; implementation report
859e32f  Plan: mark Tier 0 resolved; correct the H04 leakage-gate spec
441087b  Tier 1: repair CMI, audit provenance, quarantine degenerate result
a302d18  Tier 0-2: recover artifacts, version results, re-analyse H2 per arm
cc1c4c0  Merge branch 'main' into devikas-development
```

`.gitignore` now **versions `results/**/*.csv` and `*.md`** — this required `results/**`, not `results/`, because git cannot re-include a file whose parent directory is excluded. That single line is why no reported number was reproducible from a clone before.

Two bugs were also fixed in this session's own H₀₄ code, both caught before any result was used: `max_seq_length` was unset (queries encoded at 256 against documents at 128 — incomparable vectors), and the leakage gate as originally specified would have false-failed 79.6% of rows because patients legitimately name their own symptoms.
