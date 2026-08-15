# Unified Research Plan — Hinglish Grounded Medical RAG

**For:** Devika Jonjale & Saachi Shinde
**Target:** ICCSDI 2026 (SN Computer Science template)
**Supersedes:** `paper-readiness-plan.md` (Devika) and `plan-evaluation-and-h3.md` (Saachi) — this document merges both
**Budget:** ₹0. Every tool, model and service below is a no-card free tier.

---

## Context — Why This Document Exists

Two planning documents were written independently. Saachi's evaluated Devika's against the actual code and data, and **empirically overturned several of its central recommendations.** Before merging them I re-verified every disputed claim against the repository. The verification is in §1.

Three findings change the shape of the work:

1. **Every reported H1/H2 number was measured under oracle retrieval.** No evaluation script calls a retriever. This is an integrity issue and it is the top priority.
2. **The cached generations exist on exactly one machine.** They are not in git and never have been. This is a hard blocker on roughly half the "zero-cost" work.
3. **`run_multicare_prototype.py` cannot produce a meaningful result.** Its two arms are pinned to constants by construction. Any number traced to it must be withdrawn.

The intended outcome: a four-hypothesis paper whose central claim is measured on the pipeline that actually ships, not on a privileged configuration.

---

## Part 1 — Verification: Which Plan Was Right

Every disputed point, checked against the code. **Saachi's evaluation was correct on nearly every technical dispute.** Recording this plainly because it determines what we build.

| # | Point | Devika's plan | Saachi's finding | Verified | Resolution |
|---|---|---|---|---|---|
| 1 | **Oracle retrieval** | Not identified | All results use precomputed `evidence_text` | ✅ **Saachi correct — and understated** | Top priority. §2 |
| 2 | **M4 gold-summary scoring** | "The single most important idea" | Summary is a *question restatement + image caption*, not an answer | ✅ **Saachi correct** | M4 as written is **dropped**. Salvaged into H4 (§5) |
| 3 | **Metric is buggy** | 3 lexicons, existential threat | 5 lexicons; result *survives* repair | ✅ **Saachi correct** | Demoted from blocker to sensitivity analysis |
| 4 | **Nothing reproducible** | Artifacts absent | Present on her branch | ⚠️ **Both right, different machines** | §3 — the real blocker |
| 5 | **H2 is a null** | Rescue with TOST | Null is an artefact of testing the *gain*; test arms separately | ✅ **Saachi correct** | TOST **dropped**. Per-arm analysis instead |
| 6 | **Adaptive truncation** | Never fires because `app.py` passes `top_k` | Cannot fire at *any* `top_k` | ✅ **Saachi correct — and understated** | Needs 0.245 gap; gaps are ~0.004. 61× too small |
| 7 | **Branch 9,300 behind** | Stated as fact | Applies to `devikas-development` only | ✅ **Saachi correct** | Already resolved — merged this session |
| 8 | **Regenerate MultiCaRe (Day 1–3 blocker)** | 3-day blocker | Already on disk, skip | ⚠️ **Moot** | Already regenerated: 61,316 rows verified |
| 9 | **N=600 frozen set for everything** | Universal | Downgrade for free retrieval work | ✅ **Saachi correct** | 600 for Groq work only; retrieval on all 3,015 |

### 1.1 Where I extend both documents

Three things neither plan caught, found during verification:

**🔴 A. The oracle label leaks twice, not once.** Saachi identified the candidate filter. There is also a *corpus* pre-filter:

```python
# run_matching.py:69-71  — corpus is pre-filtered by ground-truth label
group_name = CONDITION_TO_GROUP.get(mmcqsd_cond, mmcqsd_cond)
available  = mc_dedup[mc_dedup["condition_group"] == group_name]

# run_matching.py:234-242 — then candidates are filtered by it again
compat_set = COMPAT_GROUPS.get(q_cond, {q_cond})
... if mc_conditions[e_idx] in compat_set
```

The label comes from parsing `Multimodal_images/<condition>/` out of each query's image path (`run_matching.py:196-198`) — information a deployed system does not have. There is a fallback at `:244-251` taking any-condition top-1 when nothing clears `MIN_SIM = 0.25`, so a minority of pairs are non-oracle; **the fraction is unquantifiable until the pairs file is recovered.**

**🔴 B. `run_multicare_prototype.py` is degenerate by construction.** Not a weak experiment — an arithmetic identity:

- **Zero-shot arm** (`:113-118`) is a *fixed template string*: *"…a specific diagnosis cannot be confirmed. Please consult a healthcare provider…"* — containing no lexicon concept, so it is pinned at factual `0.25` / hallucination `0.0`.
- **Grounded arm** (`:127-128`) is literally `"Based on the clinical report: " + evidence[:500 words]` — its concepts are a subset of the evidence by definition, pinning factual ≈ `1.0`.

Its "H1 effect" is `0.25 → 1.0`, with no model involved. **Action: audit every reported number for provenance (§4.1). Any figure traced to this script must be withdrawn from the poster and prior reports.**

**🟠 C. Saachi's proposed H₀₄ has a confound that would invalidate it.** Full detail and the corrected three-condition design in §5 — this is the most important design change in this document.

---

## Part 2 — The Oracle Problem, and Why It Becomes an Asset

### What happened
`run_llm_prototype.py` imports no retriever. Line 221 reads `evidence = str(row["evidence_text"])` — a column built offline using the ground-truth condition label. `run_phase6_ablation.py:189` and `run_multicare_prototype.py:191` do the same. `EvidenceRetriever` is referenced only by `src/pipeline.py` (the demo) and the poster docs. **The pipeline that was evaluated and the pipeline that is demonstrated are different pipelines.**

### Why this is not fatal
Oracle-vs-real is a legitimate, standard RAG ablation. The problem is not that it was measured — it is that it was reported *as if* it were the deployed number. Reframed as a designed contrast, it becomes one of the paper's most useful results:

| Condition | Evidence selection | Reports |
|---|---|---|
| **Oracle retrieval** | Condition-filtered (current results) | Ceiling — grounding benefit under perfect retrieval |
| **Real retrieval** | Unfiltered FAISS top-*k* | The deployed number |
| **Δ** | — | **The cost of retrieval error** |

That delta connects directly to H4: retrieval is poor *because the queries are code-mixed*. Oracle-vs-real quantifies what that costs downstream. This is a contribution, not a confession.

### What it costs
Expect real-retrieval grounded factuality to fall meaningfully below `0.554`. Saachi's preliminary R@1 = 12.7% means the top-1 case is from a different condition group ~87% of the time. **Plan for the headline to drop, and put it in the abstract deliberately.** A number that survives disclosure is worth more than a larger number that does not.

---

## Part 3 — 🔴 The Blocker: Artifacts Exist on One Machine

**Verified conclusion: no data file has ever been committed to any branch in the entire repository history.** `.gitignore` excludes `*.csv`, `*.json`, `*.npy`, `*.index`, `results/`, `data/processed/`. Confirmed with `git rev-list --all | git ls-tree -r` — the result set is empty.

| Artifact | Devika | Saachi | Recoverable? |
|---|---|---|---|
| `results/combined_h1h2/combined_scored.csv` (1,165 cached generations) | ❌ | ✅ | **Only by file transfer** — regenerating costs ~2,330 Groq calls |
| `data/processed/mmcqsd_multicare_paired.csv` (3,015 pairs) | ❌ | ✅ | Rebuildable via `run_matching.py`, but **will not byte-match** |
| `data/faiss_index/` | ❌ | ✅ | Rebuildable via `build_index.py` |
| `data/processed/multicare_filtered.csv` | ✅ (61,316 rows) | ✅ | Already regenerated |
| `data/processed/mmcqsd_queries.csv` (3,015 rows) | ✅ | ✅ | Present |

> ### ✅ RESOLVED — the handoff already existed
> Saachi had already produced `plans/next-phase/handoff-tier1-essential.zip` and
> `handoff-tier2-corpora.zip`; they were sitting in the repo **unextracted**. All
> artifacts are now in place and the Tier 0 gate passes: 3,015 pairs at
> similarity **0.49966**, and 1,165 cached generations. Her regenerated corpus
> also matches Devika's **exactly** (61,316 / 39,652 / 18, byte-identical
> per-condition breakdown), which closes the cross-machine mismatch risk (§10.6).
> The remaining lesson stands: this was invisible because nothing was tracked.

### Action — do this before anything else
1. ~~**Saachi zips and uploads `results/` + `mmcqsd_multicare_paired.csv` + `data/faiss_index/`**~~ — done; extract the zips at the repo root.
2. **Both parties then rebuild from the same pairs file**, so the two machines agree.
3. **Fix the root cause immediately:** add `!results/**/*.csv` negation patterns to `.gitignore`, or better, commit result CSVs to a `results-archive/` path that is explicitly *not* ignored. Losing an experiment to a `.gitignore` rule twice would be unforgivable.

> ⚠️ Until step 1 completes, Devika cannot reproduce, re-score, or verify a single reported number. Everything in Tier 2 that is described as "zero cost" is zero cost **only for whoever holds the files.**

---

## Part 4 — The Paper: Framing and Hypotheses

### 4.0 The headline claim

> **Code-mixing breaks clinical retrieval, and evidence grounding is what allows generation to survive it.**

Adopting Saachi's repositioning. The rationale is her own sensitivity analysis (§1, row 3): the H1 conclusion *survives* metric repair — gain holds at +0.217→+0.241 and Cohen's *d* **rises** as the metric gets stricter. That result partially falsifies "rigorous measurement overturns conclusions," so the measurement story cannot carry a headline. It becomes the Methods section that makes the headline credible — which is where it belongs.

### 4.1 The causal chain — the paper's spine

Each link is separately measurable with assets already on disk:

```
Code-mixed query
      │
      ▼  H4 ── code-mixing degrades retrieval        [zero API cost]
Weak / off-condition evidence
      │
      ▼  H1(real vs oracle) ── retrieval error degrades grounding   [~1,200 calls]
Reduced but positive grounding benefit
      │
      ▼  H2 ── grounded is flat across CMI; zero-shot declines      [zero API cost]
Grounding absorbs the damage
      │
      ▼  H3 ── and what you ground on still matters                 [~1,800 calls]
```

### 4.2 Hypothesis set

| ID | Statement | Status | Cost |
|---|---|---|---|
| **H₀₁** Grounding effect | No difference in factual consistency / hallucination between zero-shot and grounded | Measured under oracle; **must re-run under real retrieval** | ~1,200 calls |
| **H₀₂** Code-mixing × grounding *(generation)* | Increasing code-mixing does not change the grounded-vs-zero-shot difference | **Re-analyse per arm** — do not test the gain | Zero |
| **H₀₃** Evidence provenance | Authoritative clinical text is no better than general biomedical text | Not run. Needs matched-topic design (§6) | ~1,800 calls |
| **H₀₄** Code-mixing penalty *(retrieval)* — **NEW** | No retrieval-quality difference between Hinglish queries and equivalent English, encoder/index/labels held constant | Preliminary evidence exists; **needs the §5 redesign** | **Zero** |

**H₀₂ is re-analysed, not re-run.** Testing `factual_gain` (a difference of two noisy arms) found nothing, p = 0.144. Testing the arms separately on the same 1,165 rows:

| Quantity | Spearman ρ vs CMI | p | Reading |
|---|---:|---:|---|
| **Grounded factual** | +0.015 | 0.612 | **Flat** — code-mixing does not hurt |
| **Zero-shot factual** | −0.068 | **0.021** | **Declines** significantly |
| Zero-shot hallucination | +0.081 | **0.006** | Rises significantly |

Tertile means: grounded `0.554 / 0.544 / 0.563` (flat) vs zero-shot `0.352 / 0.303 / 0.303` (declining). That is a positive directional finding — and it is why **TOST is dropped**: an equivalence test on the gain would discard a real effect rather than rescue a null.

---

## Part 5 — 🔴 H₀₄ Redesign: Removing the Caption Confound

**This is the most important technical correction in this document.** H₀₄ is the paper's best experiment — zero API cost, tests the core thesis, runs on all 3,015 pairs. As currently specified it would not survive review.

### The confound

Saachi proposes retrieving with MMCQSD's `english_summary_or_target` as the gold-English control. Verified structure of that column (n = 3,015, **100% of rows**):

> "What could be the cause of high fever, swollen tonsils with white spots … in an 11-year-old girl recently diagnosed with MRSA? **The image here shows a medical condition related to `swollen_tonsils`. The back of the throat has swelling with whitish mass accumulation.**"

The trailing caption **literally contains the condition-group token** — and the relevance label for retrieval evaluation *is* condition-group match. Retrieving with that string is retrieving with the answer key. Two template phrasings, both leaking: `"The image here shows"` (36.75%) and `"The image shows"` (63.25%).

**Consequence:** the preliminary R@1 = 23.4% for "gold English" is inflated. The true Hinglish-vs-English gap is unknown and is **smaller** than 12.7% → 23.4%. The claim "code-mixing costs half the retrieval signal" cannot be made from that number.

### The corrected design — three conditions, still zero cost

Split the summary on the caption boundary (handling both phrasings) and treat the halves as separate conditions. This is strictly better than the 2-condition version — it converts a bug into an extra result:

| # | Query used | Interpretation |
|---|---|---|
| **Q1** | `hinglish_query` | **Deployed path** |
| **Q2** | English question clause, caption stripped | **Translation ceiling** — perfect Hinglish→English translation |
| **Q3** | Full summary incl. caption | **Multimodal ceiling** — perfect translation *and* a perfect image reader |

- **Q1 vs Q2 = H₀₄ proper.** The clean, unconfounded code-mixing penalty.
- **Q2 vs Q3 = free bonus result.** Quantifies the headroom an image encoder would buy — an evidence-backed argument for the multimodal future work, *without doing any multimodal work.* This is worth a paragraph in Discussion and a line in Future Work.

**Tests:** McNemar on Recall@1 hit/miss (paired binary), Wilcoxon signed-rank on reciprocal rank, bootstrap 95% CI on ΔRecall@k, per-condition-group breakdown. Run on all 3,015 pairs, not a subset — it costs nothing.

**Validation gate:** after stripping, assert no Q2 string contains the substring `"The image"` or an **underscore-joined group label** (`swollen_tonsils`). If the assertion fails, the strip is incomplete and the result is invalid.

> ⚠️ **Correction to an earlier draft of this gate.** It originally said "no Q2 string contains any `condition_group` token." That is wrong and would false-fail on **79.6%** of rows: a patient question legitimately describes its own symptom ("a lump in my neck"), and those words *are* the query, not leakage. Only the templated machine-readable label is the leak. Implemented in `src/analysis/h4_retrieval.py::assert_no_leakage`.

---

## Part 6 — Workstreams, Priority-Ordered

No calendar — you'll set the deadline later. These are ordered by value-per-hour so you can cut from the bottom at any point. **Tiers 0–2 are the minimum defensible paper.**

### 🔴 Tier 0 — Unblock (hours, must be first)

| # | Task | Owner | Notes |
|---|---|---|---|
| 0.1 | **Saachi uploads `results/` + pairs CSV + FAISS index** | Saachi | §3. Blocks everything |
| 0.2 | Fix `.gitignore` so results are never lost again | Either | Negation patterns or `results-archive/` |
| 0.3 | Rebuild FAISS index from `multicare_filtered.csv` | Devika | `build_index.py`; already has the input |
| 0.4 | Agree the CSV schema contract (below) | Both | 30 min, prevents a week of pain |

### 🔴 Tier 1 — Integrity (mostly free, non-negotiable)

| # | Task | Cost | Fixes |
|---|---|---|---|
| 1.1 | **Provenance audit.** Trace every number on the poster and in `plans/` to the script that produced it. Withdraw anything from `run_multicare_prototype.py` | Free, ~3 h | §1.1-B |
| 1.2 | **Real-retrieval H1 runner.** Reuse `RAGPipeline.query()` in `src/pipeline.py` — do not write a new retriever. Same pairs, same prompts, evidence from FAISS instead of the column | ~1,200 calls | §2 |
| 1.3 | **Quantify the oracle fallback rate** — what fraction of pairs used the any-condition fallback (`run_matching.py:244-251`) | Free | Needed for honest disclosure |
| 1.4 | **Unify the 5 lexicons** into `src/evaluation/concept_lexicon.py`; word-boundary regex; drop the `0.25`; report coverage. Publish the before/after table as a sensitivity analysis | Free, ~1 day | §1 row 3 |
| 1.5 | **Fix the CMI measure.** The 129-token list contains `"doctor"`, `"please"`, `"pls"` — which appear in nearly every MMCQSD query, inflating CMI and compressing variance (SD only 0.075). Rebuild on Das & Gambäck, re-cut tertiles, report old-vs-new correlation | Free, ~half day | H₀₂ validity |
| 1.6 | **Delete or fix `run_multicare_prototype.py`** | Free | Prevents reuse of a broken script |

> ⚠️ **1.4 caution:** hallucination is far more fragile than factuality. Dropping the `0.25` moves zero-shot hallucination `0.468 → 0.710`, because removed rows scored `0.0` by definition. Report factuality with confidence; **re-derive every hallucination number from scratch.**

### 🟢 Tier 2 — The Contribution (free, highest value)

| # | Task | Cost |
|---|---|---|
| 2.1 | **H₀₄ three-condition retrieval study** (§5) on all 3,015 pairs | **Zero** |
| 2.2 | **H₀₂ per-arm re-analysis** + regression of each arm on corrected CMI | **Zero**, minutes |
| 2.3 | **Retrieval baselines:** BM25, TF-IDF, LaBSE, `multilingual-e5-base`, MuRIL, random floor. Recall@{1,3,5,10}, MRR, nDCG | **Zero** (CPU/Kaggle) |
| 2.4 | **Retrieval→generation coupling:** is factuality conditional on retrieving the right condition group? Rides free on the 1.2 output | **Zero** |
| 2.5 | **Fix `threshold_ratio`** and sweep it vs fixed *k*. Currently needs a 0.245 gap against observed gaps of ~0.004 — fires 0/299 | **Zero** for the sweep |

### 🟡 Tier 3 — Extension (quota-consuming)

| # | Task | Cost |
|---|---|---|
| 3.1 | **H₀₃ provenance, matched-topic design** (§7) | ~1,800 calls |
| 3.2 | **Gemini LLM-judge**, blind A/B randomised | Free (separate quota) |
| 3.3 | **Controlled CMI ladder** — five levels by lexical substitution | ~2,000 calls |
| 3.4 | Generation controls: random-evidence, oracle-evidence | ~1,200 calls |

### ⚪ Tier 4 — Polish

Benjamini–Hochberg FDR across the full test family · bootstrap 95% CIs (10,000 resamples) · per-condition-group heatmap · Zenodo deposit (~380 MB) · README/`config.yaml`/`requirements.txt` truth-up · figures at 300 DPI via the existing `generate_plots.py` (928 lines — adapt, don't rewrite).

### ❌ Cut list (in order)

Generator-scale 70B ablation → full 3,015-pair generation run → M2 NLI+IndicTrans2 (two model pipelines, out-of-distribution on Romanized Hindi, no longer load-bearing) → QLoRA/DPO/multimodal (already out of scope).

---

## Part 7 — H₀₃ Requires a Matched-Topic Design

Verified corpus statistics — the naïve design would fail:

| Corpus | Rows | On-topic for the 18 conditions | Usable |
|---|---:|---:|---:|
| MultiCaRe | 61,316 filtered | **67.9%** | Plenty |
| MMedBench | 53,566 (**only 11,451 English**, 57.6% Chinese) | 16.4% | **~1,872** |
| PubMedQA | 273,518 | **2.1%** | ~5,600 |

Two traps:
- **PubMedQA at 2% on-topic** would measure *"an off-topic corpus retrieves worse"* — trivially true, uninformative.
- **MMedBench is 57.6% Chinese** (there is an explicit `language` column — use it). Using it whole confounds provenance with language.

**Design:** filter each corpus to topically relevant documents, then build **equal-sized indexes (~1,800 docs each)** — the binding constraint is MMedBench-English. Same encoder, same top-*k*, same prompt; only the corpus varies. Add a shuffled-MultiCaRe control. Friedman across four conditions, post-hoc Wilcoxon + Bonferroni.

The coverage table above is itself a free publishable result: *corpus suitability for code-mixed consumer-health queries.*

---

## Part 8 — Work Split

Two tracks of comparable effort. Pick freely — nothing depends on who takes which.

> **Track 1 — Retrieval & Experiment Execution.** Indexes, all retrieval experiments (H₀₄, baselines, truncation sweep), the real-retrieval H1 runner, the generation queue. *Produces model outputs.*
>
> **Track 2 — Metrics, Statistics & Provenance.** Lexicon unification, CMI repair, the provenance audit, per-arm H₀₂, all statistical analysis, the LLM-judge. *Produces scores and tests.*

**Why these don't collide:** Track 2 works on cached generations; Track 1 produces new ones. Neither waits on the other after Tier 0. **Track 2 is the natural owner for whoever holds the `results/` files today** — that is Saachi, so the lowest-friction assignment is Saachi → Track 2, Devika → Track 1. Devika already has `multicare_filtered.csv` and can rebuild the index independently.

**Schema contract — freeze on day one:**

```
results/<experiment>_raw.csv     ← Track 1 writes, Track 2 reads
  query_id, condition_group, query_text, query_variant(Q1|Q2|Q3),
  retrieval_mode(oracle|real), evidence_text, evidence_ids,
  retrieval_scores, retrieved_condition_groups, system,
  generated_answer, model_name, top_k, run_date, seed

results/<experiment>_scored.csv  ← Track 2 writes, both read
  query_id, system, m1_factual, m1_halluc, m1_coverage,
  m3_supported, m3_contra, m3_notmentioned, cmi_v2
```

Rule: **Track 1 never writes a score column; Track 2 never writes an answer column.** `retrieved_condition_groups` is what makes 2.4 free — capture it from the first run or you will re-run for it.

Standing rules: daily 15-minute standup · schema changes proposed in writing and agreed by both · weekly checkpoint where both re-run the master notebook end-to-end · if one track finishes early, spare capacity goes to the 100-sample self-annotation, targeted at the **43% of rows where the lexicon fires no concept** — that is where every automatic metric is currently guessing.

---

## Part 9 — Mapping to the ICCSDI Template

| Section | Content |
|---|---|
| **Abstract** | Code-mixing degrades clinical retrieval; grounding absorbs it. Lead with the **real-retrieval** numbers |
| **1 Introduction** | Contributions: (C1) the code-mixing retrieval penalty, isolated from translation and image cues; (C2) grounding as a stabiliser under code-mixing; (C3) matched-topic evidence-provenance comparison; (C4) the 3,015-pair cross-lingual resource |
| **2 Related Work** | By evaluation protocol — HiFACTMix, HealthAlignSumm, MedSumm, MMed-RAG, HEALTH-PARIKSHA. Gap: none isolates the *retrieval-stage* penalty for code-mixed clinical queries |
| **3 Methods** | Pipeline; the Q1/Q2/Q3 query construction (§5); the unified concept lexicon; corrected CMI |
| **4 Experimental Design** | 4.1 MMCQSD + MultiCaRe + the 11→3,015 pairing story (a genuine methods finding); **4.2 must state the oracle-vs-real distinction explicitly**; 4.3 versions, seeds, hardware, prompts |
| **5 Results** | 5.1 H₀₄ retrieval penalty + baselines · 5.2 H₀₁ oracle vs real · 5.3 H₀₂ per-arm interaction · 5.4 H₀₃ provenance · 5.5 ablations |
| **6 Discussion** | Q2→Q3 gap as the multimodal argument; retrieval→generation coupling; where the lexicon fires nothing |
| **6.1 Limitations** | Oracle contrast; condition-group relevance is a coarse proxy; no clinician validation; lexicon authored by non-clinicians; single generator; synthetic CMI ladder |
| **7 Conclusion** | Grounding does not remove the code-mixing problem — it relocates it to the retrieval stage, where it is measurable and fixable |
| **Declarations** | Data availability → Zenodo DOI (~380 MB). Code → tagged release. Ethics → not applicable |

---

## Part 10 — Risks

1. **The headline will drop.** Real-retrieval grounded factuality will land well below 0.554. This is the honest number; the §4.0 framing is what makes it a finding rather than a loss.
2. **Numbers may need public withdrawal.** If any poster figure traces to `run_multicare_prototype.py`, it must go. Better found now than by a reviewer.
3. **Relevance labels are a proxy.** Condition-group match across 18 coarse groups; a same-group case can still be clinically irrelevant. Defensible, free, must be stated.
4. **Preliminary retrieval numbers are n = 299, one seed.** Re-run on all 3,015 before they enter a table.
5. **The Q2 strip must be validated** or H₀₄ is invalid — see the §5 assertion gate.
6. **`multicare_filtered.csv` was regenerated this session** and has not been diffed against Saachi's copy. If the two differ, indexes and pairs will not match across machines. **Compare row counts and a checksum during Tier 0.**
7. **Two people, no clinician.** Tiers 0–2 are the defensible core. Ship those completely rather than all four tiers partially.

---

## Verification

**Tier 0 gate — do not proceed until all pass:**
```bash
python -c "import pandas as pd; d=pd.read_csv('data/processed/mmcqsd_multicare_paired.csv'); print(len(d), d.similarity.mean())"
# expect: 3015, ~0.4997  — and the same on BOTH machines
python build_index.py                    # data/faiss_index/ created
python -c "import pandas as pd; print(len(pd.read_csv('results/combined_h1h2/combined_scored.csv')))"
# expect: 1165
```

**Tier 1 gate:**
```bash
pytest tests/test_concept_lexicon.py     # "reduced" must NOT match "red"; "massive" must NOT match "mass"
python -m src.evaluation.audit_provenance  # every reported number → its source script
```
Real-retrieval H1 produces a factual-support number for the same pairs, and the oracle-vs-real delta is reported with a bootstrap CI.

**Tier 2 gate:**
```bash
python -m src.retrieval.eval_h4 --assert-no-leakage
```
Must hard-fail if any Q2 string contains a `condition_group` token or `"The image"`. Q1/Q2/Q3 Recall@{1,3,5,10} + MRR reported on all 3,015 with McNemar and bootstrap CIs.

**Final gate:** one notebook regenerates every table and figure from cached CSVs, end to end, no manual steps. **If a number is in the paper and not produced by that notebook, it does not go in the paper.**

---

## The One-Line Summary

**The system works; the evaluation measured a privileged version of it.** Fix the measurement to match the deployed pipeline, isolate the code-mixing penalty at the retrieval stage where it actually lives, and the result is a paper whose central claim — *code-mixing breaks retrieval, and grounding is what lets generation survive it* — is measured on the system you can actually ship.
