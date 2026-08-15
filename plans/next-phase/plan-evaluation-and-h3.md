# Evaluation of the Paper-Readiness Plan (+ H3 options)

**Reviewed:** `paper-readiness-plan (1).md`
**Against:** branch `saachis-rag-implementation` @ `a098dac`
**Method:** every claim checked against the code and the data on disk. New numbers below were computed from the repo's own files and cached generations — **no new model calls were made.**

---

## 0. What the system actually is

Before evaluating the plan, the one-paragraph honest description:

A Hinglish patient question is embedded with LaBSE (768-d, CPU, L2-normalised), searched against a FAISS `IndexFlatIP` over 10,000 English MultiCaRe case narratives balanced across 18 condition groups, and the retrieved case is injected into a Groq `llama-3.1-8b-instant` prompt instructed to answer in Hinglish using only that evidence. A zero-shot arm answers the same question with no evidence. Both answers are scored by counting medical concepts they share with the evidence text.

**One thing must be said up front, because it changes how every result is read:** the pipeline that was *evaluated* and the pipeline that is *demonstrated* are not the same pipeline. `src/prototype/run_llm_prototype.py:221` reads a pre-computed `evidence_text` column and imports no retriever. That column was built by `run_matching.py:229–241`, which extracts each query's condition group from its `Multimodal_images/<condition>/` image path and keeps only condition-compatible candidates. **All reported H1/H2 numbers were therefore measured under oracle retrieval**, using a ground-truth label a deployed system does not have.

This is not fatal — oracle-vs-real is a legitimate ablation — but it must be disclosed and the real-retrieval number must exist.

---

## 1. Verdict on the plan, item by item

| Plan item | Verdict | Evidence |
|---|---|---|
| **P1** — metric is circular | ✅ **Correct, keep** | `src/pipeline.py:158–176` scores both arms against `combined_evidence` |
| **P2** — fragile substring matcher | ⚠️ **Real but overstated** | Bugs confirmed; repairing them does **not** move the headline. See §3 |
| **P3** — nothing reproducible | ❌ **Wrong on disk** | All artifacts present; mean pair similarity reproduces at `0.4997`. Right that a *fresh clone* fails (everything is gitignored) |
| **P4** — no baselines | ✅ **Correct, highest value** | Partly closed below at zero API cost |
| **P5** — H2 is a null presented as a finding | ⚠️ **Right that it's weak, wrong about why** | The null is an artefact of testing the wrong quantity. See §4 |
| **P6** — H3 unimplemented | ✅ **Correct** | Both corpora confirmed on disk |
| **P7** — adaptive truncation never fires | ⚠️ **Understated** | It cannot fire at *any* `top_k`. See §5 |
| **P8** — branch 9,300 lines behind | ❌ **Not for this branch** | `saachis-rag-implementation` is 19 ahead / 1 behind `main`. Applies to `devikas-development` only |
| **M4** — score against MMCQSD gold summary | ❌ **Does not work as written** | The field is a restatement of the *question* + an image caption, not a reference answer. See §2 |

---

## 2. 🔴 The plan's centrepiece does not work on this data

The plan calls this *"the single most important idea in this document"* and asks for it above everything else:

> Score both grounded and zero-shot outputs against the gold English summary — a reference neither model saw.

**MMCQSD's `summary` column is not a gold answer.** It is a condensed English restatement of the patient's *question*, followed by a caption describing the accompanying image. A real row:

> "Is it necessary to take the younger son to the doctor tonight due to persistent difficulty breathing and snoring? The image here shows a medical condition related to swollen_tonsils. The back of the throat has swelling with whitish mass accumulation."

Scoring a generated **answer** against that measures how much the answer echoes the **question**. It replaces one invalid metric with a differently invalid one — and the failure is harder to catch, because the numbers would look plausible.

**The underlying insight survives in two forms, both of which are better:**

1. **The image caption is a genuine independent reference.** It describes clinical findings derived from the image, which neither text model saw. Split it from the question clause with a regex on `"The image ..."` and it becomes a legitimate (if narrow) grounding target.
2. **The question clause is a gold English query.** This is the stronger use, and it is what makes the new H3 in §7 possible.

**Action:** do not build M4 as specified. Build (1) as a supplementary metric and (2) as an experiment.

---

## 3. 🟢 The metric is buggy — and the result survives repair

All 1,165 cached generations re-scored under three metric variants. Same outputs, different instruments, zero API cost.

| Variant | n | Zero-shot | Grounded | Gain | Cohen's *d* |
|---|---:|---:|---:|---:|---:|
| Published (28-concept lexicon) | 1,165 | 0.319 | 0.554 | +0.235 | 0.576 |
| Substring + 0.25 (18-concept) | 1,165 | 0.330 | 0.554 | +0.224 | 0.539 |
| Word-boundary + 0.25 | 1,165 | 0.278 | 0.495 | +0.217 | 0.544 |
| **Word-boundary, no default** | **662** | 0.290 | 0.592 | **+0.241** | **0.590** |

The gain stays within **+0.217 to +0.241** and the effect size **rises** as the metric gets stricter. Every variant is significant at p < 10⁻⁴⁰.

**This reframes the whole WS1 workstream.** The plan treats the metric as an existential threat that must be fixed before anything else. It is not — it is a *sensitivity analysis you can publish*. Fix it, report this table, move on.

Confirmed bugs, for the record:

- **Substring matching.** `"red"` matches inside *requi**red***, *occur**red***, ***red**uced* — firing the erythema concept on 4.8% of rows spuriously. Also `"itch"` in *st**itch***, `"mass"` in *massive*.
- **The magic `0.25`.** Fires on **27.5%** of zero-shot answers and 22.6% of grounded — it directly sets the baseline.
- **Five lexicons, not three.** `pipeline.py` (18 positive concepts), `evaluate_h1.py` (7, chest-X-ray-specific, left over from the Open-i era), `run_llm_prototype.py` (24), `run_multicare_prototype.py` (26), `run_phase6_ablation.py` (24). Consequence: the Phase-6 ablation and H1 were scored with *different instruments*, so the +0.069 structured-vs-raw result is partly a lexicon artefact.

**Two cautions.**

- **Hallucination is far more fragile than factuality.** Dropping the default moves zero-shot hallucination from 0.468 → 0.710, because the removed rows were scored 0.0 by definition. Report factuality with confidence; **re-derive every hallucination number**.
- **The real limit is coverage, not substring matching.** 43% of rows contain no lexicon concept at all in one of the two answers. That is where the metric is guessing.

---

## 4. 🟢 H2 is not a null result — it tested the wrong quantity

The current analysis tests `factual_gain` (a difference of two noisy arms) and finds nothing: Kruskal–Wallis p = 0.144. Testing the **arms separately** on the same 1,165 rows:

| Quantity | Spearman ρ vs CMI | p | Reading |
|---|---:|---:|---|
| **Grounded factual** | +0.015 | 0.612 | **flat** — no effect of code-mixing |
| **Zero-shot factual** | −0.068 | **0.021** | **declines** significantly |
| Grounded hallucination | +0.061 | **0.037** | rises slightly |
| Zero-shot hallucination | +0.081 | **0.006** | rises significantly |

Tertile means make it visible — grounded `0.554 / 0.544 / 0.563` (flat) against zero-shot `0.352 / 0.303 / 0.303` (declining).

**The finding: grounding absorbs the damage that code-mixing does.** That is a positive, directional, publishable claim, and it is a far better H2 than either the current null *or* the plan's proposed TOST equivalence test.

> ⚠️ **This kills the plan's TOST recommendation.** The plan proposes equivalence testing to convert the null into a claim. Since there *is* a real directional effect in the zero-shot arm, an equivalence test on the gain would discard the finding rather than rescue it. **Test the arms; report the interaction.**

**Honest caveat that supports the plan's CMI ladder:** the existing CMI measure has almost no spread (mean 0.425, SD 0.075, IQR 0.390–0.470). All three "tertiles" are mid-range code-mixing. The controlled ladder in WS3 is worth building — not to rescue a null, but to *extend a result that already points somewhere*.

---

## 5. 🔴 Adaptive truncation is unreachable code

The plan says the rule never fires because `app.py` passes an explicit `top_k`. The real problem is one level deeper: **even called with `top_k=None`, it cannot fire.**

The rule drops rank *i* when the gap to rank *i−1* exceeds `0.5 × top_score`. With L2-normalised LaBSE the top score averages 0.492, so it waits for a **0.246 drop between adjacent neighbours**.

| Quantity | Value |
|---|---:|
| Threshold the rule requires | **0.246** |
| Largest adjacent gap observed (299 queries × 10 ranks) | **0.088** |
| Mean adjacent gap | 0.004 |
| Queries where it fires | **0 / 299** |

Lowering `threshold_ratio` to 0.05 makes it fire on 18.4% of queries; 0.02 on 70.2%; 0.01 on 95.7%.

**The "MMed-RAG-style adaptive selection" claim in the README, `config.yaml` and the poster is currently supported by nothing.** A one-constant change plus the WS5 sweep turns a false claim into a real ablation. This is cheap and must be done.

---

## 6. 🟢 The two results nobody has measured yet

These were produced during this review, cost nothing, and are the strongest material in the project.

### 6.1 The deployed retriever is weak

299 condition-stratified queries against the live 10,000-case index, no condition filter, relevance = matching condition group:

| System | R@1 | R@3 | R@5 | R@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| **Hinglish query (deployed)** | 12.7% | 29.8% | 43.1% | 62.5% | 0.255 |
| **Gold English summary** | **23.4%** | **43.1%** | **57.2%** | **75.6%** | **0.375** |
| Random floor (prevalence-weighted) | 6.2% | 17.5% | 27.5% | 47.2% | — |

Recall@1 is only **2.0× the random floor**; at rank 10 it is **1.3×**.

### 6.2 Code-mixing costs about half the retrieval signal

Same encoder, same index, same relevance labels — only the query language changes. English **nearly doubles** Recall@1 (12.7% → 23.4%).

Because MMCQSD ships a human-written English summary for every query, this is the plan's WS2 "translate-then-retrieve" baseline with **gold** translations instead of machine ones: a cleaner control, free, and no Groq quota.

**This is the project's actual thesis, finally measurable.** The README claims *"this is not a translation problem — it is a reasoning alignment problem."* These numbers let you test that claim directly instead of asserting it.

---

## 7. Third hypothesis — options evaluated

The proposal's original H₀₃ (page 15):

> **H₀₃ (Evidence Type):** Using authoritative medical text evidence does not significantly improve the factual correctness of Hinglish explanations compared to general biomedical text.

### 7.1 Feasibility check — is H₀₃ even viable?

The decisive question is whether the alternative corpora contain the 18 MMCQSD conditions at all. Measured:

| Corpus | Docs scanned | Topically relevant | Conditions with ≥50 docs |
|---|---:|---:|---:|
| **MultiCaRe** (current) | 20,000 | **67.9%** | **17 / 18** |
| **MMedBench** (English only) | 11,451 | 16.4% | 14 / 18 |
| **PubMedQA** | 60,000 | **2.1%** | 6 / 18 |

Two consequences:

- **PubMedQA is 2% on-topic.** Grounding on the raw corpus would measure *"an off-topic corpus retrieves worse"* — trivially true and uninformative. Full PubMedQA has 273,518 rows → roughly **5,600** usable documents.
- **MMedBench is 58% Chinese** (30,826 of 53,566 rows; only 11,451 English). Using it whole confounds provenance with language. **Use the English subset only** → ~1,872 usable documents.

**Verdict: H₀₃ is viable, but only with a matched-topic design.** Filter each corpus to topically relevant documents and build **equal-sized indexes (~1,800 docs each)** — the binding constraint is MMedBench English. Same LaBSE encoder, same top-k, same prompt; only the corpus changes.

The coverage table above is itself a free, publishable result: *corpus suitability for code-mixed consumer health queries*.

### 7.2 The candidates, ranked

| # | Hypothesis | API cost | Value | Verdict |
|---|---|---|---|---|
| **A** | **H₀₃ Evidence provenance** — MultiCaRe vs PubMedQA vs MMedBench-EN vs shuffled control | ~1,800 calls | High — completes the proposal | ✅ **Do it** (matched-topic design) |
| **B** | **H₀₄ Retrieval language gap** — code-mixed vs gold-English retrieval, same encoder/index | **Zero** | **Highest** — tests the core thesis | ✅ **Do it first** |
| **C** | **Retrieval→generation coupling** — is factuality conditional on retrieval being correct? | Zero extra | High — validates the architecture | ✅ Do it (rides on the real-retrieval H1 run) |
| **D** | Encoder / lexical baselines — BM25, TF-IDF, multilingual-e5, MuRIL | Zero | Medium — a baseline table, not a hypothesis | ⚪ Do if time |
| **E** | Generator scale — 8B vs 70B | ~600 calls | Low-medium — an ablation | ⚪ Cut first |

### 7.3 Recommended H₀₄ (the new one)

> **H₀₄ (Retrieval-Stage Code-Mixing Penalty):** There is no significant difference in retrieval quality between Hinglish code-mixed queries and their semantically equivalent English renderings, when encoder, index and relevance criterion are held constant.

**Why this is the best addition:**

- **Zero API cost.** Pure CPU. Can run on all 3,015 pairs, not a 600-row subset.
- **It tests what the project claims.** The whole framing rests on code-mixing breaking clinical retrieval. Right now that is asserted, not measured.
- **The control is gold, not machine-generated.** MMCQSD's English summaries are human-written, which is stronger than the plan's Groq-translation baseline and removes translation-quality as a confound.
- **It is distinct from H₀₂.** H₀₂ is about code-mixing affecting *generation*; H₀₄ is about the *retrieval* stage. Different stage, different claim, no overlap.
- **Preliminary numbers already reject H₀₄** (12.7% vs 23.4% Recall@1), so the experiment is unlikely to produce a second null.

**Test plan:** paired comparison over the same queries — McNemar's test on Recall@1 hit/miss, Wilcoxon signed-rank on reciprocal rank, bootstrap 95% CI on the Recall@k difference. Report per-condition-group breakdown.

**Together, H₀₄ and the re-analysed H₀₂ form one coherent story:** *code-mixing breaks retrieval; grounding is what lets generation survive it.*

---

## 8. What is too complex or unnecessary

| Plan item | Call | Reason |
|---|---|---|
| **WS0.2 — regenerate MultiCaRe corpus** (hours of runtime, Day 1 blocker) | ❌ **Skip** | Already on disk and verified. The whole 3-day WS0 blocker mostly evaporates on this branch |
| **M4 — gold-summary reference scoring** | ❌ **Rework** | Measures question-echo, not factuality (§2) |
| **TOST equivalence testing** | ❌ **Drop** | Would discard the real directional effect found in §4 |
| **N=600 frozen eval set for everything** | ⚠️ **Narrow it** | A downgrade for the zero-cost retrieval work. Freeze 600 only for Groq-consuming generation experiments; run retrieval on all 3,015 |
| **Re-running generations to fix the metric** | ❌ **Never** | Outputs are cached in `results/combined_h1h2/combined_scored.csv` with joinable `pair_id`. Every metric can be computed retrospectively — that is how §3 was produced |
| **M2 — NLI + IndicTrans2 back-translation** | ⚠️ **Defer** | Two model pipelines on Kaggle, and NLI is out-of-distribution on Romanized Hindi. High effort, and no longer load-bearing once §6 lands |
| **M3 — Gemini LLM-judge** | ✅ **Keep, but later** | Genuinely good (different model family kills self-preference bias). Not a blocker |
| **Full 3,015-pair generation run** | ⚪ **Cut if behind** | 1,165 is statistically ample |
| **Generator-scale ablation (70B)** | ⚪ **Cut first** | Nice-to-have |
| **QLoRA / DPO / multimodal** | ❌ **Already out of scope** | Plan is right |

---

## 9. Recommended order of work

| # | Task | Cost | Fixes |
|---|---|---|---|
| **1** | Run H1 end-to-end **through the real retriever** on the same pairs | ~1,200 Groq calls | The oracle-retrieval problem (§0). Without it the headline is not defensible |
| **2** | Formalise **H₀₄** — retrieval eval on all 3,015 pairs, Hinglish vs gold English, + BM25/TF-IDF baselines | **Zero API** | §6, P4 |
| **3** | Re-run H2 **on the two arms separately**; per-arm regression on CMI | **Zero API**, minutes | §4, P5 |
| **4** | Unify the 5 lexicons; word boundaries; drop the `0.25`; report coverage | Zero API, ~1 day | §3, P2 |
| **5** | Fix `threshold_ratio`, then sweep it against fixed *k* | Zero API for the sweep | §5, P7 |
| **6** | README + `config.yaml` + `requirements.txt` to match reality | ~2 hours | P3 (see §10) |
| **7** | **H₀₃ provenance** — matched-topic indexes, 4 corpora | ~1,800 Groq calls | P6 |
| **8** | Controlled CMI ladder; Gemini judge | Gemini + CPU | WS1/WS3 |
| **9** | Zenodo deposit + BH-FDR correction + bootstrap CIs + figures | Low | WS0.4, WS6 |

**Notice how much of the high-value work costs no API quota.** Items 2–6 are all free and fix the most serious problems. The plan's premise that Groq quota is the binding constraint is wrong for the first half of the work.

---

## 10. Housekeeping the plan is right about

- **`README.md` and `config/config.yaml` describe a system that does not exist**: LLaVA-1.5, BioMedCLIP, QLoRA, DPO, `mmfcm`/BLEU/ROUGE metrics, `h2: two_way_anova`. None implemented.
- **`requirements.txt` omits `groq`**, which is imported at module load in three files, while pinning `peft`, `trl`, `bitsandbytes`, `open_clip_torch`, `wandb`, `pydicom` for code that does not exist, plus an unused `ollama`. A fresh clone cannot run the demo.
- **Dead code**: `app/streamlit_app.py` (70-line stub the README points at), `src/evaluation/metrics.py` (5 × `NotImplementedError`), `src/encoding/image_encoder.py`, `src/generation/trainer.py`.
- **Archiving.** Data on disk is 4.6 GB total (`raw` 3.7 G, `processed` 807 M, `faiss_index` 101 M, `embeddings` 34 M). The reproducibility-critical deposit is only ~**380 MB**: `mmcqsd_multicare_paired.csv` (10.7 MB), `multicare_filtered.csv` (269 MB), `data/faiss_index/` (101 MB), plus `results/`. Everything in `data/raw` is publicly downloadable and should be *cited*, not re-hosted. Comfortably inside Zenodo's free tier.

---

## 11. Recommended repositioning

The plan proposes selling the paper as *"how do you measure grounding for code-mixed clinical text."* The evidence points somewhere better supported and more interesting:

> **Code-mixing breaks retrieval, and grounding is what makes generation survive it.**

§6.2 and §4 are two halves of one claim and both are already demonstrable. Retrieval under Hinglish recovers roughly half the signal the same encoder recovers from a gold English rendering of the same question — while grounded generation quality is statistically flat across code-mixing intensity. The measurement work then becomes the paper's **method** section, where it belongs, instead of having to carry the contribution alone.

This framing is also more robust to the oracle-retrieval disclosure. A paper headlined *"grounding improves factuality by 73%"* must defend a number measured under conditions the system cannot reproduce. A paper headlined on the **retrieval gap** has its central claim measured on the deployed path, with the oracle condition reported as the ceiling.

---

## 12. Risks and cons — read before committing

1. **Item 1 (real-retrieval H1) may substantially lower the headline number.** Given Recall@1 = 12.7%, grounded factuality under real retrieval will be meaningfully worse than 0.554. **This is the honest number and reporting the drop is a contribution** — but plan for it emotionally and in the abstract. It is also the single strongest argument for the §11 repositioning.
2. **Relevance labels are a proxy.** "Same condition group" is coarse — 18 groups, and a same-group case can still be clinically irrelevant. It is defensible and free, but state it as a limitation.
3. **My retrieval numbers are from n = 299, one sample, one seed.** Directionally solid; re-run on all 3,015 before they go in a table.
4. **No clinician validation anywhere.** Unavoidable, already a stated limitation. The 100-sample self-annotation is worth doing — and should be pointed at the **43% of rows where the lexicon fires no concept**, which is where every automatic metric is currently guessing.
5. **H₀₃ needs the matched-topic design or it is uninformative.** Do not build the PubMedQA index from the raw corpus (§7.1).
6. **Groq free-tier variability.** Item 1 and item 7 together are ~3,000 calls. Checkpointing already exists in the runners — preserve it.
7. **Two people, ~15 working days.** Items 1–6 are the defensible-paper core. Items 7–9 are what turn a defensible paper into a strong one. If forced to choose, ship 1–6 completely rather than all nine partially.

---

## 13. What is genuinely strong here

Worth stating, because an evaluation reads as a list of problems.

- **The pairing work is a real contribution** and the plan is right to name it: 11 usable pairs under TF-IDF against Open-i, versus 3,015 at 100% MMCQSD coverage under LaBSE with condition-aware filtering. The failure and the fix together make a genuine methods finding — and the honest reading (that the condition filter was doing more work than anyone realised) makes it *more* interesting, not less.
- **The engineering is sound where it counts:** resume-from-checkpoint on every long runner, cached embeddings, a working Streamlit demo, and a statistics pipeline that correctly tests normality before choosing between paired-*t* and Wilcoxon.
- **The problems in this document are problems of experimental design and claim calibration, not of implementation quality.** That is the good kind of problem to have three weeks out — design problems are fixable at a desk; implementation problems are not.
