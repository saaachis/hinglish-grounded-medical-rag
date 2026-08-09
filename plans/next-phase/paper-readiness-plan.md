# Taking the Hinglish Grounded Medical RAG Project to Paper-Ready

**Prepared for:** Devika Jonjale & Saachi Shinde
**Target venue:** ICCSDI 2026 (SN Computer Science template)
**Technical window:** ~2–3 weeks before writing begins
**Constraints assumed:** laptop CPU + free Colab/Kaggle GPU; no clinician annotation
**Budget: ₹0. Every tool, model, dataset and service in this plan is free — audited in Part 4.**

---

## Part 1 — Context: Where the Project Actually Stands

### 1.1 What you have built

A text-grounded Retrieval-Augmented Generation pipeline for code-switched (Hinglish) clinical queries:

| Stage | Implementation |
|---|---|
| Query encoding | `sentence-transformers/LaBSE`, 768-d, CPU, normalized (`src/encoding/text_encoder.py`) |
| Index | FAISS `IndexFlatIP` (exact cosine) over ~10,000 MultiCaRe case narratives, balanced across 18 condition groups (`build_index.py`, `src/retrieval/indexer.py`) |
| Retrieval | Top-k + "adaptive truncation" — first-big-gap rule, `threshold_ratio=0.5` (`src/retrieval/retriever.py`) |
| Generation | `llama-3.1-8b-instant` via Groq API, grounded vs. zero-shot system prompts, `temperature=0.3`, `max_tokens=300` (`src/generation/generator.py`) |
| Scoring | Lexical concept-overlap over 18 medical concept categories with a negation guard (`src/pipeline.py`) |
| Statistics | Shapiro–Wilk → Wilcoxon/paired-t, Cohen's d, 95% CI, Kruskal–Wallis, Mann–Whitney U + Bonferroni, Spearman ρ (`src/prototype/run_h1h2_analysis.py`) |
| Demo | Streamlit app with 6 example queries, evidence cards, side-by-side grounded/zero-shot metric tiles (`app.py`) |

**Data pairing achievement (the "limitation resolution"):** the original Open-i × MMCQSD pairing via TF-IDF produced only **11 usable pairs** (~96% of queries had zero topical overlap). Switching to MultiCaRe (93k raw → 61,316 filtered cases) and TF-IDF → LaBSE+FAISS with condition-group filtering produced **3,015 pairs at 100% MMCQSD coverage** — a 274× increase, mean similarity 0.500, 51.3% above 0.50. This is a genuine, defensible engineering contribution and should be a named contribution in the paper.

### 1.2 Current headline results (n = 1,165 paired queries)

**H1 — grounded vs. zero-shot**

| Metric | Zero-shot | Grounded | Δ | Test |
|---|---:|---:|---:|---|
| Factual support | 0.319 | 0.554 | **+0.235 (+73.5%)** | Wilcoxon p = 3.09×10⁻⁶⁴, d = 0.576, CI [0.211, 0.258] |
| Hallucination | 0.500 | 0.280 | **−0.220 (−44.0%)** | Wilcoxon p = 5.33×10⁻⁵¹, d = 0.492 |

**H2 — across CMI tertiles** (n = 385 / 384 / 396)

| CMI level | Mean CMI | Factual gain | Halluc. reduction |
|---|---:|---:|---:|
| Low | 0.351 | +0.202 | +0.206 |
| Medium | 0.428 | +0.241 | +0.208 |
| High | 0.493 | +0.260 | +0.245 |

Kruskal–Wallis H = 3.879, **p = 0.144 (n.s.)**; Spearman ρ = 0.070, p = 0.016. Currently framed as "robustness."

**Phase-6 ablation** (n = 401, raw vs. LLM-structured evidence): factual 0.571 → 0.639, hallucination 0.240 → 0.196, d 0.555 → 0.677.

**H3 — never run.**

### 1.3 The problems a reviewer will find (ranked by severity)

These are not nitpicks. Each one is a plausible reason for rejection, and each one is fixable in the time you have.

**🔴 P1 — The evaluation metric is circular.** Both grounded and zero-shot answers are scored against *the same retrieved evidence string*. The grounded model was conditioned on that exact text; the zero-shot model never saw it. A large part of the +73.5% gain is therefore an artifact of the metric definition, not a measured improvement in factuality. Your own documentation acknowledges this. A reviewer will not accept it as the headline result.

**🔴 P2 — The metric is a fragile substring matcher.** `"red"` matches inside `"reduced"`; `"itch"` matches inside `"stitch"`; `"mass"` matches inside `"massive"`. When no concept fires, the score defaults to a hard-coded `0.25` — an arbitrary constant that directly sets the zero-shot baseline. And **three divergent lexicons coexist** across `src/pipeline.py` (18 concepts), `run_llm_prototype.py` (28 concepts) and the old `evaluate_h1.py` (8 concepts). There is exactly one metric, it is unvalidated, and it is buggy.

**🔴 P3 — Nothing is reproducible.** `multicare_filtered.csv`, `mmcqsd_multicare_paired.csv`, `data/faiss_index/` and the entire `results/` directory are gitignored and **absent from disk**. Every number in §1.2 is attested only by committed markdown. `build_index.py` and `app.py` both fail immediately on a fresh checkout. `requirements.txt` omits `groq` and `python-dotenv`, which are imported at module load.

**🟠 P4 — No baselines.** There is no BM25/TF-IDF retrieval baseline, no alternative encoder, no retrieval-quality metric (Recall@k, MRR, nDCG), and no random-evidence control on the generation side. "Our method beats no method" is not an experimental comparison.

**🟠 P5 — H2 is a null result presented as a finding.** "We failed to reject H₀₂" is not the same as "grounding is robust to code-mixing." As written, this is the weakest section of the paper.

**🟠 P6 — H3 is unimplemented,** despite being in the proposal, and despite `pubmedqa_all_raw.csv` (487 MB) and `mmedbench_all_raw.csv` (65 MB) already sitting on your disk.

**🟡 P7 — Adaptive truncation never actually fires** in the demo, because `app.py` always passes an explicit `top_k`. The "MMed-RAG-style adaptive selection" claim is currently unsupported by any experiment.

**🟡 P8 — Only 1,165 of 3,015 pairs evaluated;** the branch you work on (`devikas-development`) is ~9,300 lines behind `main`; `README.md` and `config/config.yaml` still describe LLaVA-1.5, QLoRA, DPO and BioMedCLIP, none of which exist.

---

## Part 2 — The Strategy

> **Reposition the paper.** Do not sell this as "RAG improves factuality" — that is a known result and your evidence for it is currently circular. Sell it as: **"How do you *measure* grounding for code-mixed clinical text, and what does rigorous measurement reveal?"** The measurement problem is genuinely open, nobody has solved it for Hinglish, and it is the one contribution you can fully deliver in three weeks on a CPU.

### The three contributions we will build

**C1 — A validated multi-metric factuality suite for code-mixed clinical generation.**
Four independent metrics, cross-validated against each other by agreement analysis. This converts P1 and P2 from fatal weaknesses into the paper's central contribution.

**C2 — A controlled code-mixing robustness study (upgraded H2).**
Stop measuring naturally-occurring CMI and *manipulate* it: generate the same query at five controlled code-mixing levels and measure the dose–response curve for both retrieval and generation. A designed experiment beats an observational null result.

**C3 — Evidence-provenance comparison (H3), completing the proposal's hypothesis set.**
Clinical case narratives vs. general biomedical abstracts vs. exam text vs. random control — same queries, four sources.

Supporting all three: **retrieval and generation baselines** (P4), **full reproducibility** (P3), and **the ablations that justify your architectural claims** (P7).

### The single most important idea in this plan

**MMCQSD ships gold English summaries for every Hinglish query.** You are not using them.

This breaks the circularity in P1 completely. Score both grounded and zero-shot outputs against the *gold English summary* — a reference neither model saw — instead of against the retrieved evidence the grounded model was conditioned on. Any gain measured this way is a real gain. If you implement one thing from this document, implement this.

---

## Part 3 — Workstreams

### WS0 — Foundation & Reproducibility (Days 1–3) 🔴 BLOCKING

Nothing else can run until this is done.

| Task | Detail |
|---|---|
| **0.1 Branch consolidation** | Merge `main` into `devikas-development` (or branch fresh from `main`). You are currently 9,300 lines behind the real system. Do this first. |
| **0.2 Regenerate MultiCaRe corpus** | Run `src/data/download_multicare.py` → `multicare_filtered.csv` (61,316 cases). Depends on the `multiversity` package; budget several hours of runtime. **Start this on Day 1 and let it run in the background.** |
| **0.3 Rebuild pairs & index** | `run_matching.py` → `mmcqsd_multicare_paired.csv` (3,015 pairs); `build_index.py` → `data/faiss_index/`. Verify: 3,015 rows, 100% MMCQSD coverage, mean sim ≈ 0.500. If these do not reproduce, everything downstream is at risk — flag immediately. |
| **0.4 Archive artifacts properly** | Push `multicare_filtered.csv`, the pair file, the FAISS index and every results CSV to a **Zenodo deposit or HF Dataset** and cite the DOI in the paper's Data Availability statement. The template requires this section. Keep them out of git (they are large), but they must exist somewhere citable. |
| **0.5 Fix `requirements.txt`** | Add `groq`, `python-dotenv`, `rank_bm25`, `krippendorff`. Remove unused heavy deps (`peft`, `trl`, `bitsandbytes`, `open_clip_torch`, `wandb`, `pydicom`) or move to `requirements-future.txt`. Pin versions — §4.3 of the template demands library versions. |
| **0.6 Set the frozen evaluation set** | Sample **N = 600** query–evidence pairs, stratified by the 18 condition groups and by CMI tertile, `random_seed=42`, saved as `data/eval_set_600.csv`. **Every experiment in this plan uses this same set.** This is what makes results comparable across experiments and keeps you inside the token budget. |
| **0.7 Delete or quarantine dead code** | `app/streamlit_app.py` (the stub the README points at), `src/evaluation/metrics.py` (all `NotImplementedError`), `src/encoding/image_encoder.py`, `src/generation/trainer.py`. Fix `README.md` and `config/config.yaml` to describe the system that actually exists. A reviewer who opens your repo and finds LLaVA-1.5 and QLoRA advertised but absent will not trust the paper. |

---

### WS1 — The Metric Suite (Days 3–8) 🔴 HIGHEST VALUE

This is contribution C1 and it fixes P1 + P2. Build all four metrics as a single module, `src/evaluation/factuality_suite.py`, with one shared interface so the analysis scripts can swap between them.

#### M1 — Repaired concept overlap (keeps continuity with your existing results)
- Replace substring matching with **word-boundary regex** (`\brash(es)?\b`) — fixes `red`/`reduced`, `itch`/`stitch`, `mass`/`massive`.
- **Unify the three lexicons into one** canonical file, `src/evaluation/concept_lexicon.py`. Ship it as a supplementary artifact — a curated Hinglish–English clinical concept lexicon is itself a small citable contribution.
- **Remove the magic `0.25`.** When no concept fires, return `NaN` and report *concept-coverage rate* as a separate diagnostic. Then re-run the old analysis with and without it and report how much the headline number moves — this becomes a sensitivity analysis, which is exactly what §6 of the template asks for.
- Extend negation handling to more Hinglish negators (`nahi`, `na`, `bilkul nahi`, `koi nahi`).

#### M2 — NLI entailment factuality (Colab GPU, free)
Split each generated answer into sentences; for each sentence, run a multilingual NLI model against the evidence passage. Model: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
- Factuality = fraction of sentences labelled *entailment*; Hallucination = fraction *contradiction* or *neutral*.
- ⚠️ Romanized Hindi is out-of-distribution for this model. **Mitigation:** run the NLI check on an English back-translation of the generated answer. Preferred free route: **`ai4bharat/IndicTrans2` locally on Colab/Kaggle GPU** — zero API quota consumed, fully reproducible, and citable. Fallback: Gemini free tier. Report the translation step honestly as a limitation.
- Model weights, GPU, and inference are all free; the full 600-sample set fits in one Colab session (or one Kaggle session — **Kaggle gives 30 GPU-hours/week, which is more generous and less likely to disconnect than free Colab; prefer Kaggle for the long runs**).

#### M3 — LLM-as-judge (Google AI Studio free tier — free, and methodologically better)
⚠️ **Do not use Groq/Llama as the judge.** Groq's free tier for `llama-3.3-70b-versatile` is far tighter than for `8b-instant` (~100K tokens/day, not 500K), and judging Llama's output with Llama introduces **self-preference bias** that a reviewer will flag.

**Use Google AI Studio's free tier instead** (`gemini-2.0-flash`): free, no card required, ~1,500 requests/day per account, and a *different model family* from your generator — which removes the self-evaluation bias and is a genuine methodological improvement, not just a cost workaround. Three accounts ≈ 4,500 free judge calls/day, far more than you need.

Structured rubric:
1. Decompose the generated answer into atomic clinical claims.
2. Label each claim: `SUPPORTED` / `CONTRADICTED` / `NOT_MENTIONED` w.r.t. the reference text.
3. Return strict JSON.
- Factuality = SUPPORTED / total claims; Hallucination = (CONTRADICTED + NOT_MENTIONED) / total.
- **Run the judge blind** — randomize which of the two answers is "A" vs "B" and never tell it which system produced which. Log the mapping. This is a real methodological safeguard and worth one sentence in the paper.
- ~5,000 judge calls total → roughly **one day** of one free Gemini account's quota.

#### M4 — Reference-based scoring against the MMCQSD gold summary ⭐
The circularity fix. For every query, MMCQSD provides a gold English summary. Score against it:
- LaBSE cosine similarity (answer ↔ gold summary),
- BERTScore-F1,
- M1 concept-F1 against the gold summary.

Both systems are now scored against text **neither of them saw.** Report this as the primary H1 result and the evidence-based scores as secondary. Expect the effect size to shrink — that is the honest number, and reporting the shrinkage *is* the scientific contribution.

#### Metric validation (the part that makes this a contribution, not just plumbing)
- Pairwise **Spearman ρ** and **Krippendorff's α** across M1–M4 on all 600 samples.
- Report a metric-agreement matrix as a paper table.
- **Disagreement analysis:** hand-inspect 30 cases where the metrics diverge most and characterize *why*. One qualitative table of representative failure cases is worth a great deal to reviewers.

> 💡 **Strong recommendation despite your "no human annotation" answer:** you three (Devika, Saachi, Manjiri) independently annotating **just 100 samples** against a written rubric, then reporting Krippendorff's α among yourselves *and* correlation between your labels and each automatic metric, would upgrade the paper from "we have four proxies that agree with each other" to "we have four proxies validated against human judgment." It costs roughly one afternoon. If you can only add one thing beyond this plan, add this. Non-clinician annotation is a stated limitation, not a disqualifier — many published papers do exactly this.

---

### WS2 — Baselines (Days 5–10) 🟠

Fixes P4. This produces the paper's main comparison table (Table 1 in the template).

#### Retrieval baselines
Relevance labels come free: **a retrieved case is relevant if its `condition_group` matches the query's.** That gives you real IR metrics with no annotation.

| System | Notes |
|---|---|
| BM25 | `rank_bm25`, lexical baseline. Should fail badly on Hinglish — that failure *is* your motivation, quantified |
| TF-IDF | Your original approach — quantifies the 274× improvement claim properly |
| **LaBSE + FAISS** | Your system |
| `intfloat/multilingual-e5-base` | Strong modern multilingual retriever; may beat LaBSE — report it honestly either way |
| `google/muril-base-cased` | Indian-language-specialized; the most interesting comparison for your framing |
| Random retrieval | Floor |

Metrics: **Recall@{1,3,5,10}, MRR@10, nDCG@10, mean similarity.** All CPU-feasible on 600 queries; encoder swaps run on Colab.

#### Generation baselines
| Condition | Purpose |
|---|---|
| Zero-shot | Existing |
| **Random-evidence grounding** | ⭐ Critical control. Grounds on a *random* case. If factuality barely drops, your gain comes from prompt framing ("stick to evidence") rather than retrieval. You must know this answer before a reviewer asks |
| Grounded (LaBSE, top-3) | Your system |
| **Oracle evidence** | Ground on the MMCQSD gold summary — upper bound on what perfect retrieval could achieve |
| Translate-then-retrieve | Hinglish → English via Groq, then retrieve in English. Directly tests your core "this is not a translation problem" thesis. **High rhetorical value** |

---

### WS3 — Controlled CMI Robustness (Days 8–13) 🟠

Contribution C2. Fixes P5.

1. **Fix the CMI measure first.** The current one is a ratio over a hand-written ~100-word Hindi list. Replace with the standard **Das & Gambäck CMI** formulation, using a Romanized-Hindi word list (`ai4bharat` resources) or a lightweight Roman-Hindi/English token classifier. Report correlation between old and new CMI so your prior results remain interpretable.
2. **Build a controlled CMI ladder.** Take 200 English clinical queries (from MMCQSD gold summaries) and generate **five versions each** at target CMI ≈ 0.0 / 0.2 / 0.4 / 0.6 / 0.8, via dictionary-based lexical substitution using a curated English↔Roman-Hindi medical term map. Validate a sample by hand for naturalness.
3. **Run the ladder** through retrieval (Recall@k per level) and generation (all four metrics per level), grounded and zero-shot.
4. **Analyze as a dose–response curve:** Friedman test across the five levels (repeated measures on the same underlying query), post-hoc Wilcoxon with Benjamini–Hochberg, and a regression of factuality on CMI with the slope and its CI reported.
5. **⭐ Use TOST equivalence testing for the robustness claim.** This is how you convert a null result into a positive finding. Instead of "we failed to reject H₀₂", you get "grounding benefit at high CMI is statistically equivalent to that at low CMI within a margin of ±0.05 (p < 0.05)." That is a *demonstrated* robustness claim, and it is a substantially stronger sentence than anything currently in your H2 section.

---

### WS4 — H3: Evidence Provenance (Days 10–14) 🟠

Contribution C3. Fixes P6. Both datasets are already on your disk.

Four retrieval corpora, same 600 queries, grounded generation only:

| Source | Corpus | Character |
|---|---|---|
| **A. Clinical case narratives** | MultiCaRe (current) | Authoritative, case-level |
| **B. General biomedical** | PubMedQA contexts (`data/data/raw/pubmedqa/`) | Research abstracts |
| **C. Medical exam text** | MMedBench (`data/data/raw/mmedbench/`) | Didactic/textbook |
| **D. Random control** | Shuffled MultiCaRe | Floor |

Build a separate FAISS index per source (same LaBSE encoder, same top-k, same prompt — only the corpus changes). Analysis: **Friedman test** across the four conditions, post-hoc Wilcoxon signed-rank with Bonferroni, effect sizes throughout.

This directly answers the proposal's H₀₃ and is a clean, self-contained results section. It is also the cheapest new experiment relative to its value: three extra index builds and 1,800 generations.

---

### WS5 — Ablations & Full-Scale Run (Days 12–16) 🟡

| Ablation | Design | Fixes |
|---|---|---|
| **Adaptive vs. fixed truncation** | Sweep `threshold_ratio ∈ {0.3, 0.5, 0.7}` vs. fixed k ∈ {1,3,5,10}. Report **how often adaptive truncation actually fires** and its effect | P7 — this claim is currently unsupported |
| **Top-k sensitivity** | k ∈ {1,3,5,10} on the 600-set | Justifies the default |
| **Evidence position** | Best-match first vs. last vs. shuffled | Lost-in-the-middle effect; cheap and interesting |
| **Generator scale** | `llama-3.1-8b-instant` (Groq) vs. `llama-3.3-70b` — get the 70B **free from Cerebras Cloud's free tier** (~1M tokens/day, no card) rather than burning Groq's tight 70B quota | Does grounding help small models *more*? An excellent finding if true — it is the deployment argument for low-resource settings |
| **Structured vs. raw evidence** | Already done (n=401) — re-run on the 600-set with the new metrics | Consistency |
| **Full-coverage H1** | If tokens allow, extend H1 to all **3,015 pairs** with the fixed metrics | P8 |

---

### WS6 — Statistical Rigor & Figures (Days 14–17) 🟡

- **Family-wise correction across the whole paper.** You will run 40+ tests. Apply **Benjamini–Hochberg FDR** across the full family and report both raw and adjusted p-values. Reviewers check for this.
- **Bootstrap 95% CIs** (10,000 resamples) for every reported mean and difference — more defensible than the current analytic CIs for bounded, non-normal scores.
- **Per-condition-group breakdown** across all 18 groups: where does grounding help most/least? A heatmap here is high-value and you already have the plotting infrastructure.
- **Retrieval-quality → generation-quality correlation:** does higher retrieval similarity actually predict higher factuality? If yes, that validates the whole architecture in one scatter plot. If no, that is an important negative finding worth reporting.
- **Reuse `research-poster-work/generate_plots.py`** (928 lines, already written) — adapt it rather than starting over. Regenerate everything at **300+ DPI, vector PDF preferred**, per the template's figure requirements.

---

## Part 4 — Cost Audit: Everything Here Is Free

### 4.1 Full inventory

| Component | What we use | Cost | Notes |
|---|---|---|---|
| **Generation (main)** | Groq `llama-3.1-8b-instant` | **Free** | 500K tokens/day/key, no card. Already your setup |
| **Generation (70B ablation)** | **Cerebras Cloud free tier**, `llama-3.3-70b` | **Free** | ~1M tokens/day, no card. Avoids Groq's tight 70B quota |
| **LLM-judge (M3)** | **Google AI Studio**, `gemini-2.0-flash` | **Free** | ~1,500 req/day/account, no card. Different model family = no self-preference bias |
| **Overflow / backup** | **OpenRouter `:free` models**, Together AI free tier | **Free** | Safety net if a quota runs dry mid-run |
| **Encoders** | LaBSE, `multilingual-e5-base`, MuRIL | **Free** | Open weights, run on CPU/Colab |
| **NLI (M2)** | `mDeBERTa-v3-base-xnli-multilingual` | **Free** | Open weights |
| **Translation (M2)** | `ai4bharat/IndicTrans2` | **Free** | Open weights, runs locally — no API quota at all |
| **BERTScore (M4)** | `bert-score` + open HF model | **Free** | pip |
| **Retrieval / stats libs** | `faiss-cpu`, `rank_bm25`, `scipy`, `statsmodels` (TOST), `krippendorff`, `pingouin` | **Free** | pip, open source |
| **GPU** | **Kaggle (30 GPU-hrs/week)** primary; free Colab secondary | **Free** | Prefer Kaggle — longer sessions, fewer disconnects |
| **Datasets** | MMCQSD, MultiCaRe, PubMedQA, MMedBench | **Free** | Open access; PubMedQA + MMedBench already on your disk |
| **Data archiving** | **Zenodo** (DOI, 50 GB) or HF Datasets | **Free** | Required for the template's Data Availability section |
| **Code archiving** | GitHub tagged release | **Free** | — |

**Nothing in this plan requires a payment method.** Every quota is a no-card free tier.

### 4.2 Free-quota budget

Three people × three providers, all free:

| Workload | Provider | Calls | Daily free capacity | Days |
|---|---|---:|---|---:|
| Reuse existing 1,165 H1 results | — | **0** | — | 0 |
| Baselines: random / oracle / translate-first (600×3) | Groq (3 keys) | 1,800 | ~1,000 gens/day | ~2 |
| H3: three evidence sources (600×3) | Groq (3 keys) | 1,800 | ~1,000 gens/day | ~2 |
| CMI ladder (200×5×2) | Groq (3 keys) | 2,000 | ~1,000 gens/day | ~2 |
| Ablations: top-k, position | Groq (3 keys) | 1,200 | ~1,000 gens/day | ~1.5 |
| Ablation: 70B generator | **Cerebras** (separate quota) | 600 | ~600/day | ~1 |
| LLM-judge M3 | **Gemini** (separate quota) | ~5,000 | ~4,500/day | ~1.5 |
| Back-translation M2 | **IndicTrans2 local** | ~2,000 | unlimited | 0 |

**Groq-bound work: ~7–8 days.** Because the judge, the translation and the 70B run all moved off Groq onto *separate* free quotas, they run **in parallel** with the Groq generation queue rather than competing with it. That is the difference between this fitting in three weeks and not.

Discipline that makes it work:
- ✅ **Start Groq generation runs on Day 3**, the moment WS0 finishes. Keep the queue busy every single day — an idle day is a permanently lost 1.5M tokens.
- ✅ **Resume-from-checkpoint on every runner** (your existing scripts already do this — preserve it).
- ✅ **Cache aggressively.** M1, M2 and M4 re-score cached outputs at zero API cost. Only M3 spends quota.
- ✅ **Never re-run the existing 1,165 H1 generations** — re-score them with the new metrics.
- ⚠️ **If you fall behind:** cut the 70B ablation and the full 3,015-pair run first. Never cut WS1 (metrics) or WS4 (H3).

---

## Part 5 — Parallel Work Split

The work divides into two tracks of **roughly equal effort**. Decide between yourselves who takes which — nothing in the plan depends on who picks what.

> **Track 1 — Data, Retrieval & Experiment Execution**
> Owns the corpus, the FAISS indexes, all retrieval-side experiments, and keeps the generation queue running. **Produces model outputs.**
>
> **Track 2 — Metrics, Scoring & Statistics**
> Owns the entire evaluation suite, the CMI measure, and all statistical analysis. **Produces the scores and tests applied to those outputs.**

**Why this split has no dependency:** Track 2 builds and unit-tests every metric against the **existing 1,165 cached results**, which already exist. It never waits for Track 1's new runs. Track 1's experiments write raw outputs and never need a score to proceed. The two only meet at the end, when finished metrics are applied to finished outputs.

**Effort balance:** Track 1 is more *runtime*-heavy (long downloads, index builds, API queues — much of it unattended background time). Track 2 is more *code*-heavy (four metrics, validation analysis, statistics). Roughly equal hands-on hours; Track 1 has more waiting, Track 2 more writing. Whoever prefers debugging pipelines takes Track 1; whoever prefers metrics and statistics takes Track 2.

### 5.1 The handoff contract (agree this on Day 1, before anything else)

Everything crosses between you through two fixed file formats. Freeze these on Day 1 and neither track can break the other.

**`results/<experiment>_raw.csv`** — Track 1 writes, Track 2 reads:
```
query_id, condition_group, query_text, gold_summary, system,
evidence_text, evidence_ids, retrieval_scores, generated_answer,
model_name, top_k, run_date, seed
```

**`results/<experiment>_scored.csv`** — Track 2 writes, both read:
```
query_id, system, m1_factual, m1_halluc, m1_coverage,
m2_entail, m2_contra, m3_supported, m3_contra, m3_notmentioned,
m4_labse_cos, m4_bertscore, m4_concept_f1
```

Rule: **Track 1 never writes a score column; Track 2 never writes an answer column.** One shared `data/eval_set_600.csv` (seed 42) anchors `query_id` across everything.

### 5.2 Day-by-day

**Days 1–3 — Foundation (the only point where the two tracks touch)**

| | Track 1 | Track 2 |
|---|---|---|
| Day 1 | **Together, ~1 hr:** merge `main` → working branch; agree the two CSV schemas above; register 3 Groq + 3 Gemini + 1 Cerebras free keys | ← same |
| Day 1 | Kick off `download_multicare.py` — **hours of runtime, start it before anything else and let it run unattended** | Fix `requirements.txt` (add `groq`, `python-dotenv`, `rank_bm25`, `krippendorff`, `statsmodels`; pin versions) |
| Day 2 | Build pairs (`run_matching.py`) → verify 3,015 rows / mean sim ≈ 0.500; build FAISS index | Quarantine dead code (`app/streamlit_app.py`, `src/evaluation/metrics.py`, `image_encoder.py`, `trainer.py`); rewrite `README.md` + `config/config.yaml` to match reality |
| Day 3 | Build `data/eval_set_600.csv` (stratified, seed 42); **launch the first Groq generation batch** | Start M1 repair — word-boundary regex, unify the 3 lexicons, remove the `0.25` default |

**Days 3–17 — Fully parallel, no blocking**

| Days | Track 1 | Track 2 |
|---|---|---|
| 3–6 | **WS2 retrieval baselines** — BM25, TF-IDF, LaBSE, `multilingual-e5`, MuRIL, random. Recall@{1,3,5,10}, MRR, nDCG using condition-group relevance labels. *No LLM calls — pure CPU/Kaggle, runs alongside the generation queue* | **WS1 M1 + M4** — repaired concept metric, unified lexicon, plus gold-summary reference scoring (LaBSE cosine, BERTScore, concept-F1). Unit tests: `"reduced"` must not match `"red"`. **Validated on the cached 1,165 results** |
| 5–8 | **WS2 generation baselines** — random-evidence, oracle-evidence, translate-then-retrieve (600×3 via Groq) | **WS1 M2 + M3** — IndicTrans2 back-translation + mDeBERTa NLI on Kaggle; Gemini judge with blind A/B randomization |
| 8–10 | **WS4 H3 indexes** — build three new FAISS indexes (PubMedQA, MMedBench, shuffled-random) and run 600×3 grounded generations | **WS1 validation** — metric agreement matrix (Spearman + Krippendorff's α), plus hand-read 30 max-disagreement cases. **This is contribution C1 — Track 2's single biggest deliverable** |
| 10–13 | **WS3 CMI ladder execution** — run the 5-level ladder through retrieval (Recall@k per level) and generation (200×5×2) | **WS3 CMI measure** — implement proper Das & Gambäck CMI; build the English↔Roman-Hindi medical substitution dictionary; hand-validate 30 generated queries for naturalness |
| 13–15 | **WS5 ablations** — adaptive-vs-fixed truncation (report *how often it actually fires*), top-k sweep, evidence position, 70B via Cerebras | **WS3/WS4 analysis** — Friedman + post-hoc for H3; dose–response regression and **TOST equivalence test** for H2 |
| 15–17 | **WS0.4 archiving** — Zenodo deposit + DOI, GitHub release tag, verify clean-checkout reproduction | **WS6 statistics** — Benjamini–Hochberg FDR across the full test family, bootstrap CIs, per-condition-group breakdown |
| 16–17 | **Figures** — adapt the existing `generate_plots.py` (928 lines, already written) to 300 DPI / vector PDF | **Results tables** — every table in template format, plus the retrieval→factuality correlation plot |
| 17 | **Together:** run the single master notebook end-to-end. Freeze results. **Begin writing.** | ← same |

**Deliverable count per track:** Track 1 → corpus + 4 indexes + 5 experiment result sets + figures + archive. Track 2 → 4 metrics + validation study + 4 statistical analyses + tables. Deliberately balanced.

### 5.3 Rules that keep you unblocked

1. **Track 2 builds every metric against the cached 1,165 results, never against Track 1's new runs.** This is the whole reason the tracks don't collide — the metric suite is fully testable from Day 3.
2. **Track 1 owns all API keys and the generation queue.** One person watching quotas prevents two people silently burning the same key.
3. **Schema changes are proposed in writing and agreed by both before anyone edits.** A mid-week column rename costs a day.
4. **15-minute standup, daily.** What landed, what's running overnight, what's blocked.
5. **Sunday checkpoint:** both re-run the master notebook. If it regenerates every table and figure cleanly from cached CSVs, you're on track. Fix that before adding anything new.
6. **A third contributor** (if Manjiri is an author): third Groq + Gemini key, plus co-owning the 100-sample self-annotation and the disagreement-case reading — both parallelizable and requiring no code.
7. **If one track finishes early,** the spare capacity goes to the 100-sample self-annotation (§WS1) or the full 3,015-pair H1 run — in that order.

---

## Part 6 — How the Results Map to the ICCSDI Template

| Template section | Content |
|---|---|
| **Abstract** | Problem → metric-validity gap → 4-metric suite + controlled CMI study + H3 → headline numbers (from M4, the honest ones) |
| **1 Introduction** | Hinglish clinical CDS; three contributions (C1/C2/C3) as the bullet list the template asks for |
| **2 Related Work** | Organize by *evaluation protocol*, not by paper — HiFACTMix, HealthAlignSumm, MedSumm, MMed-RAG, HEALTH-PARIKSHA, Ke et al. Your gap: none validates factuality metrics for code-mixed clinical text |
| **3 Materials and Methods** | 3.1 problem formalization (query, corpus, retrieve-then-generate objective); 3.2 architecture figure (reuse `08_pipeline_architecture.png`); 3.3 pseudocode for adaptive truncation + the metric suite |
| **4 Experimental Design** | 4.1 MMCQSD + MultiCaRe + the 3,015-pair construction (**include the 11-pair failure and the 274× fix — it is a genuine methodological finding, not an embarrassment**); 4.2 baselines + four metrics + validation; 4.3 library versions, hardware, seeds, Groq model IDs, prompt release |
| **5 Results** | 5.1 retrieval baselines; 5.2 H1 under all four metrics + agreement matrix; 5.3 controlled CMI (H2, with TOST); 5.4 H3 provenance; 5.5 ablations |
| **6 Discussion** | Why the effect shrinks under reference-based scoring; where metrics disagree and why; per-condition-group variation; the retrieval→generation correlation |
| **6.1 Limitations** | No clinician validation; non-clinician-authored lexicon; automated proxies only; text-only; Groq dependency; synthetic CMI ladder; NLI model out-of-distribution on Romanized Hindi |
| **7 Conclusion** | Grounding helps but *measurement design determines the apparent size of the effect* — a claim you will have actually demonstrated |
| **Declarations** | Data availability → Zenodo DOI from WS0.4. Code availability → tagged GitHub release. Ethics → "Not applicable" (public de-identified corpora) |

---

## Part 7 — Explicitly Out of Scope (and why)

| Deferred | Reason |
|---|---|
| **QLoRA / DPO fine-tuning** | Needs sustained GPU; free Colab sessions time out mid-training; and reviewers would demand baselines you cannot afford. State as future work — your proposal already does |
| **Multimodal / BioMedCLIP** | MMCQSD has images, but building a second index, a fusion strategy and multimodal metrics is a paper of its own |
| **Live clinician study** | Needs ethics approval and recruitment time you do not have |
| **Full 61,316-case index** | Current 10,000-case balanced index is defensible and fast. Optionally add an index-size ablation (1k/5k/10k) if time permits — cheap and shows scaling behaviour |

---

## Part 8 — Decisions You Two Need to Make

1. **Does the headline number become the reference-based (M4) score?** It will be smaller than +73.5%. Our strong recommendation: **yes** — report M4 as primary, evidence-based as secondary, and make the difference between them a finding. Defending an inflated number is far riskier than reporting an honest one.
2. **Will you do the 100-sample self-annotation?** (~1 afternoon; disproportionately large credibility gain.)
3. **Which do you cut if you fall behind** — generator-scale ablation, or the full 3,015-pair run? (Recommend cutting the full run; the 600-set is statistically sufficient.)
4. **Who owns the Zenodo/HF deposit** for data availability?
5. **Is Manjiri contributing a third Groq key**, and is she an author on this paper?

---

## Verification — How You Know It Worked

**After WS0 (Day 3) — hard gate, do not proceed until all pass:**
```bash
git log --oneline -1                      # should show main's history merged in
python build_index.py                     # data/faiss_index/ created
python -c "import pandas as pd; d=pd.read_csv('data/processed/mmcqsd_multicare_paired.csv'); print(len(d), d.similarity.mean())"
# expect: 3015, ~0.500
streamlit run app.py                      # loads, answers a query end-to-end
pip install -r requirements.txt           # clean install in a fresh venv, app still imports
```

**After WS1 (Day 8):**
```bash
pytest tests/test_factuality_suite.py     # new unit tests: "reduced" must NOT match "red"
python -m src.evaluation.validate_metrics # prints the M1-M4 agreement matrix
```
All four metrics produce scores for all 600 samples; Spearman ρ between metrics is reported; ≥30 disagreement cases have been read by a human.

**After WS2–WS4 (Day 14):**
Each experiment writes a versioned CSV under `results/` **and** a markdown report. Sanity checks: BM25 should underperform LaBSE on Hinglish; random-evidence grounding should score clearly below real grounding (if it does not, that is a finding — investigate before writing).

**Final gate (Day 17):**
One notebook regenerates **every** table and figure in the paper from cached CSVs, top to bottom, with no manual steps. If a number is in the paper and not produced by that notebook, it does not go in the paper.

---

## The One-Line Summary

**You have a working system with a weak measuring instrument.** Spend three weeks fixing the instrument, add the baselines and the two missing experiments, and you will have a paper whose central claim — *that measuring grounding in code-mixed clinical text is harder than it looks, and here is how to do it properly* — is both novel and fully supported by what you can actually run on a laptop.
