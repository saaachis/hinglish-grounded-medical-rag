# Hardening Report — `saachi-hardening`

**Date:** 2026-08-24 · branch `saachi-hardening` (off `main`)
**Scope:** the five items blocking the paper, from `H4-RESULTS-AND-REVIEW.md` §6
**API spend:** 804 Groq calls (one day's free quota). Everything else is CPU.

---

## 0. Scoreboard

| # | Item | Status | Outcome |
|---|---|---|---|
| 1 | Real-retrieval H1 | ✅ **Done** (n=268) | Grounding survives real retrieval — but see §5 |
| 2 | M4′ caption metric | ✅ **Done** | Effect shrinks **6×** under unbiased scoring |
| 3 | Unify five lexicons | ✅ **Done** | One module, 21 regression tests |
| 4 | CMI figures | ✅ **Done** | Regenerated continuous, tertiles dropped |
| 5 | Fix `threshold_ratio` | ✅ **Done** | Clean **negative result** — 0 of 6 wins |
| — | Retrieval baselines | ⏳ Running | BM25/TF-IDF in; encoders pending |

**Bottom line:** the oracle-retrieval problem turned out to be *far less damaging* than
feared — and four independent results now converge on a different, more interesting
explanation of what this system actually does. See §5.

---

## 1. Real-retrieval H1 — the oracle problem, closed

`src/analysis/h1_real_retrieval.py` · `results/h1_real_retrieval/`

Three arms, same 268 pairs, same prompts, one generator: zero-shot, **oracle** evidence
(condition-filtered, the old ceiling) and **real** evidence (FAISS top-1, no label).

| Contrast | n paired | zero-shot | grounded | Δ | Cohen's *d* | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| Oracle evidence (ceiling) | 161 | 0.3142 | 0.5045 | **+0.1902** | 0.500 | 7.2×10⁻⁹ |
| **Real retrieval (deployed)** | 158 | 0.2720 | 0.4557 | **+0.1837** | 0.492 | 1.1×10⁻⁸ |

**Oracle − real grounded factuality: +0.0755, p = 0.106 — not significant.**

This is the good news, and it is genuinely surprising. The oracle-retrieval problem was
called "the top structural threat" and predicted to cost a large chunk of the headline.
It costs almost nothing measurable. **The grounding benefit does not depend on the
condition filter.**

### ⚠️ Two blockers found while running it

**The original generator no longer exists.** `llama-3.1-8b-instant` returns 404 on every
key — Groq decommissioned it, and no Llama chat model remains on this account. Two of the
six keys in `.env` are also revoked (401).

This is a reproducibility problem independent of everything else: §4.3 of the ICCSDI
template requires model IDs, and a reviewer attempting to reproduce will hit a 404. It
must be stated in Limitations.

Because the old model is gone, reusing cached outputs would confound *retrieval change*
with *model change*, so all three arms were regenerated on `openai/gpt-oss-20b`. Two
alternatives were tested and rejected: `qwen/qwen3.6-27b` emits `<think>` traces, and
`openai/gpt-oss-120b` spends its whole token budget on reasoning. **This converts the
problem into a result: H1 now replicates on a second generator family** — a model-transfer
robustness check the paper did not previously have.

**Throughput.** Rotating API keys only on error left the spares idle, because the SDK
absorbs 429s internally and the error never surfaced. 47% of calls were rate-limited and
throughput was 0.77 rows/min. Round-robining every call across the live keys raised it to
5–10 rows/min. The run stopped cleanly at row 268 on daily quota exhaustion; it resumes
from checkpoint.

---

## 2. The refusal finding — report this beside every factuality number

| Arm | evidence | **refusal rate** | scoreable (has concepts) |
|---|---|---:|---:|
| zero-shot | none | **0.0%** | 90.7% |
| oracle | condition-filtered | **82.8%** | 62.7% |
| real | FAISS top-1 | **84.0%** | 61.9% |

McNemar oracle vs real: n01=31, n10=28, **p = 0.79** — the refusal rate is the same
whether evidence is oracle-filtered or not.

**The grounded model declines rather than confabulates.** Handed another patient's case
report, it says *"evidence mein koi jaankari nahi hai"* instead of inventing a link. That
is correct, safe behaviour — and it is invisible to the concept metric, because a refusal
asserts no clinical concept and therefore scores `nan`.

> **This is a measurement trap.** A naive mean over non-`nan` rows makes the system look
> *healthiest exactly where it fails*, because the failures are dropped. Any factuality
> number for a grounded arm must be reported with its refusal rate and its coverage.

---

## 3. M4′ — the circularity fix, and the paper's headline

`src/evaluation/caption_reference.py` · `results/m4_caption/`

Both arms scored against the MMCQSD image description — human-written, derived from the
image, seen by neither model — after stripping both the question clause and the
boilerplate clause that names the `condition_group` label.

| Metric | Reference | Δ | 95% cluster CI | Cohen's *d* |
|---|---|---:|---|---:|
| Evidence-based (circular) | retrieved evidence | +0.2749 | [+0.2471, +0.3035] | 0.678 |
| **M4′ caption (unbiased)** | image description | **+0.0462** | [+0.0152, +0.0646] | **0.181** |

**The effect shrinks roughly 6× under unbiased measurement, and survives.** Cohen's *d*
falls from medium (0.678) to small (0.181).

The CIs resample **clusters, not rows**: the reference has only 412 unique strings across
2,988 rows and one covers 22.3% of the corpus, so row-level bootstrapping would have
overstated significance badly. `score_frame` refuses to return a bare aggregate without
its per-condition table.

M4′ is narrower than evidence-based scoring (~1.5 vs ~3.9 concepts per reference) — it
covers visible findings only. Report the pair: **unbiased** beside **generous**. The
contrast is the contribution, not either number alone.

---

## 4. Truncation and the lexicon

### Adaptive truncation — a clean negative result

`src/analysis/truncation_sweep.py` · `results/truncation_sweep/` · n = 3,015

The shipped rule needs a 0.248 similarity drop between adjacent neighbours; the largest
gap that occurs anywhere in the data is 0.109. It fired on **0 of 3,015** queries.

Swept against fixed *k* at matched budget, **adaptive truncation wins 0 of 6 settings**.
Precision is flat (~0.113–0.115) across *every* setting while recall falls monotonically
as fewer cases are kept — meaning **the similarity gap carries no relevance information**.

> **Recommendation:** drop the "MMed-RAG-style adaptive context selection" claim from
> `README.md`, `config/config.yaml` and the poster, and report the negative result. That a
> published heuristic does not transfer to this setting is worth more than the claim was.

The default is now 0.05 and configurable, so the sweep is reproducible.

### Lexicon unified

`src/evaluation/concept_lexicon.py` — one module replacing five copies (18/7/24/26/24
positive concepts), 21 regression tests, every one of them a real bug from the old code:
`red` firing inside *requi**red***/*occur**red***/***red**uced*, `itch` inside *st**itch***,
`mass` inside *massive*. The hard-coded `0.25` — which fired on 27.5% of zero-shot answers
and therefore *set the baseline* — is replaced by `nan` plus an explicit coverage
diagnostic.

One subtlety: `-algia` had to be declared a **suffix** pattern. A naive `\balgia\b` would
never match *neuralgia*/*myalgia*, turning the bug fix into a silent regression.

The five old copies are retained verbatim so published numbers stay reproducible, each now
carrying a banner naming its own defects.

### CMI figures

`src/analysis/h2_figures.py` · `results/h2_figures/` — 300-DPI PNG + vector PDF,
regenerated on `hindi_prop_v2` with continuous binned means. Tertiles dropped: bucketing
manufactured the old null (p = 7.7×10⁻⁵ continuous vs 0.127 across tertiles).

A defect in the first version was caught and fixed: a global OLS line rose visibly in the
grounded panel while Spearman ρ = −0.001, because 95% of the mass sits at x = 0.5–0.9 and
the line extrapolated through empty space. Replaced with quantile-binned means and
bootstrap CIs, which now agree with the statistic.

Supersedes `04_h2_cmi_levels.png`, `06_cmi_scatter.png`, `12_h2_grounded_factual_by_cmi.png`.

---

## 5. 🔴 The finding that reframes the paper

Four independent results, none of which was designed to test this, all point the same way.

| Evidence | Result |
|---|---|
| Oracle vs real retrieval | Δ = +0.0755, **p = 0.106 (n.s.)** — the condition filter buys nothing |
| Retrieval correctness → factuality | **p = 0.53** — being topically correct does **not** predict a better answer |
| Unbiased vs circular metric | Effect shrinks **6×** (d 0.678 → 0.181) |
| Lexical vs dense retrieval | BM25 **0.1343** vs LaBSE **0.1144** on Hinglish — the lexical baseline *wins* |

Grounded factuality by whether retrieval was topically correct:

| Retrieval top-1 | n | grounded factual | refusal rate |
|---|---:|---:|---:|
| wrong | 230 | 0.4471 | 85.2% |
| correct | 38 | 0.5079 | 76.3% |

Mann-Whitney U = 1644, **p = 0.53**.

### What this means

**The measured grounding benefit is substantially an echo effect, not a retrieval effect.**

The grounded arm is told *"base your response strictly on the evidence"*. It then either
declines (84% of the time) or restates concepts from whatever text it was handed. Because
the evidence-based metric scores the answer *against that same text*, restating anything
scores well — whether or not the evidence was relevant to the patient.

Every one of the four results above is what you would predict if that were true: relevance
would not matter (it doesn't, p=0.53); the oracle filter would not matter (it doesn't,
p=0.106); an unbiased reference would collapse the effect (it does, 6×); and a lexical
retriever would do about as well as a cross-lingual one (it does).

### This is a better paper, not a worse one

The claim "RAG improves factuality by 73%" was never going to survive review. What is
defensible, and considerably more interesting:

> **Standard evidence-based factuality metrics substantially overstate the benefit of
> retrieval-augmented generation for code-mixed clinical queries, because they reward
> echoing the supplied evidence rather than answering correctly. Under a reference the
> model never saw, the benefit is real but roughly six times smaller — and it does not
> depend on the retrieved evidence being topically correct.**

That claim is fully supported by data now in the repo, it is novel, and it generalises
beyond Hinglish. The safety finding — that a well-behaved grounded model *refuses* 84% of
the time rather than confabulating — is a genuine positive result and belongs in the
abstract.

---

## 6. What is still open

| # | Item | Cost | Why |
|---|---|---|---|
| 1 | **Finish H1 to n≈400** | Resumes on tomorrow's quota | 268 is ample (p<10⁻⁸); more only narrows CIs |
| 2 | **Encoder baselines** (e5, MuRIL) | Running | Completes Table 1 |
| 3 | **Investigate the echo effect directly** | ~600 calls | ⭐ Add a **random-evidence arm**. If factuality holds with deliberately irrelevant evidence, §5 is proven outright. This is now the single most valuable experiment left |
| 4 | **H₀₃ provenance** | ~1,800 calls | Matched-topic, equal-size indexes |
| 5 | **Update README/config/poster** | ~2 h | Remove LLaVA/QLoRA/BioMedCLIP and the adaptive-selection claim |
| 6 | Gemini judge · CMI ladder · Zenodo | Free tiers | Depth |

**Item 3 is the priority.** §5 currently rests on four converging indirect results. A
random-evidence control would test it head-on and is cheap. If factuality barely drops on
random evidence, the paper's central claim is demonstrated rather than inferred.

---

## 7. Files

| Path | What |
|---|---|
| `src/evaluation/concept_lexicon.py` | Canonical lexicon (replaces 5) |
| `src/evaluation/caption_reference.py` | M4′, with cluster-aware statistics |
| `src/analysis/h1_real_retrieval.py` | 3-arm runner, key round-robin, resume |
| `src/analysis/h1_oracle_vs_real.py` | Oracle-vs-real analysis |
| `src/analysis/run_m4_rescore.py` | M4′ vs evidence-based comparison |
| `src/analysis/truncation_sweep.py` | Truncation sweep + verdict |
| `src/analysis/h2_figures.py` | Corrected CMI figures |
| `src/analysis/h4_baselines.py` | Retrieval baselines (local CPU) |
| `tests/test_concept_lexicon.py` | 21 regression tests |
| `results/h1_real_retrieval/` | H1 oracle vs real |
| `results/m4_caption/` | M4′ scores + per-condition |
| `results/truncation_sweep/` | Sweep + negative-result verdict |
| `results/h2_figures/` | 300-DPI PNG + vector PDF |
