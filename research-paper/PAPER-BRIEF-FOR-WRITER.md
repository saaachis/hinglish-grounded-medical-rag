# Paper Brief — everything needed to write the manuscript

**Purpose.** This is a self-contained briefing. A writer with only this document and
the template can produce the full paper without reading the codebase. Every number
below is measured and traceable to a committed result file.

**Venue.** ICCSDI 2026 (International Conference on Computational Science and Data
Intelligence), 11–12 December 2026, NMIMS Mumbai. Springer SN template.

**Authors.** Devika Jonjale and Saachi Shinde — equal contribution, alphabetical
order. M.Sc. Data Science, NMIMS Nilkamal School of Mathematics, Applied Statistics
& Analytics, Mumbai, India. devikajonjale04@gmail.com · saachi.shinde28@gmail.com

---

## 0. Files to supply alongside this brief

| File | Why |
|---|---|
| `ICCSDI2026_Word_Template (1).docx` | the required format — fill it, don't rebuild it |
| `results/retrieval_figures/fig1_table1.png` | Fig: retrieval by system and language |
| `results/retrieval_figures/fig2_penalty.png` | Fig: the code-mixing penalty asymmetry |
| `results/h1_figures/fig3_h1_reference_effect.png` | Fig: grounding effect by scoring reference |
| `results/h2_figures/h2_dose_response.png` | Fig: factual support vs code-mixing intensity |

A current draft exists at `research-paper/draft/Hinglish_RAG_ICCSDI2026.docx`. It is
structurally correct but the prose should be rewritten.

---

## 1. What the research is

A retrieval-augmented generation (RAG) system for **Hinglish** — romanised
Hindi–English code-switched — patient questions, grounded in **English clinical case
reports**. A patient writes *"Doctor, meri beti ko skin pe rash hai aur bahut khujli
ho rahi hai"*; a clinical report says *"erythematous maculopapular rash with
pruritus"*. The clinical content is present, the surface form is not what the system
expects.

**Pipeline.** Hinglish query → LaBSE embedding → FAISS search over 10,000 English
MultiCaRe case narratives (18 condition groups, passage-chunked into 41,746 passages)
→ retrieved case injected into a grounded prompt → LLM answers in Hinglish. A
zero-shot arm answers the same question with no evidence. Answers scored by
clinical-concept overlap.

**Framing that fits the evidence.** This is a **measurement study**, not a
system-improvement paper. Several original claims did not survive scrutiny and are
reported as corrections. The strongest results are on the *retrieval* side.

---

## 2. The four hypotheses (nulls, as pre-registered in the proposal)

| | Null hypothesis | Outcome |
|---|---|---|
| **H₀₁** | Grounding does not change factual consistency vs zero-shot | **Rejected** |
| **H₀₂** | Code-mixing intensity does not affect the grounded/ungrounded difference | **Rejected, per arm** |
| **H₀₃** | Authoritative case evidence does not improve factual correctness vs general biomedical text | **Partially answered** |
| **H₀₄** | Retrieval quality does not differ between code-mixed and equivalent English queries | **Rejected, every system** |

*Note for the writer:* "rejected" is the positive outcome — it means the effect is
real. Consider adding a plain-language gloss in the hypotheses table, e.g.
"H₀₁ — Rejected — i.e. grounding significantly improves factual support."

---

## 3. Datasets and construction

- **Queries:** MMCQSD — Hinglish patient questions, each with a human-written English
  summary and an image caption.
- **Evidence:** MultiCaRe clinical case reports. 61,316 filtered, 10,000 indexed,
  balanced across 18 condition groups. Median case = **554 words**.
- **Pairs: 3,015 at 100% query coverage.**

**A methodological finding worth reporting.** Matching Open-i radiology reports to
MMCQSD by TF-IDF yielded only **11 usable pairs** — the corpora had almost no topical
overlap. Switching to MultiCaRe with LaBSE plus condition-aware filtering produced
**3,015 pairs, a 274-fold increase**.

**A leakage gate was required.** MMCQSD's English summary contains an image caption
naming the condition group verbatim in **96.2%** of rows — i.e. it contains the
relevance label itself. Any "English vs Hinglish" comparison using the full summary is
inflated. The caption is stripped and a gate asserts no label survives.

---

## 4. Results — exact numbers

### 4.1 H₀₄ — retrieval penalty (Table 1). n = 3,015

| System | Hinglish | English | Penalty | McNemar p |
|---|---:|---:|---:|---:|
| Hybrid (RRF) | **0.1751** | 0.1973 | +0.0222 | 0.018 |
| LaBSE (passages) | 0.1280 | 0.1486 | +0.0206 | 0.017 |
| BM25 | 0.0935 | 0.1847 | **+0.0912** | 9.9×10⁻²⁶ |
| TF-IDF | 0.0842 | 0.1529 | — | — |
| *random floor* | *0.0626* | *0.0626* | — | — |

**Two findings:**
1. The penalty is significant for **every** retrieval method. Testing per system is
   deliberate — a penalty appearing under one configuration is a property of that
   configuration; one holding across methods is a property of code-mixing.
2. **Lexical retrieval is ~4.4× more damaged than dense.** BM25 is the *best* system
   on English and the *worst* on Hinglish. This crossover is the paper's empirical
   argument for cross-lingual embedding.

### 4.2 Script, not language — the MuRIL result

On the single-vector index: MuRIL (pretrained on Indian languages) scores **0.0640**
on Hinglish against a random floor of **0.0626** — chance. It recovers to **0.1821**
on the same content in English. MuRIL is trained on *Devanagari* Hindi; MMCQSD is
*romanised*. **Script mismatch, not language mismatch.** Practitioners deploying
Indian-language encoders on romanised user text should expect near-chance retrieval.

Also measured on that index: multilingual-e5-base 0.1303, BM25 0.1343, TF-IDF 0.1167,
LaBSE 0.1144.

### 4.3 Validity of the scoring instrument

Scored against the unbiased reference (n = 1,876):

| Answer | Concept precision |
|---|---:|
| copy the reference verbatim | 1.0000 |
| **constant `"swelling"`** | **0.7132** |
| constant `"swelling and erythema"` | 0.6586 |
| constant `"erythema"` | 0.6039 |
| **GROUNDED SYSTEM** | **0.1528** |
| **ZERO-SHOT SYSTEM** | **0.1066** |
| constant `"pain"` | 0.0032 |

**A hard-coded one-word answer beats the real system by 4.7×.** Concept precision has
no recall term, so its optimum is terseness. **Absolute levels carry no information
about answer quality; only paired deltas between arms scored identically are
interpretable.** Report F₁, and ship these baselines with every table.

Also: the commonly reported **"hallucination rate" is exactly 1 − precision**
(verified min = max = 1.0). Reporting both double-counts one result.

### 4.4 H₀₁ — the grounding effect

Under the evaluation protocol in common use (scoring against the retrieved evidence):

| Generator / evidence | n | zero-shot | grounded | Δ | Cohen's d | p |
|---|---:|---:|---:|---:|---:|---:|
| llama-3.1-8b, oracle (as originally published) | 1,165 | 0.3190 | 0.5535 | **+0.2345** | 0.576 | 3.1×10⁻⁶⁴ |
| llama-3.1-8b, oracle (repaired lexicon) | 669 | 0.2796 | 0.5758 | +0.2962 | 0.720 | 2.8×10⁻⁴⁹ |
| gpt-oss-120b, oracle | 223 | — | — | +0.2659 | 0.662 | 5.6×10⁻¹⁷ |
| gpt-oss-120b, real retrieval | 237 | — | — | +0.2238 | 0.616 | 6.5×10⁻¹⁷ |

**H₀₁ is rejected, and replicates across two generator families.**

**But the magnitude depends on the reference.** Same generations, concept F₁
(grounded − zero-shot), Benjamini–Hochberg corrected:

| Generator / evidence | Circular reference | Unbiased reference |
|---|---:|---:|
| llama-3.1-8b, oracle | +0.203 | **+0.062** |
| gpt-oss-120b, oracle | +0.093 | −0.021 (n.s.) |
| gpt-oss-120b, real retrieval | −0.032 (n.s.) | **−0.047** |

*Circular* = the retrieved evidence, which the grounded arm was conditioned on and the
zero-shot arm never saw. *Unbiased* = the MMCQSD image description, which **neither**
arm saw.

Mechanism: grounding raises precision and lowers recall (−0.138 recall, BH p =
0.00025), making the model more conservative. Whether that trade is profitable depends
on the generator.

**Oracle vs real retrieval:** +0.0875, p = 0.041 — the condition filter buys a small
but significant amount over honest retrieval.

### 4.5 Refusal is a first-class outcome

gpt-oss-120b, n = 467:

| Arm | Refusal rate | Answers with ≥1 scoreable concept |
|---|---:|---:|
| zero-shot | 0.2% | 92.9% |
| oracle evidence | 29.8% | 49.3% |
| real retrieval | 33.8% | 52.0% |

The grounded arm **declines** rather than confabulating, typically saying the evidence
concerns a different patient. Refusals assert no clinical concept, so they score as
missing and vanish from a naive mean — **which would make the system look healthiest
exactly where it fails.** Report coverage beside every factuality number.

Retrieval top-1 was condition-correct in only **10.7%** of these rows.

### 4.6 H₀₂ — code-mixing robustness. n = 1,165

Spearman ρ between code-mixing intensity and performance, per arm, BH corrected:

| Arm | ρ | 95% CI | BH p | Reading |
|---|---:|---|---:|---|
| **Grounded factual support** | −0.001 | [−0.058, +0.057] | 0.98 | **flat** |
| **Zero-shot factual support** | −0.116 | [−0.171, −0.059] | **0.0003** | **degrades** |
| Grounded hallucination | −0.022 | [−0.080, +0.036] | 0.59 | flat |
| Zero-shot hallucination | +0.042 | [−0.018, +0.101] | 0.30 | flat |

**Grounding absorbs the degradation that code-mixing induces in the ungrounded model.**

This is stronger than the non-significant result obtained by testing the *gain* — a
difference of two noisy arms, under-powered by construction.

**The measure required repair.** The original code-mixing index counted the English
words *"doctor"* and *"please"* as Hindi; they occur in **68.2%** and **35.7%** of
queries. Two hallucination effects reported under the original measure did not survive
repair and are **withdrawn**.

### 4.7 H₀₃ — evidence provenance. n = 160

Four topically matched, equal-sized corpora of **1,872 documents each**: MultiCaRe
case reports, PubMedQA abstracts, MMedBench English exam text, and a
sentence-shuffled MultiCaRe control.

Matching was necessary — the corpora differ enormously in coverage of the 18
conditions: MultiCaRe **67.9%**, MMedBench-English **16.4%**, PubMedQA **2.1%**. An
unmatched comparison would measure corpus *topicality*, not evidence *provenance*.

| Evidence | Refusal | mean F₁ | n scoreable |
|---|---:|---:|---:|
| MultiCaRe (case reports) | **88.1%** | 0.3265 | 63 |
| Shuffled control | 82.5% | 0.3186 | 61 |
| PubMedQA (abstracts) | 79.4% | 0.3476 | 70 |
| MMedBench (exam text) | **76.2%** | 0.2659 | 66 |

**Significant result:** evidence type changes how often the model refuses —
**Cochran's Q = 9.09, df = 3, p = 0.028**. Case reports, the "authoritative" source,
provoke the **most** refusals.

**Undetermined:** answer quality. Pairwise differences are small and none significant.
The four-way omnibus rests on only **13 of 160 rows** where all four conditions
produced a scoreable answer — refusal rates of 76–88% make that joint event rare
(~0.1%). **Report this as absence of evidence, not evidence of absence.**

---

## 5. Two negative results to report in full

**Adaptive truncation does not transfer.** A published similarity-gap heuristic for
adaptive context selection fires on **0 of 3,015** queries at its specified threshold:
it needs a 0.248 similarity gap between adjacent neighbours, and the largest gap
occurring anywhere in the data is **0.109**. Swept across thresholds against a fixed
*k* at matched evidence budget, it wins **0 of 6** settings. Precision is flat
(0.112–0.115) at every setting while recall falls monotonically — the similarity gap
carries no relevance information here.

**A configuration defect dominated the retrieval comparison.** The index was built
with a 128-token sequence limit, truncating **100%** of case narratives (median 307
tokens) and 52.5% of queries, while lexical baselines read the full document — so
BM25 had ~2.4× more information. Correcting this and adding passage chunking moved
dense Recall@1 on Hinglish from 0.1144 to 0.1280 and **reversed** the ordering against
BM25. Report it: the uncorrected comparison supported the opposite conclusion.

---

## 6. Limitations (all must appear)

- **Relevance labels are coarse** — condition-group matching over 18 groups admits
  same-group but clinically irrelevant cases. Absolute Recall is a lower bound.
- **The concept lexicon covers 26 concepts and is not clinician-validated.**
- **The unbiased reference is narrow and repetitive** — averages 1.5 concepts, takes
  412 distinct values over 2,988 rows, one covering 22%. Non-independence handled with
  cluster bootstraps, but the construct is partial.
- **Generator dependence** — two generation results differ in *sign* between
  generators. Both reported; neither selected.
- **Provider instability** — `llama-3.1-8b-instant`, the generator behind the n=1,165
  results, was **decommissioned mid-study** and now returns HTTP 404. Those outputs
  are archived and re-scorable but cannot be regenerated. A real hazard for work on
  hosted models.
- **H₀₃ answer quality is under-powered** for the reason in §4.7.
- **No clinical deployment claim.** Absolute Recall@1 of 0.175 is far below what
  patient-facing use would require.

---

## 7. What must NOT be claimed

1. **Do not report "+73.5% factuality" and "−44% hallucination" as two findings.** The
   second is arithmetically the first, sign-flipped.
2. **Do not quote an absolute concept-precision level as a quality measure.** A
   one-word constant beats the system 4.7×.
3. **Do not claim adaptive context selection works.** It never fired.
4. **Do not claim the system is deployable.** Retrieval is ~2–3× the random floor.
5. **Do not describe the system as multimodal** — no images are used. Earlier project
   documents mention LLaVA, BioMedCLIP, QLoRA and DPO; **none were implemented.**

---

## 8. Placeholders the authors must fill

Library versions and compute (§4.3 Implementation) · Zenodo DOI for Data availability ·
repository URL and release tag for Code availability · Acknowledgements ·
**Related Work citations** (papers are in `research-work/papers/`: HiFACTMix, MedSumm,
MMed-RAG, CroCoSum, LLaVA-Med, plus the MMCQSD and MultiCaRe dataset papers, LaBSE,
MuRIL, BM25, reciprocal-rank fusion).

---

## 9. Suggested framing for the abstract and conclusion

The defensible story, in one sentence:

> Code-mixing significantly degrades clinical retrieval — four times more for lexical
> than for dense methods — and while grounding measurably improves factual support and
> shields it from code-mixing intensity, the apparent size of that benefit depends
> substantially on the evaluation protocol.

Three contributions:

1. **The retrieval-stage code-mixing penalty**, measured against gold human
   translations with a leakage gate, with the lexical/dense asymmetry and the
   script-not-language mechanism.
2. **Hypothesis tests for grounding and code-mixing robustness**, replicated across
   two generator families.
3. **A demonstration that evaluation design changes the reported benefit**, with the
   degenerate baselines needed to detect it.
