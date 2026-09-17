# Revision: what the code now says

**For:** Devika Jonjale & Saachi Shinde
**Paper:** ICCSDI 2026 #216, Minor Revision · new deadline 19 September 2026
**Branch:** `review-revisions`
**Companion:** `REVIEW-RESPONSE-PLAN.md` (what each reviewer asked for and why)

This file is the results half. Every number below comes from a committed script;
run `python -m src.analysis.reproduce_all --list` to see the stages.

---

## 1. Three numbers in the submitted paper do not hold

Run `python -m src.analysis.verify_paper_numbers`. It checks 36 headline claims
against the committed result files: **33 verify, 3 do not.** These must be fixed
before resubmission, and none of them is a typo.

### 1.1 A non-significant result is reported as significant  ⚠ most serious

> §5.3: *"Oracle evidence is superior to real retrieval by +0.0875 (p = 0.041):
> the condition filter is a small but significant gain over honest retrieval."*

**That number appears in no committed file.** The two runs that exist say:

| Source | Oracle − real | n | p |
|---|---:|---:|---:|
| `results/h1_real_120b/` (gpt-oss-120b) | +0.0748 | 76 | **0.226** |
| `results/h1_real_retrieval/` (gpt-oss-20b) | +0.0755 | 125 | **0.106** |

Both are **non-significant**. Replace the sentence with something like:

> Oracle evidence scores higher than real retrieval by +0.075, but the difference
> is not significant on either generator (p = 0.226, n = 76; p = 0.106, n = 125),
> so the condition filter's advantage is not established at this sample size.

### 1.2 Table 3's gpt-oss rows do not match the re-scored results

| Row | Paper | `results/rescored/contrasts.csv` | `results/h1_real_120b/` |
|---|---|---|---|
| gpt-oss-120b, oracle | +0.2659, n=223, d=0.662 | +0.2608, n=214, d=0.640 | +0.2849, n=124, d=0.694 |
| gpt-oss-120b, real retrieval | +0.2238, n=237, d=0.616 | **+0.0704**, n=221, d=0.191 | +0.2347, n=131, d=0.661 |

No file reproduces the paper's values. **Use `contrasts.csv`**: it is the canonical
re-scoring (negation-repaired lexicon, BH-corrected across the family), which is
what §4.2 says the paper does.

**This changes a claim.** The real-retrieval row is what supports *"replicates
across two generator families"*. At **+0.0704 (d = 0.191)** rather than +0.2238,
the replication under real retrieval is much weaker than stated. Under oracle
evidence it does replicate (+0.2608, d = 0.640). Say exactly that.

### 1.3 One sentence is right but ambiguous, and the ambiguity flips its sign

> §5.3: *"grounding trades recall for precision: it decreases concept recall by
> 0.138 (BH p = 0.00025)"*

Correct — **for gpt-oss-120b**. The llama arm moves the **opposite** way
(recall **+0.0774**, BH p = 9.3×10⁻⁷). Name the generator, or the sentence
asserts as general something that one of your two generators contradicts.

---

## 2. H₀₂ now matches its own hypothesis (R3.3)

`python -m src.analysis.h2_prespecified_test` → `results/h2_per_arm/h2_prespecified_test.md`

The 9 Feb README planned *"Two-way ANOVA / Kruskal-Wallis"*. With arm varying
within query, that interaction **is** the test of whether the paired gain changes
across code-mixing levels. Reported now as primary:

| Test | Result |
|---|---|
| Interaction, ANOVA | F = 3.03, **p = 0.049** |
| Interaction, Kruskal–Wallis | H = 5.11, **p = 0.078** |
| Interaction, continuous (Spearman) | ρ = **+0.081**, 95% CI [+0.023, +0.138], p = 0.006 |
| Grounded arm alone (9 Feb wording) | ρ = −0.001, **equivalent to zero** (90% CI inside ±0.10) |
| Zero-shot arm (exploratory) | ρ = −0.116, p = 7.7×10⁻⁵ |

Mean gain by tertile: **+0.2041 → +0.2253 → +0.2741**.

**The two pre-specified tests disagree** (0.049 vs 0.078). Report both. Choosing
the one that crosses 0.05 after seeing both is exactly what a pre-specified plan
is supposed to prevent.

**Defensible wording:** grounded factual support is *stable* across code-mixing
intensity (statistically equivalent to flat); the *gain* rises with Hindi
proportion because the ungrounded baseline falls. Evidence against H₀₂ is
borderline on the tertile tests and significant on the continuous measure.

⚠ The paper's claim that the difference test is "non-significant" is only true
under the **old** code-mixing measure (KW p = 0.193). Under the repaired measure
it is not. Update that sentence.

---

## 3. Retrieval metrics: deeper, and correctly computed (R3.2)

`python -m src.analysis.retrieval_v2` → `results/retrieval_v2/`

### 3.1 A metric bug, fixed
`nDCG@10` was computed with the ideal DCG **fixed at 1.0** and the result clipped,
which is valid only when a query has exactly one relevant document. These queries
have a **median of 597**. Two of the five copies of the metric function had this
bug. All are replaced by `src/evaluation/retrieval_metrics.py` (8 tests).

**"Recall@1" is a hit rate, not recall** — with ~597 relevant cases, recall@10
could never exceed 10/597. Renamed **Success@k** throughout. Say so once in §4.2;
reviewers in IR will notice otherwise.

### 3.2 New Table 1 numbers

| System | Query | S@1 | S@5 | S@10 | MRR@10 | nDCG@10 | graded nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Hybrid (RRF) | Hinglish | 0.1751 | 0.4604 | 0.6564 | 0.2987 | 0.1364 | 0.1068 |
| Hybrid (RRF) | English | 0.1973 | 0.5347 | 0.7393 | 0.3438 | 0.1692 | 0.1439 |
| LaBSE | Hinglish | 0.1280 | 0.4299 | 0.6325 | 0.2584 | 0.1204 | 0.0947 |
| LaBSE | English | 0.1486 | 0.4604 | 0.6783 | 0.2835 | 0.1257 | 0.0964 |
| BM25 | Hinglish | 0.0935 | 0.3844 | 0.6116 | 0.2173 | 0.1012 | 0.0715 |
| BM25 | English | 0.1847 | 0.5499 | 0.7373 | 0.3385 | 0.1745 | 0.1537 |
| TF-IDF | Hinglish | 0.0842 | 0.3357 | 0.5599 | 0.1972 | 0.1018 | 0.0745 |
| TF-IDF | English | 0.1529 | 0.4292 | 0.6153 | 0.2729 | 0.1475 | 0.1220 |

Every S@1 value is unchanged from the submitted Table 1, so nothing there breaks.

### 3.3 The penalty is far more robust than one cutoff showed

Tested for all four systems at four depths plus MRR and two nDCGs — **28
combinations. The penalty is positive in 28 of 28 and significant in 26.**

The asymmetry also *grows* with depth:

| System | S@1 | S@3 | S@5 | S@10 | MRR@10 | graded nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | +0.0912 | +0.1629 | +0.1655 | +0.1257 | +0.1212 | +0.0822 |
| LaBSE | +0.0206 | +0.0186 | +0.0305 | +0.0458 | +0.0251 | **+0.0016 (n.s.)** |

**On graded relevance LaBSE shows no significant penalty at all**, while BM25
loses 0.082. The "4.4× larger for lexical" figure is the *weakest* version of
your finding; on rank-sensitive measures the gap is an order of magnitude.

Also: **TF-IDF's missing test is now filled in** (+0.0687, p = 1.9×10⁻¹⁹). The
submitted Table 1 left it as "—" with no explanation.

New figure `results/retrieval_figures/fig3_penalty_by_depth.png` shows this, and
is a better use of a page than the current Fig. 3 (which duplicates Table 4).

### 3.4 Graded relevance (an automatic proxy, not clinical)
grade 2 = same condition group **and** ≥1 shared lexicon concept with the gold
English question; grade 1 = same group only. 90.6% of queries can reach grade 2.
**Call it an automatic proxy in the paper.** The clinical version is §6.

---

## 4. H₀₃: the power calculation, and a problem nobody raised (R3.6, R1.e)

`python -m src.analysis.h3_power` → `results/h3_provenance/h3_power.md`

**The four-way omnibus needs 616 queries for 50 usable ones** (2.8M tokens ≈ 14
days of one Groq organisation's daily quota); 200 usable needs 2,462 queries.
With the low-refusal prompt already implemented it would need only ~77. That is
the sentence to put in the paper instead of a vague "underpowered".

⚠ **New finding, not raised by any reviewer: Table 5's F1 column is computed
mostly on refusals.** Of the 63 MultiCaRe answers counted as scoreable, **59 are
also flagged as refusals** — hedges that decline to answer but still name a
concept ("main confirm nahi kar sakta, lekin rash ho sakta hai").

| Corpus | Refusal | Scoreable | of which also refusals | Answered *and* scoreable |
|---|---:|---:|---:|---:|
| MultiCaRe | 88.1% | 63 | 59 | 4 (2.5%) |
| Shuffled control | 82.5% | 61 | 53 | 8 (5.0%) |
| PubMedQA | 79.4% | 70 | 53 | 17 (10.6%) |
| MMedBench | 76.2% | 66 | 56 | 10 (6.2%) |

Requiring a genuine answer leaves **0 of 160** queries usable under all four
conditions. Either drop Table 5's F1 column or carry this caveat with it.

Largest pairwise effect observed is dz = 0.22, which would need ~164 jointly
scoreable pairs for 80% power.

---

## 5. Transliteration and translation baselines (R1, R2)

*(Filled in when the two runs finish — see §8.)*

---

## 6. Annotation sheets are ready for you  ← your task

`results/annotation/` — generated, blinded, and shuffled:

| File | Rows | What it gives |
|---|---:|---|
| `annotation_sheet_A_annotator[AB].csv` | 120 | extractor precision/recall vs human labels; a reference-free quality score |
| `annotation_sheet_B_annotator[AB].csv` | 240 | clinical relevance 0/1/2 for the top-3 retrieved cases |
| `README.md` | — | labelling instructions |
| `concept_vocabulary.csv` | 26 | the lexicon, for the `missed_concepts` column |

Then: `python -m src.evaluation.annotation_agreement`

**This single effort answers four reviewer points** (R3.4, R1.b, R1.c, R1.d).
Sheet A is stratified by arm and code-mixing tertile with the arm hidden; sheet B
hides system and language. Two people must label **independently** or the κ is
meaningless.

If you can get a clinician for 2–3 hours, their copy is the one that lets you
write "clinician-validated". Without one, report author agreement and keep
clinician validation as a limitation — do **not** write "clinically validated".

---

## 7. What the paper must now say

| # | Section | Change |
|---|---|---|
| 1 | §5.3 | Oracle-vs-real is **+0.075, not significant** (was "+0.0875, significant") |
| 2 | Table 3 | gpt-oss rows from `contrasts.csv`; real-retrieval replication is **+0.0704, d = 0.191** |
| 3 | §5.3 | Name the generator in the "trades recall for precision" sentence |
| 4 | §5.4, Table 4, abstract | H₀₂ primary = interaction test; per-arm = exploratory; grounded arm *equivalent* to flat |
| 5 | §4.2, Table 1, §5.1 | Success@k not Recall@k; add S@10, MRR@10, nDCG@10; 28/28 positive, 26/28 significant |
| 6 | Table 1 | Fill in TF-IDF's penalty and p-value |
| 7 | §5.5, abstract | H₀₃ exploratory and provisional; refusal is a **secondary** outcome; add the 616-query cost |
| 8 | §5.5 | Table 5's F1 rests on refusals — caveat or drop the column |
| 9 | §1, §4 | "pre-specified", not "pre-registered"; cite commit `0917322` (9 Feb 2026); H₀₄ was added later |
| 10 | §4.2 | Add the deviations-from-plan paragraph |
| 11 | Figures | Replace Fig. 3 with `fig3_penalty_by_depth.png` |

---

## 8. Commands

```bash
python -m src.analysis.verify_paper_numbers      # 36 claims vs committed results
python -m src.analysis.h2_prespecified_test      # H02 as planned
python -m src.analysis.retrieval_v2              # Table 1 (~35 min, CPU)
python -m src.analysis.retrieval_figures         # Figs 1-3
python -m src.analysis.h3_power                  # H03 power
python -m src.analysis.transliteration_baseline  # MuRIL script test
python -m src.analysis.translation_baseline --n 500   # resumes from checkpoint
python -m src.evaluation.annotation_sample       # rebuild sheets
python -m src.evaluation.annotation_agreement    # score them
python -m pytest -q                              # full suite
```
