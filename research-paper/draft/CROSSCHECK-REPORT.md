# Cross-Check: Draft #1 vs. Committed Result Files

**Checked:** every table value and in-text statistic in `research paper - draft #1.docx`
**Against:** `results/**` on branch `saachi-hardening`
**Method:** values read from the CSVs; where a committed `.md` report disagreed with the draft, the statistic was **recomputed from the raw scored data** to determine which is current.

**Verdict: the draft is in very good shape.** Tables 2, 5, 6 and 7 and Fig. 3 reproduce exactly. One wording defect, three precision issues, and one 2-s.f. rounding slip. **No finding or conclusion changes.**

> ### ⚠️ Correction to this report (issued after regenerating the source report)
>
> Section B originally claimed the Table 4 gpt-oss p-values and the §5.3
> oracle-vs-real p were **wrong by an order of magnitude**. **That claim was
> itself wrong, and has been withdrawn.**
>
> The cross-check recomputed those statistics with `scipy.stats.wilcoxon(...,
> zero_method="zsplit")`. The project's own analysis script
> (`src/analysis/h1_oracle_vs_real.py`) calls `stats.wilcoxon(d)` — scipy's
> **default `zero_method="wilcox"`, which discards zero-differences**. The two
> conventions give materially different p-values on these data. The draft
> followed the project's established convention and was correct.
>
> Regenerating the report at the current n = 467 reproduces the draft's values:
> oracle **5.547 × 10⁻¹⁷**, real **6.533 × 10⁻¹⁷**, oracle − real **+0.0875,
> p = 0.0411**.
>
> **The lesson, which applies to the rest of the validation pass:** a
> re-computation that uses a different method is not a verification. Match the
> original procedure, or the disagreement measures the analyst, not the paper.
>
> The one real defect in B is a rounding slip: 5.547 × 10⁻¹⁷ is **5.5** × 10⁻¹⁷
> to two significant figures, not 5.6.

---

## A. Verified exactly — no action

| Item | Status |
|---|---|
| **Table 2** (all 4 systems × 2 languages, penalties, McNemar p) | ✅ exact vs `retrieval_v2/h4_v2_tests.csv` |
| Random floor 0.0626 | ✅ `h4_metrics.csv` = 0.062603 |
| §5.1 single-vector index (MuRIL 0.0640 / 0.1821; BM25 0.1343; e5 0.1303; TF-IDF 0.1167; LaBSE 0.1144) | ✅ all exact |
| **Table 5** (refusal/coverage, n = 467) | ✅ recomputed: 0.21/92.93, 29.76/49.25, 33.83/52.03 |
| **Table 6** (H₀₂, all 4 rows, ρ/CI/BH p) | ✅ exact vs `h2_corrected_cmi_stats.csv` (`hindi_prop_v2` rows) |
| **Table 7** (H₀₃, all 4 corpora) | ✅ exact vs `h3_summary.csv` |
| Cochran's Q = 9.09, df 3, p = 0.028 | ✅ 9.092, p = 0.02809 |
| **Fig. 3** all six reference-effect values | ✅ exact vs `h1_reference_effect.csv` |
| §5.3 recall −0.138, BH p = 0.00025 | ✅ −0.138448, p_bh = 2.467×10⁻⁴ |
| §5.3 top-1 condition-correct 10.7% | ✅ recomputed 10.71% |
| **Table 4** llama rows (n = 1,165 and n = 669) | ✅ exact |
| Table 4 gpt-oss **n, Δ, Cohen's d** | ✅ recomputed 223/+0.2659/0.662 and 237/+0.2238/0.616 |
| §5.6 adaptive truncation (threshold 0.248, max gap 0.109, 0/3,015, 0 of 6, precision 0.112–0.115) | ✅ exact vs `truncation_report.md` |
| §5.6 LaBSE 0.1144 → 0.1280 | ✅ |
| §6.1 unbiased reference (1.5 concepts, 412 distinct / 2,988 rows, one covers 22%) | ✅ recomputed 412 / 2,988 / 22.5% |
| 26-concept lexicon | ✅ `src/evaluation/concept_lexicon.py` = 26 |
| Median case 554 words; 61,316 filtered; 10,000 indexed; 18 groups; 41,746 passages; 3,015 pairs | ✅ |

---

## B. ✅ WITHDRAWN — the p-values were correct; one rounding slip remains

### B1. Table 4, gpt-oss-120b p-values — draft was CORRECT

| Row | Draft | Regenerated at n = 467 | Verdict |
|---|---:|---:|---|
| gpt-oss-120b, oracle | 5.6 × 10⁻¹⁷ | 5.547 × 10⁻¹⁷ | ⚠️ round to **5.5** |
| gpt-oss-120b, real retrieval | 6.5 × 10⁻¹⁷ | 6.533 × 10⁻¹⁷ | ✅ correct |

The only change needed is 5.6 → **5.5** on the oracle row.

### B2. §5.3 "+0.0875 (p = 0.041)" — draft was CORRECT

Regenerated: **+0.0875, n = 138 both-scoreable, p = 0.0411**. Leave as printed.

### B3. 🔴 The stale report — this part stands, and was the real problem

`results/h1_real_120b/h1_oracle_vs_real_report.md` was committed at **n = 256** while its scored CSV had grown to **n = 467**. It reported *+0.0748, n = 76, p = 0.226* — **non-significant**, contradicting the manuscript, and it also showed top-1 correctness as 8.2% against the paper's 10.7%.

**Regenerated.** It now reproduces the draft throughout: refusal 0.2 / 29.8 / 33.8%, coverage 92.9 / 49.3 / 52.0%, Δ +0.2659 and +0.2238, top-1 correct 10.7%, oracle − real +0.0875 (p = 0.0411).

The generator (`src/analysis/h1_oracle_vs_real.py`) had hard-coded paths, which is why the 120b run's report was never refreshed. It now takes `--scored` / `--out`, so both runs can be regenerated:

```
python -m src.analysis.h1_oracle_vs_real --scored results/h1_real_120b/h1_real_scored.csv --out results/h1_real_120b
```

`results/h1_real_retrieval/` (gpt-oss-20b, n = 268) was checked and is **current** — no action.

---

## C. 🔴 One methodological claim uses the wrong word

### §4.1: "…names the query's condition group **verbatim** in 96.2% of rows"

Measured on all 3,015 pairs:

| Claim | Measured |
|---|---:|
| Summary contains the underscore-joined label verbatim (`swollen_tonsils`) | **33.6%** |
| Summary contains an image caption at all (`"The image …"`) | **100.0%** |
| Summary mentions its condition **in words** (`"swollen tonsils"`) | ~96–98% |

So 96.2% is defensible as a *word-level* match but **not as "verbatim"** — verbatim is 33.6%. Since the leakage gate is a methodological centrepiece, the wording should be precise. Suggested rewrite:

> "…embeds an image caption that names the query's condition group in 96.2% of rows, and reproduces the underscore-joined group label exactly in 33.6%. The gate strips the caption and asserts that no label string survives."

This is *stronger*, not weaker: it shows you measured leakage at two levels of strictness.

---

## D. 🟠 Three precision issues — should fix

### D1. Table 3's "n = 1,876" is wrong for three of seven rows

| Row | Actual n |
|---|---:|
| Constant `"swelling"`, `"swelling and erythema"`, `"erythema"`, `"pain"` | 1,876 ✅ |
| Copy of the reference, verbatim | **1,705** |
| Grounded system (0.1528) and Zero-shot system (0.1066) | **701** (`m4_caption/m4_summary.csv`) |

The headline "4.7×" therefore compares 0.7132 (n = 1,876) against 0.1528 (n = 701) — different samples. The point survives (the gap is enormous), but the caption should say so. Suggested caption:

> "…degenerate baselines (n = 1,876), the verbatim-copy ceiling (n = 1,705), and the two systems (n = 701). Sample sizes differ because rows are dropped where an arm asserts no scoreable concept."

### D2. `const:six-common` is measured but omitted from Table 3

`results/rescored/degenerate_baselines.csv` includes a six-word constant scoring **0.2338** — still **1.5× the grounded system**. It is arguably the most persuasive row in the table, because a reader can dismiss a one-word answer as pathological but not a six-concept one. Recommend adding it.

### D3. §5.4 "doctor" 68.2% / "please" 35.7% — say which sample

Those figures are for the **n = 1,165 evaluated subset** (recomputed: 68.2% / 35.7% ✅). Across all **3,015** queries they are **71.0% / 38.0%**. Either is fine; the draft just needs to name the denominator.

---

## E. 🟡 One nuance worth a footnote

§5.6 says correcting the sequence limit "and adding passage chunking" raised Hinglish Recall@1 from 0.1144 to 0.1280. True — but `results/index_truncation/comparison.csv` and `retrieval_v2_report.md` show that simply raising the limit to 256 tokens gives **0.1310**, *higher* than passage chunking's 0.1280. On English (Q2) the 256-token index is **worse** (0.1466 vs 0.1602).

So passage chunking is not a monotone improvement; it is the configuration that puts dense and lexical on **matched content**, which is the methodological point. Recommend one clause making that explicit, so a reader comparing your own files doesn't think a better number was passed over.

---

## F. RESOLVED — the four unsourced claims, measured directly

These figures existed **only in a docstring** (`src/analysis/rebuild_index_full.py`); no code in the repository computes them. They were therefore re-measured from the committed data with the LaBSE tokenizer (`BertTokenizerFast`, `add_special_tokens=True`, documents capped at 200 words exactly as `build_index.py` does).

| Claim in draft | Measured | Verdict |
|---|---:|---|
| case narratives, median **307** tokens | **300** | ⚠️ corrected to 300 |
| case narratives, **100%** truncated at 128 | **99.9%** | ⚠️ corrected to 99.9% |
| queries, **52.5%** truncated at 128 | **60.0%** | 🔴 corrected to 60.0% |
| BM25 read **~2.4×** more | **2.34–2.38×** | ✅ correct |
| 11 pairs → 3,015, **274-fold** | `plans/limitation/limitation-resolution-summary.md` | ✅ sourced |

The query figure was the largest gap. No tokenisation convention reproduces 52.5% — the alternatives tested (without special tokens 58.6%, on the n = 1,165 subset 57.6%, subset without special tokens 56.6%) all sit near 60%, not 52.5%. **The direction of the argument is unaffected and slightly strengthened:** more queries were truncated than the draft claimed.

*(Note: 132 tokens is the median for Hinglish **queries**, 300 for **case narratives**. The draft assigns each to the right object — there is no conflation.)*

**All four are applied in draft #2.**

---

## I. Repository audit (branch `saachi-hardening`, commit `2d7c927`)

Because Code availability now points reviewers at the repository, it was audited for anything that would contradict the manuscript.

### Clean — no action

| Check | Result |
|---|---|
| README / `config.yaml` capability claims | ✅ **Correct.** Both carry explicit disclaimers. README: *"This is a text-only system. It does not use LLaVA, BioMedCLIP, QLoRA or DPO."* `config.yaml` opens *"This file describes the system that EXISTS"* and has an **Explicitly NOT implemented** section |
| Superseded results | ✅ `results/ARCHIVE-NOTE.md` separates citable from non-citable and names three specific traps |
| Report staleness (10 report/CSV pairs) | ✅ All current. The four flagged by keyword were summary tables (8 or 10 rows) whose reports correctly state n = 3,015 |
| Canonical lexicon | ✅ `src/evaluation/concept_lexicon.py` = **26 concepts**, matching the paper |
| Reproducibility entrypoint | ✅ `python -m src.analysis.reproduce_all --list` runs, reports 13 stages with per-stage input status, and documents what is *not* reproducible |
| Large artifacts excluded from git | ✅ indices, pairing tables and corpora correctly ignored; 117 result files versioned |

### 🟡 Worth knowing

1. **Two stages cannot run from a clean clone**, because their inputs are gitignored binaries:
   - `retrieval-v2` needs `data/passage_index/passage_emb.npy` — **this produces Table 2, the headline result** (~3 h CPU to rebuild)
   - `h3-corpora` needs `pubmedqa_records.csv` and `mmedbench_questions.csv` (derivable from the public sources)

   This is the strongest argument for completing the Zenodo deposit: without it, the paper's central table is not reproducible from the repository alone.

2. ~~**`config.yaml` lists H03 under `future_work`**~~ — **fixed** (commit `1a8e7e3`). The entry now records that H03 *was* run and that only the answer-quality half remains open, as a power problem rather than an unrun experiment.

3. **Five concept lexicons still coexist** (26 canonical, plus 18/28/28/35 in `src/prototype/`). The paper's number is the canonical one and is correct; the others belong to superseded prototype code. Harmless, but a reader may wonder which is authoritative.

4. **Nine analysis scripts write to hard-coded output directories with no CLI override.** This is the same class of defect that produced the stale 120b report — a run that writes elsewhere leaves the committed report untouched. `h1_oracle_vs_real.py` has been parameterised; the rest have not.

5. Residual stubs: `src/data/hmg_builder.py` and `src/data/preprocess.py` still raise `NotImplementedError`. Neither is on any path used by the paper.

---

## G. Reference [2] — MultiCaRe (was a placeholder)

From the dataset's official *How to Cite*:

> **[2]** Nievas Offidani, M., Roffet, F., González Galtier, M.C., Massiris, M., Delrieux, C.: An open-source clinical case dataset for medical image classification and multimodal AI applications. Data **10**(8), 123 (2025). https://doi.org/10.3390/data10080123

Cite the Zenodo deposit too if you want to pin the version you used:

> Nievas Offidani, M.: MultiCaRe: an open-source clinical case dataset for medical image classification and multimodal AI applications (version 3) [Data set]. Zenodo (2025). https://doi.org/10.5281/zenodo.10079369

### The other four placeholders, in the same house style

> **[3]** Feng, F., Yang, Y., Cer, D., Arivazhagan, N., Wang, W.: Language-agnostic BERT sentence embedding. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 878–891 (2022)
>
> **[12]** Robertson, S., Zaragoza, H.: The probabilistic relevance framework: BM25 and beyond. Found. Trends Inf. Retr. **3**(4), 333–389 (2009). https://doi.org/10.1561/1500000019
>
> **[13]** Cormack, G.V., Clarke, C.L.A., Buettcher, S.: Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In: Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 758–759 (2009). https://doi.org/10.1145/1571941.1572114
>
> **[14]** Khanuja, S., Bansal, D., Mehtani, S., Khosla, S., Dey, A., Gopalan, B., Margam, D.K., Aggarwal, P., Nagipogu, R.T., Dave, S., Gupta, S., Gali, S.C.B., Subramanian, V., Talukdar, P.: MuRIL: multilingual representations for Indian languages. arXiv:2103.10730 (2021)

Spot-check [3] and [14] page/venue details against the PDFs, as you planned for the other nine.

---

## H. Priority

All items below are **applied** in `research paper - draft #2.docx` (the original draft #1 is untouched), except F, which needs the authors.

| | Action | Status |
|---|---|---|
| 1 | Title → *A Measurement Study for Hinglish Clinical Decision Support* | ✅ applied |
| 2 | Reword the "verbatim / 96.2%" leakage claim (C) | ✅ applied |
| 3 | Table 3 caption sample sizes + `six-common` row (D1, D2) | ✅ applied |
| 4 | Denominator for doctor/please (D3) | ✅ applied |
| 5 | Footnote the 256-token nuance (E) | ✅ applied |
| 6 | Paste the five citations (G) | ✅ applied |
| 7 | Oracle-row p rounding 5.6 → 5.5 (B1) | ✅ applied |
| 8 | Regenerate the stale 120b report (B3) | ✅ done on `saachi-hardening` |
| 9 | Correct the four §F truncation figures (307→300, 100%→99.9%, 52.5%→60.0%) | ✅ applied |
| 10 | §4.3 library versions, hardware, compute budget | ✅ applied |
| 11 | Acknowledgements | ✅ applied — **review the wording; add mentors only with their permission** |
| 12 | Code availability — repo URL + **release tag `v1.0-iccsdi2026`** (tagged and pushed) | ✅ applied |
| 13 | Data availability | ⚠️ partly — reworded to be true as written; **Zenodo DOI still needed** |

### Page count — trimmed from 13 to 12

Draft #1 rendered at **11 pages** (verified in Microsoft Word, not the stale cached metadata). The placeholder-filling pass added 2,449 characters across §4.3, the Declarations, and the five new references, pushing the real rendered count to **13 pages** — with page 13 holding only the final reference (`[14]` MuRIL) as a single orphaned line.

Tightened the §4.3 implementation paragraph and the three Declarations items (Acknowledgements, Data availability, Code availability) — removing explanatory phrasing only, no facts, versions, hardware specs, counts, or links were dropped. Verified fact-by-fact against the trimmed text; all 14 checked items present. Re-rendered in Word after each change: **12 pages**, and page 12 now runs cleanly through all 14 references with no gap.

A backup of the pre-trim file is kept at `research paper - draft #2 (pre-trim backup).docx` if any of the tightened wording needs reverting.

### The only outstanding item

**Mint the Zenodo deposit and paste the DOI.** It is the last placeholder in the manuscript, and §I.1 shows it is not merely a formality: the passage-embedding cache behind **Table 2** is not in the repository, so without the deposit the paper's headline table cannot be reproduced from the code alone. Deposit `data/passage_index/`, `data/faiss_index/`, `mmcqsd_multicare_paired.csv` and `multicare_filtered.csv` (≈ 400 MB, inside Zenodo's free tier).
