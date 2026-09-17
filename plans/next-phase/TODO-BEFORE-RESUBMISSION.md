# To-do before resubmission

**Paper:** ICCSDI 2026 #216 · Minor Revision · **deadline 19 September 2026**
**Branch:** `review-revisions`
**Code status:** finished. Everything below is writing, formatting and administration.
**Revised manuscript:** `research-paper/research paper - devika jonjale & saachi shinde - REVISED.docx`
(built by `research-paper/apply_revision.py` from the submitted file; every change is
colour-coded and carries a comment — yellow = correction, green = new for reviewers,
turquoise = reworded/renamed; comments beginning ACTION or CHECK need a decision from you).
Preview: `research-paper/draft/REVISED-preview.pdf`. Items 1.1–1.8 and 2.1–2.13, 2.16 are
applied in that file; 2.14–2.15 are applied in condensed form (labelled intro beats,
positioning Table 1). Still yours: 4.1–4.7, the response letter, and proofreading.

Companion documents:
- `REVIEW-RESPONSE-PLAN.md` — what each reviewer asked for, and why
- `REVISION-CODE-RESULTS.md` — every new number, with its source

---

## 0. Do these two things first, today

| # | Task | Why it cannot wait |
|---|---|---|
| 0.1 | **Check CMT** — is the revised-upload still open? | The original deadline (15 Sep) has already passed |
| 0.2 | **Email the chair** naming a specific date | A short, concrete request is far more likely to be granted than an open one |

---

## 1. Corrections that must be made (the paper is wrong without them)

Run `python -m src.analysis.verify_paper_numbers` after editing; it re-checks 36
claims against the committed results.

| # | Where | Change | Est. |
|---|---|---|---|
| 1.1 | §5.3 | **"+0.0875 (p = 0.041)… small but significant" → "+0.075, not significant"** (p = 0.226, n = 76; p = 0.106, n = 125). The current sentence claims a significant result that no run produced | 10 min |
| 1.2 | Table 3 | Replace both gpt-oss rows with `results/rescored/contrasts.csv`: oracle **+0.2608** (n = 214, d = 0.640), real retrieval **+0.0704** (n = 221, d = 0.191) | 15 min |
| 1.3 | §5.3, §7, abstract | Because 1.2 changes it: replication across generator families holds **under oracle evidence**, and is **much weaker under real retrieval**. Do not say "replicates" unqualified | 20 min |
| 1.4 | §5.3 | Name the generator in "grounding trades recall for precision": it is **gpt-oss-120b**; the llama arm moves the opposite way (+0.0774) | 5 min |
| 1.5 | §1, §4 | **"pre-registered" → "pre-specified"** everywhere; cite commit `0917322` (9 Feb 2026) with its GitHub URL; state that **H₀₄ was added later** | 20 min |
| 1.6 | §4.2 (new short para) | Add the five **deviations from the planned analysis** — copy from `results/prespecification_record.md` | 20 min |
| 1.7 | Abstract, §3.1, §4.1, §5.1 | **"gold human translations" → "gold English summaries".** MMCQS supplies clinician-style *summaries* (~21 words) not translations (~97 words). The current wording misdescribes the data | 20 min |
| 1.8 | §5.1, §6.1 | Add the consequence: the penalty compares a patient narrative against a clinical summary, so it is an **upper bound on the language effect**, conflating language with conciseness and register. The translate-then-retrieve arm is the cleaner language-only comparison | 30 min |

---

## 2. Reviewer-requested content (all numbers are ready)

| # | Where | Change | Source | Est. |
|---|---|---|---|---|
| 2.1 | §4.2, Table 1, §5.1 | **Rename Recall@k → Success@k** and define it once (with ~597 relevant cases per query it is a hit rate, not recall) | `retrieval_v2_report.md` | 20 min |
| 2.2 | Table 1 | Add **S@10 and MRR@10** columns; fill the **missing TF-IDF penalty** (+0.0687, p = 1.9×10⁻¹⁹) | same | 30 min |
| 2.3 | §5.1 | One sentence: the penalty is **positive in 28 of 28 system×metric combinations, significant in 26**, and the lexical/dense gap widens with depth | same | 15 min |
| 2.4 | §4.2, §6.1 | Note the **corrected nDCG** (the old one fixed the ideal at 1.0, valid only for a single relevant document) and the **graded-relevance proxy** — label it automatic, not clinical | same | 15 min |
| 2.5 | §5.4, Table 4, abstract | **H₀₂ rewritten**: primary = the pre-specified interaction test (ANOVA p = 0.049, KW p = 0.078 — report both); grounded arm **equivalent to flat**; per-arm = exploratory. Delete "the per-arm design is deliberately stronger…" | `h2_prespecified_test.md` | 45 min |
| 2.6 | §5.5, abstract, §7 | **H₀₃ → exploratory and provisional**; refusal is a **secondary** outcome on a different variable; add "a powered replication needs ~616 queries for 50 usable ones" | `h3_power.md` | 30 min |
| 2.7 | §5.5 | Table 5's F1 column: **59 of 63 scoreable MultiCaRe answers are also refusals**. Add the caveat or drop the column | same | 15 min |
| 2.8 | §5.3 / Table 3 | Replace Table 3 + Fig. 2 with the **single 2×3 reference matrix** (positive in 5/5 conditions against the circular reference, 1/5 against the unbiased one) | `reference_matrix.md` | 30 min |
| 2.9 | Abstract | State the grounding effect **reference-first**: significant against the evidence given; reduced or absent against a reference neither arm saw | same | 20 min |
| 2.10 | §6.1 | **New limitation — extractor validity.** An independent LLM reader agrees only moderately (κ = 0.494; precision 0.412, recall 0.686), and the bias is **arm-dependent** (0.480 grounded vs 0.358 zero-shot), which **inflates** the measured grounding benefit. Say it yourselves | `lexicon_validation_report.md` | 25 min |
| 2.11 | §6.1 | **Lexicon coverage**: 98.3% of references carry ≥1 concept, mean 2.76, **13.9% carry exactly one** — which is why absolute scores are unstable. The unbiased reference is free-text captions with frequent misspellings | same | 15 min |
| 2.12 | §5.1, §6, §7 | **Transliteration result** — replaces "a hypothesis the study motivates but does not test" | `translit_report.md` | 25 min |
| 2.13 | §2, §4.2, §5.1 | **Translate-then-retrieve baseline** — Reviewer 2's "comparison with existing methods" | `translation_report.md` | 30 min |
| 2.14 | §1 | Restructure into **Problem → Gap → Research questions → Justification → Contributions**, each contribution with its headline number | — | 1.5 h |
| 2.15 | §2 | Add the **positioning table** vs prior work (code-mixed? clinical? retrieval isolated? gold-translation pairs? reference bias checked?) | papers in `research-work/papers/` | 1 h |
| 2.16 | Figures | Replace Fig. 3 (duplicates Table 4) with **`fig3_penalty_by_depth.png`** | `results/retrieval_figures/` | 15 min |

---

## 3. What you must NOT claim

- ❌ "clinician-validated" / "clinically validated" — **no clinician reviewed anything.** The LLM-judge check is automatic cross-validation and must be described that way.
- ❌ "pre-registered" — it is a timestamped commit, not a registry.
- ❌ "H₀₂ rejected" unqualified — the two pre-specified tests disagree.
- ❌ "replicates across two generator families" unqualified — see 1.3.
- ❌ Any absolute concept-precision value as a quality measure.
- ❌ Graded nDCG as a clinical relevance judgment — it is an automatic proxy.

---

## 4. Formatting and compliance

| # | Task | Note |
|---|---|---|
| 4.1 | **Restore the template margins** | Submitted: 0.91″ left/right. Template: 1.15″. The email says "strictly" follow the template |
| 4.2 | **Page count** | Submitted 10 pages; the call says **6–8**. You said not to optimise for this, but it is stated in the decision email — at minimum check whether the chair enforces it |
| 4.3 | Reference [11] HiFactMix | Still says "under review, ICLR 2026" — decisions are long out. Check OpenReview |
| 4.4 | Author byline | "Mentor: Prof. Dr. Rajesh Maurya" is not a template field. Confirm with him whether he should be a co-author or stay in Acknowledgements |
| 4.5 | **Create the release tag** | The paper cites `v1.0-iccsdi2026`; **that tag does not exist**. Tag the final commit and publish a GitHub release |
| 4.6 | Verify the Zenodo DOI resolves | `10.5281/zenodo.22166651`, and that it holds what Data availability claims |
| 4.7 | Proofread | The decision letter asks for it explicitly. See §4 of `REVIEW-RESPONSE-PLAN.md` for the specific wording fixes |

---

## 5. Response-to-reviewers letter

Skeleton in `REVIEW-RESPONSE-PLAN.md` §7. Rules:

- Answer **every** point in 2–4 sentences: what changed, and where.
- **Quote the new abstract sentence** under R3.5 so it can be verified in seconds.
- Where you could not do what was asked, **say so and say what you did instead**:
  > *"Clinician annotation was not feasible within the revision window. We instead
  > report automatic cross-validation of the extractor against an independent
  > model (κ = 0.494) and quantify the lexicon's coverage, and we retain clinician
  > validation as an explicit limitation."*
- Lead the R3.1 answer with the **clickable commit URL**, and disclose H₀₄ yourselves.

---

## 6. Suggested order (about 8–9 hours of writing)

| Block | Tasks | Who |
|---|---|---|
| **A. Corrections** (~1.5 h) | 1.1 – 1.6 | Saachi |
| **B. Results rewrite** (~3 h) | 2.1 – 2.8 | split: Devika 2.5–2.8, Saachi 2.1–2.4 |
| **C. Limitations + new baselines** (~1.5 h) | 2.10 – 2.13 | Devika |
| **D. Framing** (~2.5 h) | 2.14 – 2.16 | Saachi |
| **E. Format + admin** (~1 h) | 4.1 – 4.7 | both |
| **F. Response letter** (~1 h) | §5 | both |
| **G. Final check** (~30 min) | `verify_paper_numbers`, read end to end | both |

---

## 7. Final check before uploading

- [ ] `python -m src.analysis.verify_paper_numbers` — every claim VERIFIED
- [ ] `python -m pytest -q` — all tests pass
- [ ] Every number in the paper traces to a file in `results/`
- [ ] Nothing in §3 ("must not claim") appears anywhere
- [ ] Every reviewer point has a paragraph in the response letter
- [ ] Margins match the template; figures legible in greyscale
- [ ] Release tag created and cited; Zenodo DOI resolves
- [ ] Both authors have read the manuscript end to end

---

## 8. Not doing, and why (state these as limitations)

| Item | Why | Where it goes |
|---|---|---|
| Clinician validation of the lexicon | No clinician access in the window | §6.1 limitation + response letter |
| Human relevance judgments | Same | §6.1; the automatic graded proxy is reported instead |
| Powered H₀₃ | ~616 queries ≈ 14 days of one organisation's quota | §5.5, §7 future work |
| Trained transliteration (IndicXlit) | fairseq does not build on Python 3.12 | §5.1 note: the hand-verified mapping is a lower bound |
| Full 3,015-query translation baseline | Daily token quota | §4.2: stratified 500-query subset, paired tests still valid |

The annotation tooling (`src/evaluation/annotation_sample.py`,
`annotation_agreement.py`, and the sheets in `results/annotation/`) stays in the
repository unused. It is ready if a clinician becomes available for the journal
extension, which is where this work belongs.
