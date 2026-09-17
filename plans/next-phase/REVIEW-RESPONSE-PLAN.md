# ICCSDI 2026 — Paper #216 — Review Response Plan

**For:** Devika Jonjale & Saachi Shinde
**Decision:** Revision required (overall recommendation: Minor Revision)
**Prepared:** 17 September 2026, from the submitted PDF, the three reviews, and the repository

---

## 0. Read this first: the deadline has passed

The email gives the revision deadline as **15 September 2026**. Today is **17 September**.

Before doing any of the work below:

1. **Log in to CMT today** and check whether the revised-manuscript upload is still open. Conference deadlines are often soft for a few days, but you cannot assume it.
2. **Email the ICCSDI 2026 chair today**, whether or not CMT is open. Keep it short: paper #216, you are revising, and you ask for a specific new date (e.g. 3–4 days). Asking for a specific short date gets a yes far more often than an open-ended request.
3. **Plan the work in two tiers** (Section 6). Tier A is writing and formatting only and fits in 1–2 days. It answers every reviewer point honestly. Tier B/C adds new analyses. Do them only if the chair gives you the time.

Everything below assumes you get a few days. If you get less, do Tier A only.

---

## 1. What the reviews actually say

### Overall
The decision is **Minor Revision**. The reviewers are not asking you to redo the study. They are asking you to (a) stop overclaiming in a few specific places, (b) report a few extra metrics, (c) make the problem, gap and novelty easier to see, and (d) fit the template. Acceptance depends on addressing *every* comment, including the vague ones.

### Reviewer 1: positive, with a wishlist
Read carefully: R1's sections 5 and 6 are **copy-pastes of the first paragraph**. There are no hidden extra requests. The whole review is one paragraph of strengths and one sentence of requests.

**What R1 praised** (keep all of these; do not weaken them while revising):
- 3,015 matched Hinglish–English query pairs
- hypotheses fixed in advance
- multiple retrievers, two generator families, statistical testing
- *"unusually transparent reporting of failed assumptions and corrected results"*
- the BM25 vs dense code-mixing asymmetry
- MuRIL at chance on romanised Hindi

**What R1 criticised:**
| # | Weakness | Asks for |
|---|---|---|
| R1.a | Absolute retrieval performance is low | (implicit) justify / contextualise |
| R1.b | Clinical relevance labels are coarse | clinician-validated relevance judgments |
| R1.c | 26-concept lexicon not clinician-validated | clinician validation |
| R1.d | Grounding benefit is highly reference-dependent | richer factuality metrics |
| R1.e | Provenance study (H03) underpowered | larger provenance experiments |
| R1.f | Hosted-model decommissioning limits reproducibility | (implicit) mitigation |
| R1.g | — | stronger **transliteration-aware baselines** |

Clarity was rated **"Average"**, the only below-par score, so clarity edits matter.

### Reviewer 2: generic, and the biggest writing job
Two sentences, no specifics. It reads like a template review, but you **must still answer it visibly**, because the chair checks that every reviewer was addressed. It asks for:
- clearer **research problem**, **research gap**, **methodology**, **contributions**, **novelty**
- **stronger justification** for the approach
- **comparisons with relevant existing methods**

The last one is the only concrete request in R2, and a reviewer will look for it. See R2 in Section 3.

### Reviewer 3: specific, careful, the most important
Six precise points. R3 read the paper closely and caught real problems. Treat every one as mandatory.

| # | Point | Type |
|---|---|---|
| R3.1 | Provide the pre-registration record/timestamp | **Integrity: the claim as written is partly false (see §3)** |
| R3.2 | Recall@1 is coarse → add Recall@k, MRR, nDCG, finer relevance | Metrics |
| R3.3 | H02 hypothesis ≠ H02 test | **Integrity: real misalignment** |
| R3.4 | Lexicon not clinician-validated | Validation |
| R3.5 | Abstract must separate reference-dependent gain from general quality gain | Framing |
| R3.6 | H03 must be explicitly provisional (13 queries) | Framing |

---

## 2. What is good and must survive the revision

The reviewers are buying the *honesty* of this paper. The biggest risk in revising is defensively smoothing over the corrections. Don't.

- The retrieval penalty (H04) is significant under every system, with the 4.4× lexical/dense asymmetry and the BM25 crossover. R1 called it valuable.
- The MuRIL script-mismatch result. R1 singled it out.
- The label-leakage gate (96.2% of summaries contain the condition label). This is a strong methodological contribution; make it more visible, not less.
- Degenerate baselines, and the proof that hallucination = 1 − precision.
- The corrected-results section. R1's phrase *"unusually transparent"* is the best line in any review.

---

## 3. Point-by-point: what to change, where, and how

Each item gives: **what they mean → is it valid → what the repo shows → project change → paper change → response-letter line.**

---

### R3.1: Pre-registration record  ⚠ highest priority

**What they mean.** You wrote *"four pre-registered null hypotheses … all of which were fixed in the project proposal before any analysis was run."* They want proof.

**Is it valid? Yes, and the claim is partly wrong.** I checked the repository history:

| Hypothesis | Earliest record | Before analysis? |
|---|---|---|
| H01 grounding | commit `0917322`, **9 Feb 2026** (README table + `plans/aims-and-challenges.md` + `research-work/group 5 - research proposal.pdf`) | **Yes.** The first prototype is `21c81df`, 4 Mar 2026 |
| H02 code-mixing | same commit, 9 Feb 2026 | **Yes**, but the planned *test* was different (see R3.3) |
| H03 provenance | same commit, 9 Feb 2026 | **Yes** |
| H04 retrieval penalty | **not in either proposal.** First appears 16 Aug 2026 (`47fbc10`, `f4105e9`) | **No.** Added in the paper-readiness phase, after retrieval experiments had already run |

Also note: the *"final updated research proposal"* (the one committed on 29 Aug) **already reports H1/H2 prototype results**. It cannot be cited as "fixed before analysis". Only the February version can.

Two more caveats you must be honest about:
- This is **not a formal pre-registration** (OSF, AsPredicted). It is a timestamped public commit. That is still real, verifiable evidence, but the word "pre-registered" implies a registry.
- The **analysis plan** changed after the instrument audit (metrics, CMI measure, per-arm tests). Some deviation is normal, but it has to be declared.

**Project change.**
- Push nothing that rewrites history. Record the evidence: the commit hash `0917322`, its date, and the GitHub URL of that commit (`https://github.com/saaachis/hinglish-grounded-medical-rag/commit/0917322`). The GitHub URL is what the reviewer can click.
- Optional but strong: deposit the **9 Feb proposal PDF** (`research-work/group 5 - research proposal.pdf`) in the Zenodo record, so it is independently timestamped.

**Paper change** (§1 Introduction and §4 opening paragraph):
- Replace every "pre-registered" with **"pre-specified"**.
- Replace *"all of which were fixed in the project proposal before any analysis was run"* with a precise statement, e.g.:
  > H01–H03 were specified in the project proposal before any data were analysed (public repository commit 0917322, 9 February 2026). H04 was added after the initial retrieval experiments and is reported as a confirmatory test of a post-hoc hypothesis. Deviations from the original analysis plan (the scoring instrument, the code-mixing measure, and the H02 test) are listed in Sect. X.
- Add a **3–5 line "Deviations from the pre-specified plan"** paragraph (can sit at the end of §4.2). One line each: metric changed from precision to F1 after the degenerate-baseline audit; CMI measure repaired; H02 planned test vs reported test; H04 added later.

**Response line.** *"We thank the reviewer. H01–H03 are timestamped in public commit 0917322 (9 Feb 2026), before the first prototype (4 Mar 2026). H04 was added later; we now say so explicitly, replace 'pre-registered' with 'pre-specified', and add a deviations paragraph (Sect. X)."*

> Why this matters so much: a reviewer who asks for the record and finds H04 is not in it will stop trusting every other claim. Saying it yourselves turns a potential rejection into a point in your favour.

---

### R3.3: H02 hypothesis vs test misalignment  ⚠ second-highest priority

**What they mean.** H02 is about the **difference** (grounded − ungrounded) across code-mixing intensity. The paper tests each arm separately. Those are different questions.

**Is it valid? Yes.** And the repo confirms the original plan was about the difference: the 9 Feb README specifies H2 as *"Two-way ANOVA / Kruskal-Wallis (CMI level interaction)"*. An interaction test **is** a test on the difference.

The paper itself admits the difference test is non-significant (§5.4: *"that difference … is non-significant"*). So under its own pre-specified test, **H02 is not rejected**. The paper currently reports "H02 rejected per arm", which is exactly the misalignment R3 spotted.

**Project change** (cached data only, no API calls, ~1–2 hours):
1. On `results/combined_h1h2/combined_scored.csv` (n = 1,165), with the **repaired** Hindi-proportion measure (`hindi_prop_v2`, same as Table 4):
   - `gain = grounded_factual − zeroshot_factual` per query
   - **Primary (pre-specified):** Kruskal–Wallis on `gain` across CMI tertiles, **plus** Spearman ρ(`gain`, `hindi_prop_v2`) with bootstrap 95% CI (continuous version, more power)
   - **Equivalence check (optional but valuable):** if the 95% CI for ρ lies entirely inside ±0.10, you can say the gain is *statistically equivalent* across code-mixing levels (TOST logic). That is the robustness claim the original H2 wanted, stated validly. If the CI crosses ±0.10, say "inconclusive" instead.
2. Keep the per-arm results, relabelled **exploratory decomposition**.
3. Save as `results/h2_per_arm/h2_prespecified_gain_test.md` and add the step to `src/analysis/reproduce_all.py`.

**Paper change** (§5.4, Table 4, abstract, §5 opening, §7):
- Restate the outcome honestly: **"H02 not rejected under the pre-specified interaction test (Kruskal–Wallis p = …, ρ = … [CI])"**.
- Then: *"An exploratory per-arm decomposition explains why: the zero-shot arm degrades with code-mixing (ρ = −0.116) while the grounded arm is flat (ρ = −0.001)…"*
- Delete the sentence *"The per-arm design is deliberately stronger than testing the grounded-minus-zero-shot gain directly"*. A reviewer will read that as justifying a switched test.
- Add the gain row to Table 4 as its **first row**, marked "pre-specified".
- §5 opening currently says *"H04, H01 and H02 are rejected"*. Update it.

**Response line.** *"The reviewer is right. We now report the pre-specified interaction test on the grounded–zero-shot difference as the primary H02 test (not rejected; Sect. 5.4), and present the per-arm analysis explicitly as an exploratory decomposition."*

---

### R3.2: Recall@k, MRR, nDCG, finer relevance

**What they mean.** Recall@1 with condition-group relevance is a blunt instrument. They want deeper-rank metrics and, where feasible, finer relevance.

**Is it valid? Yes.** Good news: **you already computed most of it.** `results/retrieval_v2/retrieval_v2_metrics.csv` contains R@1/3/5/10, MRR@10 and nDCG@10 for all four systems in both languages:

| System | Lang | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---:|---:|---:|---:|
| Hybrid-RRF | Hinglish | 0.175 | 0.460 | 0.656 | 0.299 |
| Hybrid-RRF | English | 0.197 | 0.535 | 0.739 | 0.344 |
| LaBSE | Hinglish | 0.128 | 0.430 | 0.633 | 0.258 |
| LaBSE | English | 0.149 | 0.460 | 0.678 | 0.283 |
| BM25 | Hinglish | 0.094 | 0.384 | 0.612 | 0.217 |
| BM25 | English | 0.185 | 0.550 | 0.737 | 0.338 |

The penalty holds at every depth, and **BM25's penalty grows at R@10 (+0.126)**, which strengthens the main finding.

**⚠ Two problems to fix before reporting:**
1. **The nDCG@10 in the CSV is not standard nDCG. Do not report it as is.** In `src/analysis/retrieval_v2.py` (`rank_metrics`), the ideal DCG is fixed at 1.0 and the result is clipped to [0,1]. Each condition group has hundreds of relevant cases, so the true ideal DCG@10 is Σᵢ₌₁¹⁰ 1/log₂(i+1) ≈ 4.54. The current number is a position-weighted hit indicator, not nDCG. Fix: divide by the ideal DCG computed from min(#relevant, 10), and don't clip.
2. **"Recall@k" is really a hit rate.** With many relevant cases per group, "any relevant case in top k" is Success@k (Hit@k), not recall. IR reviewers know the difference. Either rename it to **Success@k (hit rate)** throughout, or keep the name and define it once explicitly. Renaming is cleaner.

**Project change:**
- (a) **Must, ~1 hour:** fix nDCG, rerun `retrieval_v2.py` from cached embeddings, and add **per-k penalty tests** (McNemar at each k, bootstrap CI for MRR penalty). Also compute the **missing TF-IDF McNemar p**, currently shown as "—" in Table 1 for no stated reason.
- (b) **Should, ~1–2 hours:** a **graded automatic relevance** proxy. Grade 2 = same condition group *and* shares ≥ 1 lexicon concept with the query's gold English summary (caption-stripped); grade 1 = same group only; grade 0 = otherwise. Report graded nDCG@10. **Label it clearly as an automatic proxy**, not a clinical judgment. `src/evaluation/relevance.py` already has the multi-label machinery to extend.
- (c) **Could, 1–2 days of annotation:** a **human-judged sample**. See the shared annotation plan in R3.4. This is the only honest answer to "finer clinical relevance judgments".

**Paper change** (§3.1 definitions, §4.2 metrics, Table 1, §6.1):
- Extend **Table 1** with **Success@10** and **MRR@10** columns (Hinglish / English / penalty) instead of adding a table. Page budget matters (§5).
- One sentence in §5.1: the penalty holds at every depth, and BM25's grows to +0.126 at depth 10.
- One sentence in §6.1: relevance remains group-level; graded proxy and (if done) human-judged sample results.

**Response line.** *"We now report Success@k (k = 1, 3, 5, 10), MRR@10 and nDCG@10 with graded relevance (Table 1, Sect. 5.1); the penalty holds at all depths. [If done:] A double-annotated sample of N query–case pairs provides finer relevance judgments (κ = …)."*

---

### R3.4 / R1.c: Clinician validation of the 26-concept lexicon

**What they mean.** Factuality conclusions rest on a 26-concept lexicon that nobody with clinical training has checked.

**Is it valid? Yes.** It is already listed as a limitation, but R3 says validation *"would considerably strengthen"* the conclusions, and R1 asks for it outright.

**Realistic options, best first:**

| Option | What | Effort | Honest claim |
|---|---|---|---|
| **A. One or two clinicians** (an MBBS family doctor, a relative, a friend in medical college, a doctor NMIMS knows) | (i) review the 26 concepts plus their Hinglish synonym and negation lists for omissions and errors; (ii) mark 100 sampled answers: which concepts are asserted, which negated | ~2–3 hours of clinician time | "clinician-reviewed lexicon; extractor agreement with clinician P = …, R = …, κ = …" |
| **B. Author double-annotation + UMLS mapping** | both authors independently annotate the same 100 answers; map each concept to a UMLS CUI / SNOMED CT code | ~1 day | "vocabulary mapped to standard terminology; extractor–human agreement κ = …; clinician validation remains future work" |
| **C. Writing only** | strengthen the limitation | ~15 min | limitation stated; nothing new |

**Recommendation.** Do **A if any clinician can give you 2–3 hours this week**. It directly answers two reviewers. Otherwise do **B**. Never write "clinically validated" unless A was done.

**One annotation effort can answer four reviewer points.** Design the 100-answer sample once:
- Sample 100 query IDs stratified by arm (grounded / zero-shot) and by code-mixing tertile.
- For each answer, the annotator marks asserted and negated concepts, and optionally rates the answer 0/1/2 for clinical appropriateness against the gold English summary.
- For the same 100 queries, the annotator rates the Hybrid top-3 retrieved cases 0/1/2 for clinical relevance.

This one sheet gives you lexicon validation (R3.4, R1.c), a richer human factuality metric (R1.d), finer relevance judgments (R3.2, R1.b) and inter-annotator κ. Build it as a CSV with blinded system and language labels.

**Project change:** `src/evaluation/annotation_sample.py` draws the stratified sample and writes the blinded sheet; `src/evaluation/annotation_agreement.py` computes κ and extractor P/R/F1 against the human labels. Store the sheets in `results/annotation/`.

**Paper change:** §4.2 (one sentence on validation procedure), §5.2 (extractor–human agreement next to the degenerate baselines, since both are validity evidence), §6.1 (update limitation 2).

---

### R3.5: Abstract must separate reference-dependent gain from general quality

**What they mean.** The abstract says *"Grounding significantly improves factual support"* and only then qualifies it. A skimmer takes away "grounding improves quality", which your own results do not support under the unbiased reference.

**Is it valid? Yes.** And there is a second, related clarity problem nobody flagged yet: **the same condition changes sign between Table 3 and Fig. 2.**
- gpt-oss-120b real retrieval: **+0.224 precision** (Table 3) but **−0.032 F1** circular and **−0.047 F1** unbiased (Fig. 2).

A careful reader sees a condition that "improves" in one table and "worsens" in the next figure, with the switch in metric *and* reference buried in prose. This is likely what drove R1's "highly reference-dependent" remark and the "Average" clarity score.

**Paper change:**
- **Abstract.** Rewrite the grounding sentence so the qualifier comes first, e.g.:
  > Scored against the evidence it was given, grounding significantly increases concept precision, replicating across two generator families. Scored against a reference neither arm saw, the gain shrinks (llama-3.1-8b, oracle: +0.062 F1) or disappears (gpt-oss-120b: n.s. or negative), so we report it as a reference-dependent effect, not a general improvement in answer quality.
- **H01 outcome wording** (§5 opening, §5.3, §7): "H01 rejected **under the evidence-referenced protocol**", never unqualified.
- **§5.3 / Table 3 + Fig. 2.** Merge into **one compact 2×2 view**: rows = generator/evidence; columns = *precision vs evidence*, *F1 vs evidence*, *F1 vs unbiased*. The sign pattern is then visible in one row. This also saves space (§5): Fig. 2 can go.
- **§7 Conclusion.** Replace *"grounding measurably improves factual support"* with the same qualified wording.
- **Terminology.** H01 in §1 says **"factual consistency"**; the metric is **"factual support"** (concept precision). Use one term.

**Response line.** *"We rewrote the abstract, §5.3 and §7 so the grounding effect is stated as reference-dependent: significant against the evidence reference, reduced or absent against the unbiased reference. Table 3 now shows both references side by side."*

---

### R3.6 / R1.e: H03 underpowered, must be provisional

**What they mean.** 13 of 160 queries is not a basis for conclusions.

**Is it valid? Yes.** There is also a hypothesis–test misalignment here, the same kind R3 caught for H02, which they did not mention but a second reader will:
- **H03 is about factual correctness.** The only significant result (Cochran's Q = 9.09, p = 0.028) is about **refusal rate**, a different outcome.
- So **on its own stated outcome, H03 is untested** (inconclusive), and the refusal effect is a **secondary outcome**. "Partially answered" blurs that.

**Project change.**
- **For this revision: none.** Powering H03 needs roughly 1.5M generation tokens (see `plans/next-phase/HANDOVER-CODE-COMPLETE.md` §3). That is 7+ days of one Groq organisation's daily quota. Don't attempt it before the revision.
- **For the journal extension:** run `python -m src.analysis.h3_provenance --prompt direct --model openai/gpt-oss-120b --out-dir results/h3_direct --n-queries 400` using keys from *different* accounts or another provider. It resumes from checkpoint.
- **Cheap, useful now (~30 min):** a **power statement**. With 76–88% refusal, compute how many queries are needed for ≥ 50 jointly scoreable queries, and put that number in the paper as the design requirement for a follow-up.

**Paper change** (§5.5, abstract, §5 opening, §6.1, §7):
- Retitle §5.5 to **"Evidence Provenance (H03): Exploratory"**.
- State plainly: *"H03's primary outcome (answer quality) could not be tested adequately: only 13 of 160 queries were scoreable under all four conditions. H03 is therefore neither rejected nor supported. As a secondary outcome, refusal rates differ across evidence types (Q = 9.09, p = 0.028)."*
- Abstract: end the H03 sentence with **"; this result is provisional"**.
- Add the power statement to §6.1 and §7.

**Response line.** *"Agreed. H03 is now labelled exploratory throughout; we state that its primary outcome remains untested, present the refusal effect as a secondary outcome, and give the sample size a powered follow-up requires."*

---

### R1.g + R2: Transliteration-aware baselines and comparisons with existing methods

**What they mean.**
- R1: your own MuRIL result says script mismatch is the mechanism, and the paper says transliteration is *"a hypothesis the study motivates but does not test"*. Test it.
- R2: *"include adequate comparisons with relevant existing methods"*. The standard existing answers to code-mixed queries are **translate-then-retrieve** and **transliterate-then-encode**. Neither is in the paper.

These two requests have **one answer**: add two baseline families to Table 1.

**Is it valid? Yes.** It is also the single most valuable experiment for the revision, because it turns the paper's mechanism claim from motivated to tested.

**Project change.** New script `src/analysis/transliteration_baselines.py`, reusing the pairing and McNemar code from `retrieval_v2.py`:

| Baseline | How | Cost | Why it matters |
|---|---|---|---|
| **1. Transliterate → MuRIL** | Convert romanised Hinglish query words to Devanagari (AI4Bharat **IndicXlit**; *verify install on your machine*), leaving English words (dictionary check) unchanged. Encode the **3,015 queries only** against the *existing* MuRIL single-vector case index | **Cheap**: re-encode 3,015 short queries on CPU, index already built | **Directly tests the paper's script-mismatch hypothesis.** If MuRIL jumps from 0.064 toward its English 0.182, the mechanism is confirmed, not just motivated |
| **2. Translate → retrieve** | Hinglish → English with an LLM (one short prompt per query), then BM25, LaBSE and Hybrid on the translation | 3,015 × ~150–200 tokens ≈ 0.5–0.6M tokens. Groq's limit is per organisation *per model*, so use gpt-oss-20b or spread over models; or run on a **stratified 1,000-query subset** (paired tests remain valid) | **The "existing method" comparison R2 asks for.** Also answers practitioners' obvious question: why not just translate? |
| 3. Romanised-Hinglish encoder *(optional)* | A sentence encoder trained on romanised Hinglish (L3Cube's Hing- models; *verify an embedding variant exists on HuggingFace before planning on it*) | Re-encode 41,746 passages on CPU: likely hours, run overnight | Stronger transliteration-aware dense baseline |

**Order:** 1 → 2 → 3. If time is short, do **1 only**. It is cheap and closes an open claim.

**Paper change.**
- Table 1: add rows *"MuRIL, transliterated input"* and *"Translate → Hybrid"* (or *→ BM25*).
- §5.1: 2–3 sentences on what translation and transliteration recover.
- §6 Discussion: replace *"this is a hypothesis the study motivates but does not test"* with the result.
- §4.3: the translation model name, prompt summary and date (hosted-model caveat again).

---

### R2: Problem, gap, methodology, contributions, novelty, justification

**What they mean.** A reviewer read the paper and could not quickly say what problem it solves, what is new, or why the design is right. R1's "Average" clarity score points the same way.

**Is it valid? Partly.** The content is there but scattered. The introduction mixes the patient example, the RAG premise, the "measurement paper" framing, the hypotheses and the list of corrections in one flow. The contributions list arrives after the corrections paragraph, so novelty lands late.

**Paper change** (writing, no data):
1. **§1, restructure into five labelled beats** (bold run-in heads are allowed in the template and cost no space):
   - **Problem.** One paragraph: romanised code-mixed patient queries vs English clinical evidence; nobody knows how much retrieval and grounding degrade.
   - **Gap.** Three bullets: (i) code-mixed queries never compared with gold translations of the *same* question; (ii) grounding benefits scored against the evidence the grounded arm saw; (iii) no hypothesis-driven test across generators.
   - **Research questions.** RQ1–RQ3 mapped to H04, H01/H02 and H03 in one line each.
   - **Approach and justification.** Why a *measurement* study (the goal is to quantify, so standard components are deliberate); why paired Hinglish–English gold translations (they isolate surface form from content); why a label-leakage gate; why two references.
   - **Contributions.** A numbered list of 3, each with its **headline number** (4.4× penalty asymmetry; MuRIL at chance → transliteration recovery; the reference-sensitivity of grounding with degenerate baselines).
2. **§2 Related Work: add one compact positioning table** (about 6 rows × 5 ticks) comparing MedSumm, HealthAlignSumm, Xiong et al., MMed-RAG, CroCoSum, HiFactMix and *this work* on: code-mixed queries · clinical domain · retrieval stage isolated · gold-translation pairs · reference bias checked. This is the most space-efficient way to show the gap and the existing-methods comparison at the design level. **Fill each tick only from what each paper actually does.** The PDFs are in `research-work/papers/`.
3. **§3.2 Architecture, add a small pipeline and evaluation-design figure** (query pair → leakage gate → 4 retrievers → oracle/real evidence → two generators → two references). Pay for it by removing Fig. 3 (it duplicates Table 4).
4. **Justify each design choice in half a sentence** where it is introduced: LaBSE (trained on translation pairs, so cross-lingual alignment is its objective), BM25 and RRF (standard lexical baseline and parameter-free fusion), MultiCaRe (the Open-i attempt yielded 11 pairs), the image description as reference (neither arm saw it).
5. The existing-methods **experiments** are the transliteration and translation baselines above.

**Response line.** *"We restructured §1 into problem, gap, research questions, justification and contributions; added a positioning table against prior work (§2) and a pipeline figure (§3.2); and added translate-then-retrieve and transliteration baselines as comparisons with existing approaches (Table 1)."*

---

### R1.a: Absolute retrieval performance is low

**Valid, and already a limitation.** Contextualise it rather than defending it:
- Success@10 for Hybrid on Hinglish is **0.656**. As a first-stage candidate generator for a reranker the pipeline is usable, even though top-1 is not. One sentence in §5.1 and §6.1.
- Restate that Recall@1 is a **lower bound** under group-level relevance, and give the graded or human-judged numbers if done.
- Keep the §6.1 statement that no clinical deployment claim is made.

---

### R1.f: Hosted-model decommissioning limits reproducibility

**Valid, and already disclosed.** Add mitigations, not more apology:
- **Archive completeness.** All raw generations are in the repository and Zenodo. Every number is **re-scorable** without the retired model.
- **gpt-oss-120b is open-weight.** Its results are **regenerable** in principle on self-hosted hardware. Say which results are *regenerable* vs *re-scorable only*: one sentence, or two columns in a small table in §6.1.
- **⚠ Fix the release references before resubmitting.** The paper cites **release tag `v1.0-iccsdi2026`**, and that tag **does not exist** (`git tag -l` and `git ls-remote --tags origin` both return nothing). Also confirm that `https://doi.org/10.5281/zenodo.22166651` resolves and contains the files the paper says it does. After revising, create the tag (e.g. `v1.1-iccsdi2026-revised`) on the exact commit, publish a GitHub release, and put *that* tag in the paper.

---

## 4. Problems the reviewers did not raise, but a camera-ready check will

Fix these while you're in the document. Each is quick.

| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | §5.3, §5.4 | **"Repaired lexicon" means two different things.** Table 3 means the *26-concept clinical lexicon* (negation repair); Table 4 and §5.4 mean the *Hindi-word list for the code-mixing measure* (doctor/please repair). §5.3 even says H01 *"survives the lexicon repair of Sect. 5.4"*, which points to the wrong repair | Name them distinctly: **"concept lexicon (negation-repaired)"** vs **"Hindi-proportion measure (repaired)"** |
| 2 | Table 3 vs Table 4 | Table 3 "repaired lexicon" row has **n = 669**; Table 4 has **n = 1,165**. Unexplained | Footnote: 669 = rows where both arms assert ≥ 1 scoreable concept under the repaired extractor |
| 3 | §5.6 vs §4.1 | §4.1: median case **554 words**. §5.6: *"truncating 99.9% of case narratives (median 300 tokens)"*. 554 words is well over 300 tokens | Check which text the 300-token median refers to (likely the 200-word-capped input). State it, or correct the number |
| 4 | §5.3 | *"Oracle evidence is superior to real retrieval by +0.0875 (p = 0.041)"*: metric and reference not stated | Add "(concept F1, unbiased reference)" or whichever it is. Check in `results/h1_real_120b/` |
| 5 | Table 1 | TF-IDF penalty and McNemar p shown as "—" with no reason | Compute it (same script); a gap in the main table invites questions |
| 6 | Byline | *"Mentor: Prof. Dr. Rajesh Maurya"* under the authors. The template byline is authors and affiliations only | Remove from the byline; he is already thanked in Acknowledgements. *(Check with him and your department first; some institutions expect the mentor as a co-author instead)* |
| 7 | Ref [11] | HiFactMix *"Under review, ICLR 2026"*. ICLR 2026 decisions are long out | Look it up on OpenReview; cite the published version, the arXiv version, or remove it |
| 8 | §1 | *"generate an answer that is grounded in the patient's own style"* is muddled | *"…generate an answer grounded in that case, in the patient's own language"* |
| 9 | §5.6 | *"F irst dense index was built"* is missing an article | *"The first dense index was built"* |
| 10 | Throughout | Hallucination = 1 − precision is stated **three times** (§4.2, §5.2, §6) | Once, in §5.2 |
| 11 | Throughout | *"Rejecting a null is the positive outcome"* stated twice (§1, §4) | Once, in §4 |
| 12 | Abstract | "pre-registered", "absorbs the degradation", "H02 rejected" framing | Update per R3.1, R3.3, R3.5, R3.6 |

---

## 5. Formatting: the paper is over length and off-template

Both of these will block acceptance even if every comment is answered.

### Page limit: **10 pages submitted, limit 6–8**
The email is explicit: *"Papers … should be 6–8 pages."* You need to lose at least 2 pages **and** absorb the additions above.

### Margins: **non-compliant**
| | Top | Bottom | Left | Right |
|---|---:|---:|---:|---:|
| ICCSDI template | 1.05″ | 0.75″ | 1.15″ | 1.15″ |
| Submitted paper | 0.98″ | 0.98″ | **0.91″** | **0.91″** |

Left and right were narrowed by about 0.24″ each (the "reduce margins slightly" change). The email says *"strictly according to the official conference guidelines"*. **Restore the template margins.** That narrows the text block by about 7% and makes the length problem slightly worse. Budget for it.

### Where the 2+ pages come from

| Cut | Saves (approx.) | Loses anything? |
|---|---:|---|
| Drop **Fig. 3** (it duplicates Table 4) | 0.40 p | No. Replace with the pipeline figure only if space allows |
| Merge **Table 3 + Fig. 2** into one 3-column table (R3.5) | 0.45 p | No, it is clearer |
| **§5.6 Negative results** → one 8–10 line paragraph; full detail moves to the journal version | 0.35 p | Detail only |
| **Table 2** degenerate baselines: 8 rows → 4 (verbatim copy, "swelling", grounded, zero-shot) | 0.15 p | No |
| Remove repeated statements (§4 items 10, 11) and the restated contributions in §2's last paragraph and §7 | 0.35 p | No |
| **§6 Discussion** "evaluation findings" list → 3 lines | 0.15 p | No |
| **§4.1** Open-i pairing story → 2 sentences | 0.10 p | No |
| **§4.3** implementation: compress the library list into one line | 0.10 p | No |
| **Total** | **≈ 2.05 p** | |

**Additions to absorb:** deviations paragraph (~0.15 p), H02 primary test row (~0.05 p), Success@10/MRR columns (~0.05 p), baseline rows (~0.1 p), positioning table (~0.3 p), restructured intro (net ~0), pipeline figure (~0.3 p, optional). So the target is **8 pages exactly**. If it still runs over, drop the pipeline figure first and shrink the positioning table second.

**Journal extension.** The email invites 8–15 page extended versions of selected papers. Everything cut above (full negative results, the complete degenerate-baseline table, Fig. 3, the MuRIL comparison detail, H03 at full power) becomes the core of that extension. Keep a `research-paper/journal-extension-material.md` as you cut, so nothing is lost.

---

## 6. Work plan, in priority order

### Tier A: must do, writing and formatting only (~1.5 days, both authors)
| # | Task | Owner suggestion |
|---|---|---|
| A1 | Email the chair / check CMT (§0) | whoever submitted (Saachi) |
| A2 | Pre-registration fix: "pre-specified", commit evidence, H04 disclosure, deviations paragraph (R3.1) | Saachi |
| A3 | Abstract, §5.3, §7: reference-dependent framing; merge Table 3 + Fig. 2 (R3.5) | Devika |
| A4 | H03 → exploratory, secondary outcome, provisional (R3.6) | Devika |
| A5 | Intro restructure + contributions with numbers + positioning table (R2) | Saachi |
| A6 | All §4 camera-ready fixes | Devika |
| A7 | Restore template margins; make the §5 cuts; hit ≤ 8 pages | both, last |
| A8 | Fix the release tag + verify the Zenodo DOI (R1.f) | Saachi |
| A9 | Response-to-reviewers document (§7) | both |

### Tier B: cheap computations from cached data (~1 day, no API calls)
| # | Task | Answers |
|---|---|---|
| B1 | **H02 pre-specified gain test** (Kruskal–Wallis + Spearman + equivalence bound) | R3.3 ⚠ *Treat this as Tier A in practice: the H02 text cannot be written honestly without it* |
| B2 | Fix nDCG; add per-k penalty tests; TF-IDF McNemar; rename to Success@k | R3.2 |
| B3 | Graded automatic relevance (condition + concept overlap) | R3.2, R1.b |
| B4 | H03 power statement | R3.6, R1.e |
| B5 | Resolve the 300-token vs 554-word and n = 669 inconsistencies; identify the +0.0875 metric | camera-ready |

### Tier C: new experiments, only if the deadline allows
| # | Task | Effort | Answers |
|---|---|---|---|
| C1 | **Transliterate → MuRIL** | ~half a day | R1.g, R2, and tests your own mechanism claim. **Highest value in this tier** |
| C2 | **Translate → retrieve** (full or 1,000-query subset) | 1–2 days (API quota) | R2 "existing methods" |
| C3 | 100-item annotation sheet (answers + top-3 relevance), κ, extractor agreement; clinician if available | 1–2 days | R3.4, R1.c, R1.d, R3.2, R1.b |
| C4 | Romanised-Hinglish encoder | overnight CPU | R1.g (optional) |

### Tier D: don't do now; state as limitation or journal extension
- Powered H03 (1.5M tokens)
- Large-scale clinician relevance judgments
- Clinician validation beyond a single reviewer

> **If you only have 2 days:** A1–A9 + B1 + B2. That answers every comment honestly, and nobody can say a point was ignored. The strongest single upgrade after that is **C1**.

---

## 7. Response-to-reviewers document: skeleton

Submit this alongside the revised paper. Reviewers and chairs read it first.

```
Response to Reviewers — ICCSDI 2026 Paper #216

We thank the reviewers for their careful reading. Changes in the manuscript are
marked [or: summarised below with section references]. Page count is now 8,
formatted in the official ICCSDI 2026 template with its original margins.

REVIEWER 1
R1.1 Coarse relevance labels — [what changed, where]
R1.2 Lexicon validation — [...]
R1.3 Reference-dependent grounding benefit / richer factuality metrics — [...]
R1.4 Transliteration-aware baselines — [...]
R1.5 Provenance study underpowered — [...]
R1.6 Model decommissioning — [...]
R1.7 Low absolute retrieval performance — [...]

REVIEWER 2
R2.1 Problem, gap, methodology, contributions, novelty — [...]
R2.2 Justification of approach — [...]
R2.3 Comparison with existing methods — [...]

REVIEWER 3
R3.1 Pre-registration record — [commit 0917322 link; H04 disclosure; deviations]
R3.2 Recall@k / MRR / nDCG / finer relevance — [...]
R3.3 H02 hypothesis–test alignment — [...]
R3.4 Clinician validation — [...]
R3.5 Abstract distinction — [quote new sentence]
R3.6 H03 provisional — [...]

Additional corrections made on re-reading: [list from §4 of the plan]
```

Rules for writing it:
- **Answer each point in 2–4 sentences:** what you changed, and where (section, table).
- **Where you could not do what was asked, say so and why** (e.g. no clinician access in the revision window), then say what you did instead. Reviewers accept "not feasible, here is the mitigation" far more readily than silence.
- **Quote the new abstract sentence** under R3.5. It lets the reviewer verify in 5 seconds.
- **Never claim a change that isn't in the manuscript.** Chairs spot-check.

---

## 8. Files this plan touches

| Area | File |
|---|---|
| H02 gain test | new: `src/analysis/h2_prespecified_gain_test.py` → `results/h2_per_arm/h2_prespecified_gain_test.md` |
| Retrieval metrics | edit: `src/analysis/retrieval_v2.py` (`rank_metrics` nDCG; per-k tests) → `results/retrieval_v2/` |
| Graded relevance | edit: `src/evaluation/relevance.py` |
| Transliteration / translation baselines | new: `src/analysis/transliteration_baselines.py` → `results/translit_baselines/` |
| Annotation | new: `src/evaluation/annotation_sample.py`, `src/evaluation/annotation_agreement.py` → `results/annotation/` |
| H03 power | new: small section in `src/analysis/h3_provenance.py` or a standalone script |
| Reproducibility | edit: `src/analysis/reproduce_all.py` (register the new stages) |
| Paper | `research-paper/research paper - devika jonjale & saachi shinde.docx` → save revision as a **new file** (`…-revised.docx`) so the submitted version stays intact |
| Cut material | new: `research-paper/journal-extension-material.md` |
| Response | new: `research-paper/response-to-reviewers.docx` |
| Release | git tag + GitHub release; Zenodo new version |
