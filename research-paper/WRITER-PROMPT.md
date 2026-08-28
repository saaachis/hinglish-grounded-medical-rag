# Prompt to hand to a writing model (Claude web / Fable)

Copy everything inside the box below into a new Claude conversation, and attach the
six files listed in §0 of `PAPER-BRIEF-FOR-WRITER.md`.

---

```
You are writing a complete conference manuscript for submission. I have attached a
research brief containing every measured result, and the official Word template the
paper must use.

## The task

Write the full manuscript, ready for submission. Fill the attached Word template in
place so its styles, fonts, page size and title-block formatting are preserved — do
not rebuild the document from scratch, and do not change the layout beyond the
margins already set.

## Venue

ICCSDI 2026 — International Conference on Computational Science and Data
Intelligence, 11–12 December 2026, NMIMS Mumbai, India. Springer SN Computer Science
template (attached).

## Authors and affiliation

Devika Jonjale and Saachi Shinde
Both authors contributed equally. Author order is alphabetical — keep it that way.

Affiliation for both:
  M.Sc. Data Science
  NMIMS Nilkamal School of Mathematics, Applied Statistics & Analytics
  Mumbai, India

Emails:
  devikajonjale04@gmail.com
  saachi.shinde28@gmail.com

Mark both as corresponding authors and include a footnote reading:
"Both authors contributed equally to this work. Author order is alphabetical."

## What is attached

1. PAPER-BRIEF-FOR-WRITER.md — every measured number, all caveats, and an explicit
   list of claims that must not be made. This is your only source of facts.
2. ICCSDI2026_Word_Template (1).docx — the required format.
3. fig1_table1.png — retrieval quality by system and query language.
4. fig2_penalty.png — the code-mixing penalty per retrieval system, with CIs.
5. fig3_h1_reference_effect.png — the grounding effect under two scoring references.
6. h2_dose_response.png — factual support against code-mixing intensity, by arm.

## Required structure (follow the template's numbering)

  Title, authors, affiliation, abstract, keywords
  1  Introduction
  2  Related Work and Research Gap
  3  Materials and Methods
     3.1  Problem Formulation
     3.2  Proposed Architecture or Workflow
  4  Experimental Design and Evaluation
     4.1  Datasets and Preprocessing
     4.2  Baselines and Evaluation Metrics
     4.3  Implementation Details
  5  Results          — organise by hypothesis: H04, metric validity, H01, H02, H03
  6  Discussion
     6.1  Limitations and Threats to Validity
  7  Conclusion and Future Work
  Declarations (Funding, Competing interests, Ethics, Consent, Data availability,
                Materials availability, Code availability, Author contributions)
  References

## Hard constraints

- Use ONLY numbers that appear in the brief. Do not invent, round differently, or
  extrapolate any statistic.
- Do not invent citations. Where a citation is needed, insert a clearly marked
  placeholder such as [CITATION NEEDED: MMCQSD dataset paper]. The bibliography is
  the authors' to complete.
- Obey the brief's section "What must NOT be claimed" in full. In particular: never
  present the hallucination result as independent of the factuality result; never
  quote an absolute concept-precision value as a measure of answer quality; never
  claim adaptive context selection works; never describe the system as multimodal.
- Every limitation listed in the brief must appear in Section 6.1.
- Leave the brief's listed placeholders (library versions, dataset DOI, repository
  URL, acknowledgements) clearly marked for the authors to fill.

## Framing

This is a measurement study, not a system-improvement paper. Several of the
project's original claims did not survive scrutiny, and reporting those corrections
honestly is the contribution rather than an embarrassment. The strongest results are
on the retrieval side. Lead with them.

Two points that readers reliably misunderstand, so handle them explicitly:

- "Rejecting" a null hypothesis is the positive outcome. In the hypotheses table,
  add a plain-language gloss for each row, e.g. "H01 — Rejected — i.e. grounding
  significantly improves factual support."
- The paper reports a grounding benefit AND shows its measured size depends on the
  scoring reference. These are not contradictory. Present the second as a
  methodological caution attached to the first, not as a retraction of it.

## Tone

Formal academic register for a Springer computer-science venue. Precise and
measured. State findings plainly and let the statistics carry the weight — no
overselling, no hedging where the evidence is firm. Prefer active voice. British or
American spelling, applied consistently.

## Deliverable

The completed .docx, built on the attached template, with all figures placed and
captioned, both tables rendered, and every placeholder clearly marked.
```

---

## If the writing model cannot edit .docx directly

Ask it for the full manuscript as formatted Markdown instead, then paste section by
section into the template. Keep the template's heading styles (`ICCSDI Heading 1`,
`ICCSDI Heading 2`, `ICCSDI Caption`, `ICCSDI Bullet`) rather than Word's defaults —
this is what keeps the submission looking like the template.

## Follow-up prompts worth having ready

- "Tighten the abstract to 250 words without losing any statistic."
- "Rewrite Section 6.1 so each limitation states its consequence for interpretation,
  not just its existence."
- "Check every number in the manuscript against the brief and list any that do not
  match exactly."
