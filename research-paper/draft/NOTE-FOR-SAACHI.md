# Final review before submission — note for Saachi

**Branch:** `devikas-hardening` (pushed) · **Commits:** `ad75d4d`, `e1fee9f`
**Files:** `research-paper/draft/ICCSDI2026_final_named.docx` and
`ICCSDI2026_final_anonymised.docx` — both exactly 10 pages.

---

## What's done

Zenodo is deposited and live: **https://doi.org/10.5281/zenodo.22166651** (public,
CC-BY 4.0). I fetched the record to confirm it's correctly titled and attributed,
and cross-checked completeness — Zenodo reports `data.zip` at 266.1 MB, and a
control zip I built locally from the same eight files came to 253.8 MB. Consistent,
so nothing was left out of the upload.

Every number in the paper was traced back to its source result file. **30 of 30
matched exactly** — H01 deltas and sample sizes, all four H02 correlations, all
eight H03 refusal and F1 values, the six degenerate baselines, the six
reference-effect values, and the truncation sweep.

All five DOIs were resolved and confirmed to open the correct paper, and all four
citations that had been written from memory were independently verified against
the source: LaBSE (ACL 2022, pp. 878–891), RRF (SIGIR 2009, pp. 758–759), BM25
(FnTIR 3(4), 333–389), MuRIL (arXiv:2103.10730, 14-author order). Citations run
1–14 contiguously, every one is cited in the text, none is orphaned.

Five URLs — including the new Zenodo DOI — were sitting in the document as plain
text and were **not clickable**. They're now real hyperlinks; Word reports 8 live
links in the named draft and 4 in the anonymised one. I diffed the text before and
after that surgery and it's byte-identical, so nothing was disturbed.

---

## 🔴 Red flag 1 — the hybrid retrieval result may not reproduce

**In plain terms.** Table 1 says all three retrieval systems get significantly
worse on Hinglish. When I re-ran the pipeline from scratch, two of the three came
out the same, but the **hybrid (RRF)** system did not: its p-value moved from
**0.018 to 0.074**. Since 0.05 is the conventional cut-off for "significant," that
one result crossed the line from significant to not.

**Technically.** The cause is diagnosed, not guessed. In the lexical rankers,
**6.8% of Hinglish queries and 9.0% of English queries have a tie at the top TF-IDF
score** — 204 and 272 queries respectively. `np.argsort` uses an unstable sort, so
which document wins a tie is arbitrary and can differ between runs or machines.
Separately, `TfidfVectorizer(max_features=200_000)` is a *binding* cap (the
vocabulary landed on exactly 200,000), so the feature cutoff is itself tie-broken
and can shift across scikit-learn versions. Both feed Recall@1, and the hybrid sits
close enough to the threshold that a few flipped queries move it across.

I ruled out the boring explanations: the outputs weren't stale (fresh timestamps),
and the corpus hadn't drifted — `evidence_metadata.csv` is **MD5-identical** to your
March handoff, and `passages.parquet` aligns with it on all 10,000 cases.

**What this does and doesn't threaten.** BM25 (p ≈ 1×10⁻²⁴) and LaBSE-passages are
nowhere near the threshold. The headline **4.4× lexical/dense asymmetry is
unaffected**, as is every generation result. Only the hybrid system's individual
significance claim is fragile.

**Decision taken:** Table 1 and Sect. 5.1 are unchanged. The committed numbers came
from a legitimate run against the committed artifacts, and the `passage_emb.npy`
cache used for my re-run was never in the handoff zips, so I can't prove it's the
same cache that produced Table 1. Full evidence is in
`results/_reproducibility_check/`.

**The residual risk, stated plainly:** Sect. 5.1 claims *"largest p = 0.018"* and
*"H04 is rejected for every system tested."* A reviewer who re-runs the code in a
different environment could get p > 0.05 for the hybrid and read that as an
overclaim. If you want it airtight, the fix is deterministic tie-breaking (a stable
sort, or an explicit secondary key on document index) in
`src/analysis/retrieval_v2.py` — but that changes Table 1 and forces Fig. 1 to be
regenerated, so it isn't a small edit this close to the deadline. Your call.

---

## 🟠 Red flag 2 — the review policy is still unconfirmed

I could not determine whether ICCSDI 2026 is double-blind. The conference site is
JavaScript-rendered and every fetch returned only the deadline banner, never the
author guidelines. **Please check the Microsoft CMT submission form** — it usually
states the review model on the track description.

This decides which file we submit, so it matters:

- **Single-blind / not blind →** `ICCSDI2026_final_named.docx`
- **Double-blind →** `ICCSDI2026_final_anonymised.docx`

**One thing to be careful about if it is double-blind.** The Zenodo record carries
both our names. That's why the anonymised draft deliberately does *not* cite the
DOI — it says the DOI is withheld for review — and why the GitHub URL is withheld
too. Citing either would deanonymise us in one click. Worth knowing that a
determined reviewer could still search Zenodo for the title and find it; if the
conference is strict, consider making the Zenodo record restricted until after
decisions.

The anonymised file was scrubbed at the package level, not just the visible text.
The emails and repo URL had survived my first pass as **hyperlink relationship
targets** — invisible in Word, but plainly readable to anyone who unzips the file.
It now scans clean for all twelve identifying terms in both the text and the raw
XML.

---

## 🟡 Minor things, already fixed

- The paper's truncation figures were only ever recorded in a docstring, never
  computed by any script. I re-measured them: case narratives are **median 300
  tokens with 99.9% truncated** (the draft said 307 / 100%), and **60.0% of
  queries** were truncated (the draft said 52.5%). The 2.4× BM25 information
  advantage checks out at 2.34–2.38×. Corrected in the paper; the direction of the
  argument is unchanged and slightly strengthened.
- A dangling cross-reference: after tables were renumbered, one sentence still
  pointed at "Table 1" for the hypotheses summary, which by then was the retrieval
  table. Fixed.
- Signs are consistent: 25 true minus signs (U+2212), no ASCII hyphens used as
  minus, en dashes only in numeric ranges. The 8 em dashes are all "not applicable"
  markers inside tables — none in prose.

---

## Still open for you

1. **Confirm the review policy** and pick the matching file.
2. **Decide on the hybrid claim** (leave as-is, or soften Sect. 5.1).
3. **Run the manuscript through Turnitin** via the institutional account. I can't
   produce a similarity or AI-detection score — those are proprietary systems, and
   any number I gave you would be invented.
