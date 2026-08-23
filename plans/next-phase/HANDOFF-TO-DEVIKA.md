# Handoff to Devika — Your Three Questions Answered, H₀₄ Run, Five Items Closed

**From:** Saachi's machine · 2026-08-24 · branch `saachi-hardening` (pushed)
**Read order:** this file → `HARDENING-REPORT.md` → `PAPER-READINESS-NEXT-STEPS.md`

---

## 1. Your three review questions — all checked, and you were right on both counts

### 1.1 The hallucination retraction — ✅ accepted

I recomputed it from source rather than reading your CSV, and reproduced your numbers
exactly. `doctor` fires in 68.2% of queries and `please` in 35.7%, both sitting in the
legacy Hindi list. Under `hindi_prop_v2`:

| Arm | Legacy ρ / p | Repaired ρ / p | |
|---|---|---|---|
| Grounded factual | +0.0149 / 0.612 | −0.0006 / 0.983 | survives (flat) |
| Zero-shot factual | −0.0677 / 0.021 | **−0.1155 / 0.000077** | survives, ~2× stronger |
| Grounded hallucination | +0.0610 / 0.037 | −0.0224 / 0.445 | ❌ withdrawn |
| Zero-shot hallucination | +0.0812 / 0.006 | +0.0420 / 0.152 | ❌ withdrawn |

**You were right to remove them.** Your separation of the lexicon repair from the construct
change is what made this adjudicable — without `hindi_prop_v2` holding the construct fixed,
the Das & Gambäck sign flip would have looked like a contradiction instead of a different
measure. That was the right call.

### 1.2 The Q3 caption confound — ✅ confirmed, and worse than you stated

The `condition_group` label appears in **96.2%** of `english_summary` rows (33.6% as the
underscore form, 74.3% spaced), and **95.0%** of that sits in the caption clause.

My "gold English R@1 = 23.4%" lands almost exactly on Q3 (0.2143), which confirms your
diagnosis precisely. **My number was wrong and yours is the correct framing.** Corrected:

| | My earlier claim | Corrected |
|---|---|---|
| Hinglish R@1 | 12.7% (n=299) | **11.4%** (n=3,015) |
| "English" R@1 | 23.4% *(leaked)* | **16.0%** (caption stripped) |
| Relative gap | 1.84× | **1.40×** |

Anything quoting 23.4% as an English baseline needs fixing. Your leakage gate is also
specified correctly — targeting only the templated underscore label rather than any
condition token, which is what avoids the 79.6% false-fail.

### 1.3 Caption-as-reference (M4′) — ✅ sound, built, and it is now the headline

Your stats verify (99.1% have a description, mean 74 chars, only 4.8% residual label leak).
I built it as `src/evaluation/caption_reference.py` and re-scored the cached 1,165:

| Metric | Δ | 95% cluster CI | Cohen's *d* |
|---|---:|---|---:|
| Evidence-based (circular) | +0.2749 | [+0.2471, +0.3035] | 0.678 |
| **M4′ caption (unbiased)** | **+0.0462** | [+0.0152, +0.0646] | **0.181** |

**The effect shrinks ~6× under unbiased measurement and survives.** This is the paper's
central result and it was your idea.

> **One caveat you didn't measure, and it matters.** The reference is low-cardinality: only
> **412 unique descriptions across 2,988 rows** (13.8% distinct), one covering **671 rows
> (22.3%)**, and skin_rash's 1,046 rows sharing 80 descriptions. So rows are *not*
> independent — significance tests need clustering. The CIs above resample descriptions,
> not rows; a row-level bootstrap would have overstated significance badly. `score_frame`
> now refuses to return a bare aggregate without its per-condition table. Two lines in
> Limitations, and it's fine.

---

## 2. H₀₄ — I ran it, and it did not need Kaggle

Your laptop encodes at 0.38 texts/s; this machine does **38.3**. The ~6-hour job took
**under 4 minutes on CPU**. `h4_retrieval.py` ran unmodified — your code was correct.

| Query condition | R@1 | R@10 | MRR@10 |
|---|---:|---:|---:|
| Q1 Hinglish (deployed) | 0.1144 | 0.6083 | 0.2432 |
| Q2 English question | 0.1602 | 0.6886 | 0.2985 |
| Q3 English + caption | 0.2143 | 0.7522 | 0.3592 |
| Random floor | 0.0626 | 0.4739 | — |

**H₀₄ rejected.** Q2 − Q1 = **+0.0458**, CI [+0.0292, +0.0627], McNemar p = 9.1×10⁻⁸.

Your §1.4 warning was well founded: the effect is **heterogeneous in sign** — English is
*worse* for `neck_swelling`, `foot_swelling`, `swollen_eye`, `skin_dryness`, `skin_growth` —
and skin_rash (35% of queries) dominates the aggregate. Never report it without the
per-condition table.

The Q2→Q3 increment (+0.054, larger than the code-mixing penalty itself) is your multimodal
headroom argument, and it holds.

---

## 3. Table 1 — the baselines, and they are uncomfortable

Recall@1, all 3,015 queries, same index, same relevance criterion:

| System | Q1 Hinglish | Q2 English | Q3 +caption |
|---|---:|---:|---:|
| BM25 (lexical) | **0.1343** | 0.1867 | 0.2862 |
| multilingual-e5-base | 0.1303 | **0.2454** | 0.2806 |
| TF-IDF (lexical) | 0.1167 | 0.1393 | 0.2063 |
| **LaBSE (deployed)** | **0.1144** | 0.1602 | 0.2143 |
| MuRIL | 0.0640 | 0.1821 | 0.2305 |
| *random floor* | *0.0626* | — | — |

1. **LaBSE ranks 4th of 5** on the deployed path, on R@1 and MRR@10 both.
2. **BM25 beats it.** Hinglish queries carry English clinical terms that match the English
   corpus directly; the cross-lingual embedding seems to dilute that rather than add to it.
   Our "this is not a translation problem" premise needs restating.
3. ⭐ **MuRIL is AT the random floor on Hinglish (0.0640 vs 0.0626)** but recovers to 0.1821
   on the same content in English. It is trained on *Devanagari* Hindi; MMCQSD is
   *romanised*. **Script mismatch, not language mismatch, breaks it.** Cleanest new finding
   of the session, and a real contribution to code-mixed IR.

---

## 4. The five hardening items — all closed

| Item | Outcome |
|---|---|
| **Real-retrieval H1** | Done, n=268. **The oracle problem was a false alarm** — see §5 |
| **M4′** | Done, with cluster-aware statistics |
| **Unify 5 lexicons** | `src/evaluation/concept_lexicon.py`, 21 regression tests. Old copies kept verbatim but each now carries a banner naming its defects |
| **CMI figures** | Regenerated continuous on `hindi_prop_v2`, tertiles dropped, 300-DPI + vector PDF |
| **`threshold_ratio`** | Fixed 0.5 → 0.05. Sweep is a clean **negative result**: adaptive wins **0 of 6** vs fixed *k* |

Two notes you'll care about:

- **The lexicon needed a suffix rule.** `-algia` had to be declared explicitly — a naive
  `\balgia\b` never matches *neuralgia*/*myalgia*, so the "fix" would have silently broken
  the `pain` concept. Worth knowing if you extend the lexicon.
- **The truncation sweep says drop the claim, not fix it.** Precision is flat (~0.113–0.115)
  across *every* setting while recall falls monotonically — the similarity gap carries no
  relevance information. Recommend removing "MMed-RAG-style adaptive selection" from
  README/config/poster and reporting the negative result instead.

---

## 5. 🔴 The thing that changed the paper

**Real-retrieval H1 (n=268, three arms, one generator):**

| Contrast | zero-shot | grounded | Δ | *d* | p |
|---|---:|---:|---:|---:|---:|
| Oracle evidence (ceiling) | 0.3142 | 0.5045 | +0.1902 | 0.500 | 7.2×10⁻⁹ |
| Real retrieval (deployed) | 0.2720 | 0.4557 | +0.1837 | 0.492 | 1.1×10⁻⁸ |

**Oracle − real = +0.0755, p = 0.106 — not significant.** The oracle problem, which we both
called the top structural threat, costs almost nothing.

But five results now converge on a different explanation:

| Evidence | Result |
|---|---|
| Oracle vs real | p = 0.106 |
| Retrieval correctness → factuality | **p = 0.53** |
| Circular vs unbiased metric | **6× shrinkage** |
| BM25 vs LaBSE | lexical wins |
| Refusal rate oracle vs real | p = 0.79 |

**The measured grounding benefit is largely an *echo* effect.** Told "base your response
strictly on the evidence," the model either declines or restates concepts from whatever
text it was handed — and the evidence-based metric scores the answer against that same
text, so restating anything scores well whether or not it was relevant.

**Also a genuine positive result:** the grounded arm **refuses 84%** of the time (vs 0%
zero-shot) rather than confabulating. That's correct safety behaviour — and it's invisible
to the concept metric, because refusals score `nan`. A naive mean makes the system look
healthiest exactly where it fails, so refusal rate is now reported alongside every
factuality number.

---

## 6. 🔴 Two blockers you need to know about

**`llama-3.1-8b-instant` is decommissioned.** It 404s on every key; no Llama chat model
remains on the account. **Keys #1 and #2 in `.env` are also revoked (401)** — the runner now
validates keys at startup so a dead key can't burn every retry.

Consequence: our results span two generators. The n=1,165 H1/H2 work is llama (which cannot
be reproduced); the n=268 real-retrieval work is `openai/gpt-oss-20b`. **We need to decide
this together** — my recommendation is gpt-oss as primary with llama as a replication
appendix (zero extra cost, and the model-transfer contrast becomes a feature). Re-running
everything on one model is cleaner but costs 3–4 days of quota. Either way it goes in
Limitations.

**Throughput, if you run Groq jobs:** rotating keys only on error leaves the spares idle,
because the SDK absorbs 429s internally and the error never reaches your handler. That
capped us at 0.77 rows/min with 47% rate-limited. Round-robining *every* call across the
live keys took it to 5–10 rows/min. The fix is in `h1_real_retrieval.py`.

---

## 7. What's left, in priority order

| # | Item | Cost | Who |
|---|---|---|---|
| **1** | ⭐ **Random-evidence control** — ground on a random case, re-score | ~300 calls | either |
| **2** | **Decide the generator question** (§6) | discussion | **both** |
| **3** | **H₀₃ provenance** — matched-topic, equal-size ~1,800-doc indexes | ~1,800 calls | either |
| **4** | **README / config / requirements** — still advertise LLaVA, BioMedCLIP, QLoRA, DPO | ~2 h | either |
| **5** | **Figures** for H₀₁/H₀₄/baselines/refusal/truncation | free | either |
| **6** | Encoder swap decision (e5 vs LaBSE); re-run Phase-6 ablation under unified lexicon | free | either |
| **7** | Zenodo deposit + DOI — **required** by SN Computer Science Declarations | free | either |

**Item 1 is the priority.** §5 rests on five *indirect* results. A random-evidence arm tests
it head-on: if factuality barely drops, the echo thesis is proven and becomes the paper's
central contribution; if it drops sharply, §5 needs another explanation. Either outcome is
publishable, and a reviewer will ask for this control by name.

Item 3 matters because the proposal commits to three hypotheses — delivering two invites
"why not H₃?".

---

## 8. Repo state

Branch **`saachi-hardening`**, pushed, 6 commits. `main` was fast-forwarded to include all
your `devikas-updates` work first, then this branched off it — so `main` is current and
nothing of yours was lost.

Everything reproduces from a clean clone; your `results/**` re-include is what made that
possible, and it was the right fix.

One operational annoyance: the `gh` active account reverts to `saachi-smartdecision`, which
has no write access and 403s on push. `gh auth switch --user saaachis` before pushing works.

---

**Short version:** your two corrections were both right and I've retracted mine. H₀₄ is done
on the full corpus. All five hardening items are closed. The oracle problem turned out to be
a false alarm — but chasing it surfaced something better, and the paper is now a stronger
one than we started with. The next move is the random-evidence control.
