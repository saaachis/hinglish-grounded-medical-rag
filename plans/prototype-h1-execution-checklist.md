# Prototype H1 Execution Checklist (7-Day)

> Goal: ship a **working, script-based MVP** that compares zero-shot vs grounded outputs and produces initial H1 results by 9 March.

---

## Success Definition (Must Have)

- [ ] End-to-end run: query -> retrieval -> zero-shot output -> grounded output
- [ ] Results file for at least 50-100 queries
- [ ] Factuality + hallucination comparison table
- [ ] Paired statistical test output for H1 (t-test or Wilcoxon fallback)
- [ ] One concise result summary for mentor review

---

## Scope Guardrails (Do Not Expand)

- [ ] Use only `Open-i + MMCQSD + HMG-mini`
- [ ] Keep `HMG-mini` size to 100-200 samples
- [ ] No QLoRA/DPO in this 7-day prototype pass
- [ ] No Streamlit/UI requirement for MVP
- [ ] Prefer text-first pipeline (image optional only if time remains)

---

## Environment and Setup

- [ ] Create virtual environment
- [ ] Install dependencies from `requirements.txt`
- [ ] Verify Python version and package imports
- [ ] Confirm local folders exist: `data/raw`, `data/processed`, `indices`, `results`
- [ ] Add `.env` only if model/API credentials are required
- [ ] Smoke-test one model inference call

**AI/Cursor help:** environment errors, dependency conflicts, import fixes.

---

## Data Tasks (HMG-mini)

### A. Open-i subset
- [ ] Select manageable subset of reports (500-1500)
- [ ] Normalize/clean report text
- [ ] Save clean report table with stable IDs

### B. Hinglish query set
- [ ] Build 100-200 Hinglish queries from selected reports
- [ ] Use MMCQSD style patterns for realistic phrasing
- [ ] Ensure low/medium/high code-mix examples are present
- [ ] Manual quality pass on at least 30-50 samples

### C. Final HMG-mini file
- [ ] Create `data/processed/hmg_mini.csv`
- [ ] Required columns: `sample_id`, `query_hinglish`, `report_id`, `report_text`, `cmi_bucket` (optional)
- [ ] Validate no missing critical fields

**AI/Cursor help:** scripts for cleaning, formatting, validation checks.

---

## Retrieval Pipeline Tasks

- [ ] Implement report embedding script (LaBSE)
- [ ] Generate embeddings for selected Open-i subset
- [ ] Build FAISS index and persist files in `indices/`
- [ ] Implement retrieval function (`top_k` configurable)
- [ ] Run retrieval sanity check on 20 queries
- [ ] Tune `top_k` and prompt context length

**Acceptance check:** at least ~70% sanity queries retrieve clinically relevant reports in top-k.

**AI/Cursor help:** embedding/index code, retrieval debugging, scoring utilities.

---

## Generation Baselines Tasks

### Zero-shot path
- [ ] Implement `generate_zero_shot(query)` script/function
- [ ] Freeze prompt template for consistency

### Grounded path
- [ ] Implement `generate_grounded(query, retrieved_evidence)`
- [ ] Keep same style as zero-shot except evidence injection
- [ ] Limit evidence chunk length to avoid prompt overflow

### Batch run
- [ ] Run both modes for all HMG-mini samples
- [ ] Save paired outputs to `results/h1_outputs.csv`

**Required columns:** `sample_id`, `query`, `zero_shot_output`, `grounded_output`, `retrieved_evidence_ids`.

**AI/Cursor help:** prompt engineering, batching, runtime/error handling.

---

## Evaluation Tasks (H1)

### Scoring rubric
- [ ] Finalize simple factual support rubric (0/0.5/1)
- [ ] Finalize hallucination flag rubric (0/1)
- [ ] Document rubric in file header/comments

### Scoring implementation
- [ ] Score zero-shot and grounded outputs per sample
- [ ] Save to `results/h1_scored.csv`

### Statistical comparison
- [ ] Compute mean factual score (zero-shot vs grounded)
- [ ] Compute hallucination rate (zero-shot vs grounded)
- [ ] Run paired t-test; fallback to Wilcoxon when non-normal
- [ ] Save summary to `results/h1_mvp_summary.md`

**AI/Cursor help:** scoring script, stats script, interpretation wording.

---

## Result Packaging Tasks

- [ ] Create one final compact table with key metrics
- [ ] Add 3-5 representative examples (good, bad, edge case)
- [ ] Write conclusion paragraph: does trend support H1?
- [ ] Note limitations (small sample, synthetic queries, etc.)

**Final deliverables (must exist):**
- [ ] `results/h1_outputs.csv`
- [ ] `results/h1_scored.csv`
- [ ] `results/h1_mvp_summary.md`

---

## Day-by-Day Checklist

## Day 1
- [ ] Environment ready
- [ ] Open-i subset prepared
- [ ] HMG-mini draft started

## Day 2
- [ ] HMG-mini completed (100-200)
- [ ] Data validation checks passed

## Day 3
- [ ] Embeddings built
- [ ] FAISS index working
- [ ] Retrieval sanity checks done

## Day 4
- [ ] Zero-shot and grounded generation scripts complete
- [ ] Batch outputs generated

## Day 5
- [ ] Scoring complete
- [ ] H1 statistical tests complete

## Day 6
- [ ] Summary files prepared
- [ ] Representative examples selected

## Day 7
- [ ] Final QA pass
- [ ] Mentor-ready results package complete

---

## Risk Controls (Use If Blocked)

- [ ] If inference is too slow: reduce to 50-80 samples first, then scale
- [ ] If retrieval quality is low: narrow report subset and retune `top_k`
- [ ] If generation noisy: tighten prompt and shorten evidence context
- [ ] If stats unstable: report effect direction + confidence interval with caveat

---

## Collaboration Checklist (Team + AI)

- [ ] Daily 15-minute sync: blockers + next actions
- [ ] Commit after each working milestone (small commits)
- [ ] Keep one branch for MVP stability
- [ ] Ask AI/Cursor immediately on blockers (errors, scripts, results formatting)
- [ ] End of each day: update completion status in this checklist

---

## AI/Cursor Commitment (What I will help with)

- [ ] Write/modify scripts for each phase
- [ ] Fix runtime and dependency issues
- [ ] Improve prompts and retrieval flow
- [ ] Implement scoring + statistical analysis code
- [ ] Generate concise summaries for mentor review
- [ ] Keep scope controlled so you hit deadline

