# Prototype Plan — H1 MVP Demo

> Build a small, working prototype that demonstrates **Hypothesis H1**: grounded RAG produces more factually consistent and less hallucinated Hinglish clinical explanations than zero-shot generation.

---

## 1) MVP Objective

Create a demo where the same Hinglish query produces:

1. **Zero-shot output** (no retrieved evidence)
2. **Grounded output** (with retrieved Open-i evidence)

Then compare both outputs on factuality and hallucination indicators across the same query set.

---

## 2) Scope (Mentor-Aligned and Feasible)

### In Scope (Must for MVP)
- **Open-i** (authoritative evidence source)
- **MMCQSD** (Hinglish pattern guidance)
- **HMG-mini** (small custom dataset, 100-300 samples)
- Text-first RAG pipeline (multimodal display optional)

### Out of Scope (for this MVP)
- Full-scale HMG (~4,000) generation
- Heavy fine-tuning (QLoRA/DPO) in first pass
- PubMedQA/MMed-Bench integration
- Full production-grade metric pipeline

---

## 3) End-to-End Flow

1. Prepare a small Hinglish-to-report evaluation set (`HMG-mini`)
2. Build embeddings for Open-i report subset
3. Build FAISS retrieval index
4. For each query:
   - Generate **zero-shot** response
   - Retrieve evidence and generate **grounded** response
5. Score both responses using factuality/hallucination rubric
6. Run paired H1 analysis and summarize results
7. Export results and summary for mentor review

---

## 4) Required Components

## Data
- Open-i report subset (e.g., 500-1500 reports)
- Hinglish query set with varied code-mixing (low/medium/high)
- Link each query to expected/paired report ID (for evaluation sanity)

## Models and Retrieval
- LaBSE encoder for text embeddings
- FAISS index for report retrieval
- Base generator model for zero-shot and grounded outputs

## Evaluation
- Per-sample factual support score
- Per-sample hallucination flag/rate
- Paired statistical comparison between zero-shot and grounded

---

## 5) Installation and Setup Checklist

1. Create and activate virtual environment
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Configure environment variables (if API/model key needed)
4. Verify runtime:
   - GPU if available, else CPU with reduced sample sizes
5. Prepare local folders:
   - `data/raw/`, `data/processed/`, `indices/`, `results/`

---

## 6) Implementation Plan (Phased)

## Phase A — Build `HMG-mini` (Day 1-2)
- Select Open-i subset
- Generate/curate 100-300 Hinglish queries (MMCQSD-guided patterns)
- Add code-mixing variety and quality checks
- Save to `data/processed/hmg_mini.csv`

**Output:** `hmg_mini.csv` with query, report_id, report_text, optional cmi bucket

## Phase B — Retrieval Setup (Day 2-3)
- Encode Open-i reports using LaBSE
- Build FAISS index
- Run sanity checks on 20 sample queries

**Output:** index files + retrieval sanity notes

## Phase C — Two Baseline Generators (Day 3-4)
- Implement zero-shot generation path
- Implement grounded generation path (prompt + retrieved evidence)
- Keep prompt style consistent except evidence injection

**Output:** paired outputs per query (zero-shot vs grounded)

## Phase D — H1 Evaluation (Day 4-5)
- Define simple transparent scoring rubric
- Compute per-query:
  - factual support score
  - hallucination flag/rate
- Run paired comparison (paired t-test or Wilcoxon fallback)

**Output:** `results/h1_mvp_results.csv` + summary stats

## Phase E — Review Readiness (Day 6-7)
- Add 3-5 representative example cases
- Prepare concise result tables and interpretation notes
- Prepare mentor presentation talking points

---

## 7) Suggested Repo Additions for MVP

- `src/prototype/build_hmg_mini.py`
- `src/prototype/build_index.py`
- `src/prototype/run_baselines.py`
- `src/prototype/evaluate_h1.py`
- `data/processed/hmg_mini.csv`
- `results/h1_mvp_results.csv`
- `results/h1_mvp_summary.md`

---

## 8) Evaluation Rubric (Simple and Defensible)

For each query output:

- **Factual Support Score (0-1):**
  - 1.0 = all core claims supported by retrieved evidence
  - 0.5 = partially supported / mixed
  - 0.0 = unsupported core claims

- **Hallucination Flag (0/1):**
  - 1 = contains unsupported or contradicted medical claim
  - 0 = no clear unsupported claim

Aggregate:
- Mean factual score (zero-shot vs grounded)
- Hallucination rate (zero-shot vs grounded)
- Paired significance test for H1 trend

---

## 9) Success Criteria (H1 MVP)

The MVP is successful if:

1. Retrieval returns relevant evidence for most sample queries
2. Grounded output shows higher mean factual support than zero-shot
3. Grounded output shows lower hallucination rate than zero-shot
4. Results are reproducible from scripts and clearly summarized in result files

---

## 10) Team Execution Split (Current Alignment)

- **Saachi:** architecture/integration, FAISS/retrieval wiring, end-to-end stability
- **Devika:** `HMG-mini` creation, data quality, retrieval relevance checks
- **Manjiri:** evaluation rubric + H1 statistics + results reporting

All three collaborate on interpretation and final paper writing.

---

## 11) Risks and Mitigations

- **Retrieval mismatch:** reduce corpus size and clean report text before indexing
- **Slow inference:** use smaller sample size and batch offline generations
- **Noisy synthetic Hinglish:** manually review high-impact subset
- **Ambiguous scoring:** keep rubric simple and explicit; document criteria

---

## 12) One-Week Fast Track Timeline

- **Day 1:** HMG-mini dataset draft
- **Day 2:** Retrieval index ready
- **Day 3:** Zero-shot and grounded generation scripts
- **Day 4:** Run paired outputs on mini set
- **Day 5:** H1 scoring and statistical comparison
- **Day 6:** Result summary preparation
- **Day 7:** Polish + mentor-ready walkthrough

