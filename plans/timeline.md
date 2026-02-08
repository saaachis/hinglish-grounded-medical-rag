# Project Timeline

> Phased plan from setup to research paper, with milestones and checkpoints.

---

## Overview

| Phase | Focus | Duration |
|---|---|---|
| Phase 1 | Foundation & Setup | Week 1–2 |
| Phase 2 | Data — EDA & HMG Construction | Week 3–5 |
| Phase 3 | Encoding & Retrieval Pipeline | Week 6–7 |
| Phase 4 | Generation & Fine-Tuning | Week 8–10 |
| Phase 5 | Evaluation & Hypothesis Testing | Week 11–12 |
| Phase 6 | Integration, Demo & Paper | Week 13–15 |

---

## Detailed Phase Breakdown

### Phase 1 — Foundation & Setup (Week 1–2)

- [x] Create GitHub repository and branch structure
- [x] Set up project directory structure
- [x] Define configuration (config.yaml)
- [x] Write requirements.txt with all dependencies
- [x] Create module stubs for full pipeline
- [x] Prepare planning documents
- [ ] Set up virtual environment on all team members' machines
- [ ] Familiarize all members with GitHub workflow (branching, PRs, commits)
- [ ] Download and store secondary datasets locally

**Milestone:** Repo is structured, all members can clone/pull/push, raw datasets are available.

---

### Phase 2 — Data: EDA & HMG Construction (Week 3–5)

#### Week 3 — EDA on Secondary Datasets
- [ ] EDA on Open-i: report structure, length distribution, medical entity analysis
- [ ] EDA on MMCQSD: Hinglish patterns, CMI distribution, code-switching analysis
- [ ] EDA on PubMedQA: question types, evidence passage characteristics
- [ ] Document findings in notebooks

#### Week 4 — HMG Pipeline Development
- [ ] Build prompt templates using MMCQSD linguistic patterns
- [ ] Set up Llama-3-8B-Instruct for Hinglish query synthesis
- [ ] Implement synthetic-to-real generation pipeline
- [ ] Generate first batch (~500 queries) and validate quality

#### Week 5 — HMG Completion & Validation
- [ ] Scale generation to full ~4,000 triplets
- [ ] Compute CMI scores for all generated queries
- [ ] Bucket into low / medium / high code-mixing levels
- [ ] Quality check and clean the dataset
- [ ] Finalize HMG dataset

**Milestone:** HMG dataset of ~4,000 validated triplets with CMI labels is ready.

---

### Phase 3 — Encoding & Retrieval Pipeline (Week 6–7)

#### Week 6 — Encoding
- [ ] Implement LaBSE text encoder (Hinglish queries + English reports)
- [ ] Implement BioMedCLIP image encoder (chest X-rays)
- [ ] Encode all Open-i reports into vector embeddings
- [ ] Validate embedding quality (nearest-neighbor sanity checks)

#### Week 7 — Retrieval
- [ ] Build FAISS index over report embeddings
- [ ] Implement basic top-k retrieval
- [ ] Implement adaptive context selection (similarity-score truncation)
- [ ] Test retrieval: Hinglish query → relevant English reports
- [ ] Write retrieval evaluation (precision@k, recall@k)

**Milestone:** Given a Hinglish query, the system retrieves relevant English reports from FAISS.

---

### Phase 4 — Generation & Fine-Tuning (Week 8–10)

#### Week 8 — Baseline Generation
- [ ] Set up LLaVA-v1.5-7B with QLoRA (4-bit quantization)
- [ ] Implement zero-shot generation baseline (no evidence injection)
- [ ] Implement evidence-grounded generation (with retrieved reports in prompt)
- [ ] Compare outputs qualitatively

#### Week 9 — QLoRA Fine-Tuning
- [ ] Prepare HMG dataset in training format
- [ ] Fine-tune LLaVA with QLoRA on HMG triplets
- [ ] Monitor training loss, validate on held-out set
- [ ] Save fine-tuned adapter weights

#### Week 10 — DPO Training
- [ ] Construct preference pairs (grounded = preferred, hallucinated = dis-preferred)
- [ ] Run DPO training for hallucination control
- [ ] Evaluate DPO impact on hallucination rate
- [ ] Finalize the generator model

**Milestone:** Fine-tuned model generates grounded Hinglish explanations from retrieved evidence.

---

### Phase 5 — Evaluation & Hypothesis Testing (Week 11–12)

#### Week 11 — Metrics & Scoring
- [ ] Implement MMFCM (Multimodal Fact Capturing Metric)
- [ ] Implement factual consistency scorer
- [ ] Implement hallucination rate scorer
- [ ] Score all outputs: zero-shot vs RAG, across CMI levels, across evidence types

#### Week 12 — Statistical Testing
- [ ] Run H1 test: paired t-test / Wilcoxon (RAG vs zero-shot)
- [ ] Run H2 test: two-way ANOVA / Kruskal-Wallis (CMI level interaction)
- [ ] Run H3 test: paired t-test / Wilcoxon (authoritative vs general evidence)
- [ ] Compute effect sizes, confidence intervals
- [ ] Generate plots: robustness curves, comparison charts

**Milestone:** All three hypotheses tested with statistical rigor. Results are clear.

---

### Phase 6 — Integration, Demo & Paper (Week 13–15)

#### Week 13 — Streamlit Demo
- [ ] Connect full pipeline to Streamlit interface
- [ ] Query input → retrieval display → explanation output
- [ ] Add confidence indicators and evidence highlighting

#### Week 14 — Research Paper Drafting
- [ ] Write methodology section (grounded in code and results)
- [ ] Write results section with statistical tables and plots
- [ ] Write discussion, limitations, and future scope
- [ ] Internal review within team

#### Week 15 — Final Polish
- [ ] Finalize paper
- [ ] Clean up codebase and notebooks
- [ ] Final README update
- [ ] Submission

**Milestone:** Research paper submitted. Demo is functional. Codebase is clean.

---

## Progress Checkpoints

| Checkpoint | When | What to Review |
|---|---|---|
| **CP1** | End of Week 2 | Repo setup, datasets downloaded, team onboarded |
| **CP2** | End of Week 5 | HMG dataset ready, EDA complete |
| **CP3** | End of Week 7 | Retrieval pipeline working end-to-end |
| **CP4** | End of Week 10 | Generator fine-tuned, producing grounded outputs |
| **CP5** | End of Week 12 | All hypotheses tested, results documented |
| **CP6** | End of Week 15 | Paper submitted, demo ready, codebase clean |
