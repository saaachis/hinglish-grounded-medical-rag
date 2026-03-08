# Work Distribution

> How the work is divided across the team. Each member contributes to every phase — primary ownership indicates who leads, but everyone collaborates and learns.

---

## Guiding Principles

1. **Everyone learns everything** — primary ownership means you lead that area, not that you work alone
2. **Cross-training is intentional** — members will pair up on tasks outside their primary zone
3. **All members contribute to the research paper** — each writes about the areas they led
4. **Regular sync-ups** — weekly check-ins to share progress and unblock each other

---

## Member Roles

### Saachi Shinde — Architecture & Integration Lead

**Primary ownership:** System architecture design, core pipeline integration, and mentoring the team on development workflows.

| Phase | Responsibilities |
|---|---|
| Phase 1 | Repository setup, environment configuration, onboarding team to GitHub workflows |
| Phase 2 | Architecture design for HMG pipeline, code review and guidance for Devika's generation scripts |
| Phase 3 | FAISS indexing and adaptive retrieval implementation |
| Phase 4 | QLoRA fine-tuning setup, model integration across pipeline stages |
| Phase 5 | End-to-end pipeline integration, debugging cross-module issues |
| Phase 6 | Final integration, codebase cleanup, deployment support |

**Paper sections:** Methodology (system architecture, pipeline design), Implementation Details

---

### Devika Jonjale — Data, Dataset & Retrieval Lead

**Primary ownership:** Dataset research, curation, EDA, HMG dataset construction (including code), and retrieval pipeline development.

| Phase | Responsibilities |
|---|---|
| Phase 1 | Dataset sourcing and access verification, dataset documentation, literature references setup |
| Phase 2 | EDA on all datasets (Open-i, MMCQSD, PubMedQA), HMG synthetic generation pipeline development (Llama-3 prompting, generation scripts), CMI analysis |
| Phase 3 | LaBSE text encoder integration, retrieval quality evaluation, preparing report corpus for indexing |
| Phase 4 | Preference pair construction for DPO (grounded vs hallucinated examples), training data formatting and curation |
| Phase 5 | Metrics implementation (MMFCM, factual consistency), result interpretation against literature |
| Phase 6 | Data description and results section of paper, Streamlit demo (data/content accuracy and clinical display) |

**Paper sections:** Literature Review, Data Description & HMG Construction, Results & Discussion

---

### Manjiri Apshinge — Evaluation, Multimodal & Conceptual Grounding Lead

**Primary ownership:** Evaluation framework development (including code), image encoding pipeline, statistical hypothesis testing, and conceptual alignment.

| Phase | Responsibilities |
|---|---|
| Phase 1 | Understanding pipeline architecture, environment setup on local machine, reviewing research design |
| Phase 2 | HMG quality validation (reviewing generated Hinglish for realism), conceptual review of dataset completeness |
| Phase 3 | BioMedCLIP image encoder integration, embedding quality validation and sanity checks |
| Phase 4 | Evidence injection prompt design, grounded generation strategy, baseline comparison setup, DPO training pipeline |
| Phase 5 | Statistical hypothesis testing (H1, H2, H3), visualization, plot generation, effect size analysis |
| Phase 6 | Streamlit demo (primary owner — UI, layout, interaction flow, end-to-end demo), evaluation section of paper, limitations & future scope |

**Paper sections:** Evaluation & Hypothesis Testing, Limitations & Future Scope

---

## Collaboration Map

Tasks that involve multiple members working together:

| Task | Members | How |
|---|---|---|
| **HMG dataset construction** | Devika (generation pipeline code + dataset design) + Saachi (architecture guidance, code review) | Devika builds the generation scripts, Saachi reviews and helps with architecture decisions |
| **HMG quality validation** | Devika (linguistic quality) + Manjiri (conceptual completeness) | Devika checks Hinglish realism, Manjiri reviews medical concept coverage |
| **Encoding** | Devika (LaBSE text encoder) + Manjiri (BioMedCLIP image encoder) | Each builds and owns one encoder end-to-end |
| **Retrieval** | Saachi (FAISS indexing + adaptive truncation) + Devika (retrieval quality evaluation) | Saachi builds the index, Devika evaluates retrieval results |
| **Generation & fine-tuning** | Saachi (QLoRA setup) + Manjiri (DPO training + prompt design) | Split the two fine-tuning stages across members |
| **Evaluation** | Devika (metrics implementation) + Manjiri (statistical testing + visualization) | Devika implements scoring functions, Manjiri runs hypothesis tests and plots |
| **Research paper** | All three | Each writes their owned sections, all three collaborate on editing and coherence |
| **Streamlit demo** | Manjiri (primary — UI, layout, interaction flow) + Saachi (pipeline connection support) | Manjiri owns the demo end-to-end, Saachi helps with hooking up the backend |
| **GitHub & code reviews** | All three | Saachi guides PRs and branching, everyone reviews each other's code |

---

## Skill-Sharing Plan

Part of this project is learning. Planned knowledge-sharing sessions:

| Session | Led By | For | When |
|---|---|---|---|
| GitHub workflow (branching, PRs, commits) | Saachi | Devika, Manjiri | Phase 1 |
| Dataset landscape & research methodology | Devika | Saachi, Manjiri | Phase 1–2 |
| Hinglish linguistics & CMI computation | Devika | Saachi, Manjiri | Phase 2 |
| ML pipeline (encoders, fine-tuning) | Saachi | Devika, Manjiri | Phase 3–4 |
| Statistical testing & interpretation | Manjiri | Saachi, Devika | Phase 5 |
| Academic writing & paper editing | All three (collaborative) | — | Phase 6 |

---

## Weekly Sync Format

- **When:** Once per week (day TBD by team)
- **Duration:** 30–45 minutes
- **Format:**
  1. Each member: what I did, what I'm stuck on, what I need
  2. Review progress against timeline checkpoints
  3. Decide next week's priorities
  4. Update this document if responsibilities shift
