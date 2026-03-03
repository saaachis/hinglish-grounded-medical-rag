# Datasets — Selection & Rationale

> What datasets we found, why we chose them, and how each fits into our pipeline.

---

## Primary Dataset (We Build This)

### HMG — Hinglish-Medical-Grounding Dataset

| Property | Detail |
|---|---|
| **Size** | ~4,000 triplets |
| **Format** | `{Chest X-ray Image, English Radiology Report, Synthetic Hinglish Query}` |
| **Source** | Constructed from Open-i reports using Llama-3-8B-Instruct |
| **Linguistic Template** | MMCQSD (ensures realistic code-switching) |

**Why we need to build this:**  
No large-scale Hinglish radiology dataset exists. This is the core innovation — a "synthetic-to-real" pipeline that transforms formal English findings into informal Hinglish doctor-patient queries with controlled linguistic noise.

**Example triplet:**

| Component | Content |
|---|---|
| X-ray | Chest radiograph (Open-i) |
| English Report | *"Right-sided pleural effusion. Heart size is normal."* |
| Hinglish Query | *"Doctor, right side chhati me pani bhar gaya hai kya?"* |

---

## Secondary Datasets (Pre-existing, Public)

### Scope Decision (After Mentor Review)

To keep this project feasible in compute, storage, and timeline, we are narrowing the implementation scope for the current phase.

**Current implementation scope (must-use):**
- Open-i
- MMCQSD
- HMG (constructed by us from Open-i + MMCQSD patterns)

**Deferred to future scope (not used in current build):**
- PubMedQA
- MMed-Bench

This reduced scope still fully supports our core objective: evidence-grounded Hinglish clinical explanation generation with optional multimodal support.

### A. Open-i — Indiana University Chest X-Ray Collection

| Property | Detail |
|---|---|
| **Size** | 7,470 chest X-ray images + 3,955 structured radiology reports |
| **License** | CC-0 (Public Domain) — no ethics approval needed |
| **Source** | [openi.nlm.nih.gov](https://openi.nlm.nih.gov/) |

**Why we liked it:**
- Completely open and legal to use (public domain)
- Structured radiology reports serve as our "source of truth" for RAG retrieval
- Reports are in formal English — the exact counterpart to our informal Hinglish queries
- Image-report pairs allow multimodal alignment

**Role in pipeline:** Knowledge base for evidence retrieval. Every generated explanation must be grounded in these reports.

---

### B. MMCQSD — Multimodal Medical Code-Mixed Question Summarization Dataset

| Property | Detail |
|---|---|
| **Size** | 3,015 samples |
| **Format** | Medical images (Skin, ENT, Eye, etc.) + Hinglish patient queries + English summaries |
| **Source** | MedSumm-ECIR2024 |

**Why we liked it:**
- The only open dataset capturing how Indian patients actually describe symptoms in Hinglish
- Covers the "messy" real-world linguistic patterns (mixed Hindi syntax + English medical terms)
- Provides ground-truth code-switching patterns we can use as templates
- Paired English summaries enable quality validation

**Role in pipeline:** Linguistic template for our HMG dataset construction. MMCQSD examples guide Llama-3 in producing realistic Hinglish queries.

---

### C. PubMedQA

| Property | Detail |
|---|---|
| **Size** | Large-scale (1,000 expert-labeled + 61.2K unlabeled + 211.3K artificial) |
| **Format** | Question + PubMed abstract context + Yes/No/Maybe answer |
| **Source** | [pubmedqa.github.io](https://pubmedqa.github.io/) |

**Why we liked it:**
- Industry standard for testing whether an LLM understands clinical evidence
- Answers are derived from actual PubMed abstracts (real medical reasoning)
- Useful for evaluating if our model can reason over medical text before we add Hinglish complexity

**Role in pipeline (deferred):** Medical logic pre-training and evaluation.

**Decision:** Deferred for now due to high data/compute overhead and lower direct relevance to our first deliverable (radiology-grounded Hinglish RAG).

---

### D. MMed-Bench

| Property | Detail |
|---|---|
| **Size** | 25,500+ medical QA pairs |
| **Format** | Multimodal QA across Pathology, Radiology, etc. |
| **Source** | MMed-RAG paper |

**Why we liked it:**
- Massive free multimodal benchmark across medical specialties
- Includes "distractor" answers — useful for preference alignment
- Can train our model (via DPO) to strictly rely on retrieved evidence and not hallucinate

**Role in pipeline (deferred):** Preference alignment and multimodal grounding evaluation.

**Decision:** Deferred for now due to scale and added complexity. For current work, DPO/preference pairs will be derived from our HMG setup instead.

---

## Dataset Summary (Updated Scope)

| Dataset | Status | Size | Open? | Our Role |
|---|---|---|---|---|
| **HMG** (ours) | **Must-use** | ~4,000 triplets | We create it | Primary training & evaluation data |
| **Open-i** | **Must-use** | 3,955 reports + 7,470 images | CC-0 | Evidence knowledge base |
| **MMCQSD** | **Must-use** | 3,015 samples | Yes | Hinglish linguistic templates |
| **PubMedQA** | **Deferred** | 273K+ | Yes | Optional medical reasoning extension |
| **MMed-Bench** | **Deferred** | 25,500+ | Yes | Optional multimodal benchmarking / DPO extension |

---

## Minimum Viable Dataset Stack (Current Phase)

1. **Open-i** for authoritative report retrieval and optional X-ray grounding  
2. **MMCQSD** for realistic Hinglish/code-mixing patterns  
3. **HMG** (custom) as the main experiment dataset

This minimum stack is sufficient to implement and evaluate:
- Hinglish query encoding
- Evidence retrieval (FAISS)
- Grounded generation
- Factuality and hallucination-focused evaluation

---

## Data Access Notes

- **No ethics approval or CITI certificates** required for any of these datasets
- All datasets are publicly available for academic research
- Large files (images, model weights) are excluded from git via `.gitignore`
- Raw data stored locally in `data/raw/`, processed in `data/processed/`
