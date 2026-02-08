# Grounded Multimodal RAG for Code-Switching (Hinglish) in Clinical Decision Support Systems

> Evidence-first multimodal RAG framework for grounded clinical decision support under Hinglish (Hindi–English) code-switched queries, aligned with authoritative medical reports and optional radiological evidence.

**Research Discourse - I | Group 5**

---

## Problem Statement

Modern clinical AI systems assume medical reasoning begins with clean, formal language. In reality — especially in India — patient-doctor interactions occur in **Hinglish** (Hindi–English code-switching), which:

- Lacks standardized grammar
- Varies by region, education, and context
- Encodes medical meaning indirectly

| **Formal Report** | **Hinglish Patient Query** |
|---|---|
| *"Right-sided pleural effusion"* | *"Doctor, right side chhati me pani bhar gaya hai kya?"* |

**This is not a translation problem — it is a reasoning alignment problem.**

When processed by existing AI models, Hinglish expressions are frequently treated as noisy or broken English, leading to incorrect retrieval, loss of semantic intent, or fluent yet **clinically unsafe hallucinated explanations**.

---

## Proposed Approach

This research designs and evaluates an **evidence-first Retrieval-Augmented Generation (RAG) framework** that:

1. **Encodes** Hinglish clinical queries using multilingual representations (LaBSE)
2. **Retrieves** relevant evidence from curated clinical corpora (FAISS + adaptive context selection)
3. **Generates** grounded Hinglish explanations conditioned on retrieved evidence
4. **Optionally enriches** with multimodal signals (chest X-rays via BioMedCLIP)

The system is explicitly positioned as a **clinical decision support tool** — it assists interpretation while minimizing hallucinations, without attempting autonomous diagnosis.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    HINGLISH PATIENT QUERY                        │
│  "Doctor, right side chhati me pani bhar gaya hai kya?"         │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │   TEXT ENCODER (LaBSE)  │  ← Multilingual embedding
         │   Hinglish → Vector     │
         └────────────┬────────────┘
                      │
          ┌───────────▼───────────┐
          │   FAISS RETRIEVER     │  ← Adaptive context selection
          │   + Score Truncation  │    (MMed-RAG style)
          └───────────┬───────────┘
                      │
     ┌────────────────▼────────────────┐
     │   RETRIEVED CLINICAL EVIDENCE   │
     │   • English radiology reports   │
     │   • (Optional) X-ray images     │
     └────────────────┬────────────────┘
                      │
         ┌────────────▼────────────┐
         │  GROUNDED GENERATOR     │  ← LLaVA-v1.5 + QLoRA
         │  Evidence → Hinglish    │    + DPO (anti-hallucination)
         │  Explanation            │
         └────────────┬────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│              GROUNDED HINGLISH EXPLANATION                       │
│  (factually consistent with retrieved medical evidence)          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
hinglish-grounded-medical-rag/
│
├── config/
│   └── config.yaml              # Project configuration
│
├── src/
│   ├── data/
│   │   ├── download.py          # Dataset download utilities
│   │   ├── preprocess.py        # Data cleaning & CMI computation
│   │   └── hmg_builder.py       # HMG dataset construction (synthetic-to-real)
│   │
│   ├── encoding/
│   │   ├── text_encoder.py      # LaBSE — multilingual text encoding
│   │   └── image_encoder.py     # BioMedCLIP — medical image encoding
│   │
│   ├── retrieval/
│   │   ├── indexer.py           # FAISS index builder
│   │   └── retriever.py         # Adaptive context selection retriever
│   │
│   ├── generation/
│   │   ├── generator.py         # Evidence-grounded Hinglish generation
│   │   └── trainer.py           # QLoRA fine-tuning & DPO training
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # MMFCM, factual consistency, hallucination rate
│   │   └── hypothesis.py        # Statistical testing (H1, H2, H3)
│   │
│   └── utils/
│       └── helpers.py           # Logging, seeding, device detection
│
├── notebooks/
│   ├── 01_eda_openi.ipynb       # EDA — Open-i Chest X-Ray Collection
│   ├── 02_eda_mmcqsd.ipynb      # EDA — MMCQSD Hinglish medical queries
│   ├── 03_eda_pubmedqa.ipynb    # EDA — PubMedQA biomedical QA
│   └── 04_hmg_dataset_construction.ipynb  # HMG dataset pipeline
│
├── app/
│   └── streamlit_app.py         # Demo interface
│
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_hypothesis.py       # Hypothesis testing tests
│   └── test_preprocess.py       # Preprocessing tests
│
├── research-work/               # Research proposal & documents
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## Datasets

### Primary — HMG (Hinglish-Medical-Grounding)
Custom-built dataset of **~4,000 triplets**: `{Open-i X-ray, English Report, Synthetic Hinglish Query}`.  
Generated using **Llama-3-8B-Instruct** with **MMCQSD** as a linguistic template for realistic code-switching.

### Secondary

| Dataset | Size | Role |
|---|---|---|
| **Open-i** (Indiana Univ.) | 7,470 images + 3,955 reports | Clinical evidence source (CC-0) |
| **MMCQSD** | 3,015 samples | Hinglish linguistic templates |
| **PubMedQA** | Large-scale | Medical reasoning pre-training |
| **MMed-Bench** | 25,500+ QA pairs | Multimodal grounding & evaluation |

---

## Key Techniques

| Technique | Purpose |
|---|---|
| **LaBSE** | Cross-lingual encoding (Hinglish ↔ English) |
| **BioMedCLIP** | Medical image encoding for cross-modal alignment |
| **FAISS + Adaptive Truncation** | Evidence retrieval with quality filtering |
| **QLoRA (4-bit)** | Parameter-efficient fine-tuning on consumer GPU |
| **DPO** | Direct Preference Optimization for hallucination control |
| **MMFCM** | Multimodal Fact Capturing Metric for clinical evaluation |

---

## Hypotheses

| ID | Hypothesis | Test |
|---|---|---|
| **H1** | Evidence-grounded RAG significantly outperforms zero-shot LLMs in factual consistency | Paired t-test / Wilcoxon |
| **H2** | Increasing code-mixing level does not significantly degrade grounded model performance | Two-way ANOVA / Kruskal-Wallis |
| **H3** | Authoritative medical evidence significantly improves factual correctness over general biomedical text | Paired t-test / Wilcoxon |

---

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended for model training/inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/saaachis/hinglish-grounded-medical-rag.git
cd hinglish-grounded-medical-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
streamlit run app/streamlit_app.py
```

---

## Research Team

| Name | Roll No. |
|---|---|
| Devika Jonjale | B045 |
| Saachi Shinde | B048 |
| Manjiri Apshinge | B061 |

**Course:** Research Discourse - I

---

## License

This project is for academic research purposes.

---

## Acknowledgements

- [Open-i / Indiana University](https://openi.nlm.nih.gov/) — Chest X-Ray Collection (CC-0)
- [MMCQSD / MedSumm](https://arxiv.org/abs/2401.01596) — Code-mixed medical query dataset
- [PubMedQA](https://pubmedqa.github.io/) — Biomedical question answering
- [MMed-RAG](https://arxiv.org/abs/2410.13085) — Multimodal medical RAG framework
