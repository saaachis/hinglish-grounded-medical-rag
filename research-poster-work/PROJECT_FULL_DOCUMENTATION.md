# Grounded RAG for Hinglish Clinical Queries — Full Project Documentation

**Purpose.** This document is the **primary long-form reference** for the repository `hinglish-grounded-medical-rag`. It is written so that a reader (or another language model) can understand **what the project is**, **why it exists**, **how it works end-to-end**, **what was found**, **what failed and was fixed**, and **what remains to do**—without relying on slide bullets alone.

**Companion.** `project-complete-summary.md` stays a **compact** poster-oriented digest with figure names and quick numbers. Use **this file** when you need **definitions**, **narrative**, and a **worked example** through the pipeline.

**Version note.** The evaluated system is **text-grounded RAG** (no imaging in the retrieval path). Early proposals mentioned **multimodal** models and **chest X-ray** corpora; the **implemented** milestone centers on **Hinglish queries**, **English case narratives**, and **API-based** text generation.

---

## Table of contents

1. [How to use this document with an LLM](#1-how-to-use-this-document-with-an-llm)  
2. [Abstract](#2-abstract)  
3. [Executive summary](#3-executive-summary)  
4. [Project identity](#4-project-identity)  
5. [Objectives and research questions](#5-objectives-and-research-questions)  
6. [Problem statement and motivation](#6-problem-statement-and-motivation)  
7. [Scope](#7-scope)  
8. [Hypotheses](#8-hypotheses)  
9. [Methodology overview](#9-methodology-overview)  
10. [End-to-end workflow (detailed)](#10-end-to-end-workflow-detailed)  
11. [Architecture (implemented)](#11-architecture-implemented)  
12. [Data sources and pairing pipeline](#12-data-sources-and-pairing-pipeline)  
13. [Critical limitation and resolution (Open-i → MultiCaRe)](#13-critical-limitation-and-resolution-open-i--multicare)  
14. [Evaluation methodology](#14-evaluation-methodology)  
15. [Results summary](#15-results-summary)  
16. [Phase 6 ablation (evidence format)](#16-phase-6-ablation-evidence-format)  
17. [Implementation and reproducibility](#17-implementation-and-reproducibility)  
18. [Streamlit demo](#18-streamlit-demo)  
19. [Limitations](#19-limitations)  
20. [Impact and significance](#20-impact-and-significance)  
21. [Future scope](#21-future-scope)  
22. [Ethics, safety, and positioning](#22-ethics-safety-and-positioning)  
23. [Repository map (key paths)](#23-repository-map-key-paths)  
24. [Glossary](#24-glossary)  
25. [Appendix: numeric quick reference](#appendix-numeric-quick-reference)  

---

## 1. How to use this document with an LLM

When you ask another model to draft a report, abstract, or slide deck **about this project**, attach or paste **this file** and instruct it to **treat it as the authoritative description** of scope, methods, and results.

**Do:**
- Preserve **distinctions** between **proposal** (multimodal, Open-i, local fine-tuning) and **what was built** (text RAG, MultiCaRe, Groq API).
- Use **numeric results** only from [§15](#15-results-summary) and the [Appendix](#appendix-numeric-quick-reference), unless you recompute from the repository.
- State clearly that the system is **decision support**, not a **diagnostic device**.

**Do not:**
- Invent **new datasets**, **p-values**, or **model names** not listed here.
- Claim that **zero-shot** quality **collapses** as **code-mixing** increases unless your own plots show it; the **documented** H2 pattern is **stability** of **grounded** performance across **CMI** tertiles with **non-significant** differences in **gains** across groups.

**Titles for external documents.** Prefer **“Grounded RAG pipeline for Hinglish clinical decision support”** when describing the **delivered** work; retain the original **“multimodal”** title only when quoting the **initial proposal** or course submission wording.

---

## 2. Abstract

**Background.** Clinical natural-language interfaces often assume **standard English**. In Indian outpatient and telehealth settings, patients frequently describe symptoms in **Hinglish**—**Hindi** and **English** mixed in the same utterance, usually in **Roman script**. Off-the-shelf large language models can answer **fluently** but may **hallucinate** findings or **ignore** the need to tie language to **verifiable clinical text**.

**Approach.** The project implements **retrieval-augmented generation (RAG)** specialized for this setting. **Hinglish** queries are embedded with **LaBSE**, a **cross-lingual** sentence encoder. **Dense retrieval** over a **FAISS** index finds **English** **clinical case narratives** from a **MultiCaRe-derived**, **specialty-filtered** corpus. Retrieved passages are optionally **adaptively truncated** (similarity **drop-off**) before being **injected** into prompts. **Llama-3.1-8B-Instant** is called through the **Groq API** to generate **Hinglish** explanations. The same model can run **without** evidence (**zero-shot**) for **paired comparison**.

**Evaluation.** Queries come from **MMCQSD** (3,015 real code-mixed items). **Automated** scores measure overlap between **medical concepts** in the **answer** and in **retrieved evidence** (**factual support**) versus concepts **not** supported (**hallucination** proxy). **Code-mixing intensity (CMI)** is computed from a **Hindi-origin** lexicon and used to form **tertiles** for **H2**. On **1,165** cleaned **paired** evaluations, **H1** is **strongly** supported: **grounded** outputs show **much higher** factual support and **lower** hallucination scores than **zero-shot**. **H2** tests whether **gains** **differ** across **CMI** levels; **Kruskal–Wallis** tests on **factual gain** and **hallucination reduction** are **not significant**, consistent with **robust** grounding across **low**, **medium**, and **high** code-mixing in this sample.

**Deliverables.** A **Streamlit** application demonstrates **live** retrieval and **side-by-side** **grounded** vs **zero-shot** answers. **Limitations** include **no** clinician validation, **text-only** evidence, **API** dependence, and **deferred** **H3** (evidence **provenance**). **Future work** includes **multimodal** retrieval, **parameter-efficient fine-tuning**, **preference learning**, and **human-in-the-loop** studies.

**Keywords (for indexing).** Hinglish, code-switching, clinical RAG, LaBSE, FAISS, MultiCaRe, MMCQSD, Groq, Llama-3.1, decision support, hallucination, code-mixing index.

---

## 3. Executive summary

The project asks a practical question: **If a patient speaks in everyday Hinglish, can we still anchor a language model’s answer to real clinical text, and does that measurably reduce “made-up” medical content?** The answer implemented in software is **yes on automated metrics**, within the **limits** described in [§19](#19-limitations).

**What was built.** An end-to-end **pipeline** (`src/pipeline.py`) and a **browser UI** (`app.py`). The pipeline **encodes** the user’s **Hinglish** string, **searches** a **vector index** built from **MultiCaRe** case reports, **formats** the top passages into a **prompt**, and calls **Groq**’s **Llama-3.1-8B-Instant** endpoint. It can also produce a **zero-shot** answer for the **same** query. **Concept-based** scores compare both answers to the **same** concatenated evidence string for **fair** paired analysis.

**What the numbers say (high level).** Across **n = 1,165** paired items, mean **factual support** rises from **0.32** (zero-shot) to **0.55** (grounded); mean **hallucination** score falls from **0.50** to **0.28**. **Effect sizes** are **medium** to **small–medium**; **p-values** are **extremely** small for **H1**. For **H2**, **grounded** factual scores stay **roughly flat** across **CMI** tertiles (~0.54–0.56), and **tests** do **not** show **strong** evidence that **grounding gains** **differ** by **tertile**.

| Aspect | Summary |
|--------|---------|
| **Goal** | **Decision support**: Hinglish in, **evidence-conditioned** Hinglish out—not autonomous diagnosis. |
| **Core mechanism** | **LaBSE** embedding → **FAISS** inner-product search → **adaptive** context cut → **prompt** + **Groq** LLM. |
| **Data** | **MMCQSD** queries; **MultiCaRe** narratives (**61,316** filtered cases in documentation; **10,000** vectors in typical demo index). |
| **Main statistical result** | **H1** supported at scale; **H2** consistent with **stability** across code-mixing strata. |
| **Artifacts** | Code under `src/`, plots under `research-poster-work/`, evaluation CSVs often under `results/` (may be gitignored). |

---

## 4. Project identity

**Repository name:** `hinglish-grounded-medical-rag`

**Institutional context.** The work was carried out as part of **Research Discourse – I** (as noted in team materials).

**Team.** Devika Jonjale (B045), Saachi Shinde (B048), Manjiri Apshinge (B061).

**Title evolution.** The **initial proposal** emphasized **multimodal** grounding (text + **chest imaging** concepts) and **local** fine-tuning of a **vision–language** model. As constraints and **data** became clear, the **emphasis** shifted to a **text-first** pipeline that could cover **all** **MMCQSD** **condition** groups. An **updated** presentation title used in coursework is **“Grounded RAG Pipeline for Code-Switching (Hinglish) in Clinical Decision Support Systems.”** Both titles are **valid** in different contexts: **multimodal** describes **long-term** intent; **pipeline** describes the **evaluated** artifact.

**What “clinical decision support” means here.** The system is intended to help **interpret** or **explain** in **patient-facing** language **conditioned** on **retrieved** literature-style cases. It is **not** validated as a **diagnostic** tool and must **not** be deployed as **medical advice** without **regulatory** and **clinical** oversight ([§22](#22-ethics-safety-and-positioning)).

---

## 5. Objectives and research questions

**Objective 1 — Engineering.** Implement a **repeatable** **RAG** stack: **encoding**, **indexing**, **retrieval**, **generation**, and **scoring**, with **configuration** suitable for **CPU-only** embedding and **cloud** LLM inference.

**Objective 2 — Empirical (H1).** **Quantify** whether, for the **same** Hinglish query, **injecting** **retrieved** English case text **improves** **evidence-aligned** concept overlap and **reduces** **unsupported** concepts **relative** to **zero-shot** generation, using **paired** **non-parametric** tests.

**Objective 3 — Empirical (H2).** **Stratify** by **code-mixing** intensity and ask whether **grounded** **quality** or **grounding gain** **depends** on how **Hindi-heavy** the query is. The **hypothesis** is framed so that **lack** of **degradation** under **mixing** is **scientifically** **interesting** (robustness), not a claim that **zero-shot** necessarily **worsens** with **CMI** (that pattern is **not** the **main** **reported** result).

**Objective 4 — Dissemination.** Provide a **Streamlit** demo so **stakeholders** can **see** **retrieval**, **answers**, and **scores** **without** running batch scripts.

**Exploratory objective (Phase 6).** Compare **raw** narrative **chunks** vs **LLM-structured** **evidence** summaries **before** generation, holding the rest of the stack fixed, to see if **structure** **helps** **grounding**.

**Deferred objective (H3).** Compare **retrieval** from **narrowly** **clinical** **case** text vs **broader** **biomedical** text **sources** under **controlled** conditions.

---

## 6. Problem statement and motivation

**The language gap.** Hospital **notes** and **published** **case** **reports** are overwhelmingly **English**. Patients in India often **do not** speak in that register; they **code-switch** between **Hindi** and **English**, **borrow** **medical** **words**, and use **metaphor** or **idiom** (“**chest** **me** **pani**”-type expressions in other domains). A model that **only** **expects** **clean** **English** may **mis-parse** intent or **retrieve** **wrong** **documents**.

**Why this is not “just translation”.** A **literal** **translation** to English **does not** **guarantee** **correct** **clinical** **retrieval**: the **target** **evidence** **must** **match** **anatomy**, **specialty**, and **phenomenology**. The project frames the task as **alignment**: map **informal** **mixed** **language** to **passages** that **actually** **support** or **constrain** what the **LLM** **says**.

**Risk of ungrounded fluency.** Generative models can **sound** **authoritative** while **inventing** **drugs**, **diagnoses**, or **findings**. **Retrieval-augmented** **generation** is one **standard** **mitigation**: **force** the **model** to **ground** **claims** in **supplied** **text**. This project **tests** whether that **mitigation** **transfers** to **Hinglish** **queries** when **evidence** is **English**.

---

## 7. Scope

**In scope.**

- **Text** **evidence** only in the **evaluated** **system**: **MultiCaRe**-style **case** **narratives**, **filtered** to **specialties** that **overlap** **MMCQSD** **conditions**.
- **Cross-lingual** **retrieval** from **Hinglish** **queries** to **English** **passages** using **LaBSE** **embeddings** and **FAISS**.
- **Adaptive** **truncation** of **retrieval** **lists** based on **similarity** **scores** (implementation follows the **spirit** of **MMed-RAG**-style **context** **selection**).
- **Paired** **evaluation** **modes**: **zero-shot** vs **grounded** on **identical** **queries**.
- **Automated** **metrics** based on **lexicon**/**pattern** **concept** **tags** and **simple** **negation** **rules**.
- **Offline** **batch** **experiments** (prototype scripts) and **online** **demo** (**Streamlit**).

**Explicitly out of scope for the current milestone.**

- **Regulatory** **clearance** or **clinical** **trial** **design**.
- **Real-time** **EHR** **integration** or **individual** **patient** **records**.
- **Multimodal** **image** **retrieval** in the **production** **evaluated** **path**.
- **Fine-tuning** or **DPO** **training** of the **production** **generator** (the **API** **model** is **used** **as-is** with **prompting**).

**Boundary on “diagnosis”.** The **software** **does** **not** **output** a **formal** **diagnosis** **product**; **wording** in **generated** **text** must still be **reviewed** for **safety** in **any** **user-facing** **deployment**.

---

## 8. Hypotheses

**H1 (grounding effect).** *For the same Hinglish query, text-grounded RAG produces higher factual support scores and lower hallucination scores than zero-shot generation, when both are scored against the same retrieved evidence text.*

**Operationalization.** **Paired** **Wilcoxon** **signed-rank** tests on **per-query** **differences**; **Cohen’s d** for **effect** **size**; **95%** **confidence** **interval** on **mean** **factual** **gain**.

**H2 (code-mixing robustness).** *Grounding-related outcomes do not show strong systematic deterioration as code-mixing increases across tertiles of CMI in this dataset; tests for differences in gains across tertiles are not statistically significant at conventional levels.*

**Operationalization.** **Kruskal–Wallis** on **factual** **gain** and **hallucination** **reduction** **across** **three** **CMI** **bins**; **Mann–Whitney U** **pairwise** **comparisons** with **Bonferroni** **adjustment**; **Spearman** **correlation** **between** **continuous** **CMI** and **outcomes**.

**H3 (evidence provenance — deferred).** *The type of corpus (e.g., authoritative case reports vs general biomedical abstracts) changes grounded answer quality when retrieval depth and prompting are held constant.*

**Status.** **Not** **executed** as a **primary** **result** in the **documented** **semester** **milestone**; listed under [§21](#21-future-scope).

| ID | One-line statement | Primary tests |
|----|-------------------|---------------|
| H1 | Grounding beats zero-shot on paired factual/hallucination metrics. | Wilcoxon, Cohen’s d, 95% CI |
| H2 | No strong evidence that gains differ across CMI tertiles. | Kruskal–Wallis, Mann–Whitney, Spearman |
| H3 | Evidence source matters (planned). | Deferred |

---

## 9. Methodology overview

This section defines **terms** and the **experimental** **logic** before the **step-by-step** **workflow** in [§10](#10-end-to-end-workflow-detailed).

### 9.1 Study design

The **unit of analysis** is **one** **Hinglish** **query** (and its **paired** **generations**). **Each** **query** receives **two** **treatments**: **(A)** **zero-shot**—prompt **without** **evidence** **passages**; **(B)** **grounded**—prompt **with** **top** **retrieved** **passages**. **Order** **does** **not** **affect** **scores** because **generations** are **independent** **calls**; **pairing** is **preserved** **statistically**.

### 9.2 Encoder (LaBSE)

**LaBSE** (**Language-agnostic BERT Sentence Embedding**) maps **variable-length** **text** to **768-dimensional** **vectors**. It is **trained** for **cross-lingual** **semantic** **similarity**, which is **why** a **Hinglish** **query** can **lie** **near** **relevant** **English** **clinical** **paragraphs** in **vector** **space**. The **implementation** uses **`sentence-transformers/LaBSE`** on **CPU** by **default** (`TextEncoder`).

### 9.3 Vector index (FAISS)

**FAISS** stores **precomputed** **embeddings** of **evidence** **chunks** (aligned **rows** in **metadata** **CSV**). **Search** uses **inner** **product** on **L2-normalized** **vectors** (equivalent to **cosine** **similarity** for **normalized** **inputs**). The **demo** **index** **documented** in **summaries** contains **10,000** **vectors**; the **full** **filtered** **corpus** **count** used in **project** **documentation** is **61,316** **cases** **before** **sampling** **for** **some** **encoding** **steps**—see `project-complete-summary.md` **§6** for **batching** and **truncation** **details**.

### 9.4 Retriever and adaptive truncation

The **retriever** **embeds** the **query**, **queries** **FAISS** for **up** **to** **`max_k`** **neighbors**, and **returns** **structured** **records** (**case_id**, **condition_group**, **case_text**, **score**). If the **API** **passes** **`top_k=None`**, the **code** **can** **apply** **adaptive** **truncation**: **keep** **the** **longest** **prefix** of **results** **before** a **sharp** **drop** in **similarity** **score**, **discarding** **weak** **tail** **matches** that **often** **hurt** **faithfulness** in **RAG** **systems**.

### 9.5 Generator (Groq + Llama-3.1-8B-Instant)

**Groq** hosts **fast** **inference** for **open** **weights** **models**. The **project** uses **`llama-3.1-8b-instant`** (see **`GroundedGenerator`** / **`RAGPipeline`**). **Prompts** **instruct** the **model** to **respond** in **Hinglish** **while** **using** **evidence** **when** **grounded**.

### 9.6 Automated scoring

**Medical** **concepts** are **detected** by **substring** **patterns** **grouped** into **categories** (**rash**, **fever**, **pain**, **swelling**, etc.) in `src/pipeline.py`. **Negation** **phrases** (**no**, **without**, **not**, **nahi**) **suppress** **false** **positives** **near** **hits**.

- **Factual support** = (concepts in **answer** ∩ concepts in **evidence**) / (concepts in **answer**), or **0.25** if **no** **concepts** **fire** in the **answer** (**implementation** **default**).
- **Hallucination** score = (concepts in **answer** **not** in **evidence**) / (concepts in **answer**), or **0** if **no** **concepts** in **answer**.

These are **proxies**, not **clinician** **judgments** ([§19](#19-limitations)).

---

## 10. End-to-end workflow (detailed)

### 10.1 Offline: corpus preparation, pairing, and index build

**Step 1 — Obtain MultiCaRe-derived tabular data.** Raw **MultiCaRe** is **large**; the **team** **filters** to **specialties** and **condition** **groups** that **align** with **MMCQSD** (**18** **labels**).

**Step 2 — Truncate and batch text for encoding.** Long **case** **reports** are **cut** to **word** **limits** **suitable** **for** **RAM** and **LaBSE** (**documentation** cites **~200** **words** for **encoding** and **~400** for **LLM** **context** in **summaries**—verify **in** **`build_index.py`** / **configs** **if** **exact** **numbers** **matter** **for** **replication**).

**Step 3 — Encode evidence with LaBSE.** **Embeddings** are **computed** in **batches** (e.g. **batch_size=32** per **summary**) on **CPU**.

**Step 4 — Build FAISS index and metadata.** **`build_index.py`** writes **`evidence.index`** and **`evidence_metadata.csv`** under **`data/faiss_index/`**, with **row** **alignment** **between** **vectors** and **metadata**.

**Step 5 — Encode MMCQSD queries and pair to evidence.** For **evaluation** **datasets**, **each** **query** is **embedded**, **searched** **with** **condition-aware** **filtering** so **a** **dermatology** **question** **does** **not** **retrieve** **irrelevant** **specialties**. This yields **3,015** **query–evidence** **rows** (**full** **MMCQSD** **coverage**).

**Step 6 — LLM scoring passes.** **Subset** **runs** **produced** **1,165** **clean** **paired** **scored** **rows** **after** **merging** **multi-day** **Groq** **runs** (**rate** **limits**).

### 10.2 Online: one query through `RAGPipeline.query` (Streamlit or API)

When a **user** **submits** a **string**, **`RAGPipeline.query`** **executes**:

1. **`EvidenceRetriever.retrieve`** embeds the **query** and **returns** a **list** of **evidence** **dicts**.
2. **`GroundedGenerator.generate`** builds the **grounded** **prompt** and **returns** **Hinglish** **text**.
3. If **`include_zero_shot`**, **`generate_zero_shot`** **runs** **second**.
4. **Evidence** **texts** are **joined** **into** **one** **string** **for** **scoring**.
5. **`factual_support_score`** and **`hallucination_score`** **run** **on** **each** **answer** **against** **that** **combined** **evidence**.

**Why both modes use the same evidence string for scoring.** The **evaluation** **asks**: “**If** **these** **are** **the** **passages** **a** **clinician** **might** **have** **retrieved**, **how** **much** **does** **each** **answer** **respect** **them?**” **Zero-shot** **typically** **scores** **worse** **because** it **was** **not** **trained** **in** **this** **call** **to** **use** **those** **passages**—that **is** **the** **intended** **contrast**.

### 10.3 Worked example: relatable query, retrieved evidence, and two outputs

This subsection walks through **one** **realistic** **scenario** using **language** **patterns** **documented** in **project** **poster** **materials**. The **exact** **similarity** **score**, **`case_id`**, and **verbatim** **retrieved** **paragraph** **depend** on **your** **built** **index** **version**; the **grounded** and **zero-shot** **answers** below are **taken** from the **documented** **evaluation** **example** for a **spreading** **itchy** **rash** (see `project-complete-summary.md` **§15**, Example 1). An **illustrative** **English** **evidence** **excerpt** is **provided** to show **what** **kind** **of** **text** **conditions** the **grounded** **model**; it is **representative** **of** **MultiCaRe** **narrative** **style**, **not** **guaranteed** to be **byte-identical** to **your** **local** **top-1** **hit**.

---

#### User input (Hinglish — relatable complaint)

> *“Mujhe ek samasya hai mere skin me jo mere legs par develop hui aur ab mere arms aur wrists par hai, woh bahut khujali kar sakti hai, woh lal dots jaise hain.”*  
> *(Rough sense: A skin problem started on my legs and spread to arms and wrists; it itches a lot; looks like red dots.)*

This is **typical** of **how** **people** **describe** **rashes** in **mixed** **Hindi–English**: **body** **parts** **in** **English**/**Hindi**, **symptoms** **blended**, **no** **single** **ICD** **code**.

---

#### Stage A — Encoding

**LaBSE** maps the **query** to a **768-dimensional** **vector**. **Intuition:** **dimensions** **encode** **semantic** **features** **shared** **across** **languages**, so **“lal** **dots**”, **“khujali”**, **“skin”** **jointly** **point** **toward** **dermatology**/**inflammatory** **skin** **content** in **English** **case** **text**.

---

#### Stage B — Retrieval (representative evidence)

**FAISS** **returns** **ranked** **case** **narratives** from the **index** (e.g. **top-k** **3** in the **demo**). **Illustrative** **retrieved** **passage** (**English**, **abbreviated** **for** **documentation**):

> *Representative MultiCaRe-style excerpt (illustrative): A young adult presented with a pruritic erythematous eruption beginning on the lower extremities and extending to the upper limbs. Laboratory studies showed neutrophilia (neutrophil count 9.9 × 10⁹/L). Physical examination documented scattered maculopapular lesions. Further work-up and management followed standard inpatient protocols…*

**What the UI would show.** **Expanders** **per** **hit**: **`condition_group`** (e.g. **skin**/**rash** **bucket**), **`case_id`**, **LaBSE** **similarity** **score**, and a **truncated** **`case_text`** **excerpt**.

---

#### Stage C — Grounded generation (documented project output)

The **model** **sees** the **Hinglish** **question** **plus** **the** **retrieved** **excerpts** **(not** **shown** **in** **full** **here)**. **Documented** **grounded** **answer** **(excerpt)** from **poster** **materials**:

> *“Didi, aapki samasya ka sabse pehle kya hai, yehi dekhna hai. Aapki blood report mein neutrophil count 9.9 hai, jo thoda high hai… Lekin aapki skin par lal dots aur khujali hai…”*

**Automated scores** **reported** **for** **this** **example** **pair**: **factual** **support** **1.00**; **hallucination** **0.00** — **concepts** **in** **the** **answer** **overlap** **heavily** **with** **concepts** **detected** **in** **the** **evidence** **string** **used** **for** **that** **run**.

---

#### Stage D — Zero-shot generation (documented project output)

**Same** **model**, **no** **evidence** **in** **prompt**. **Documented** **zero-shot** **answer** **(excerpt)**:

> *“Bhai, mujhe lagta hai ki aapko ek skin problem hai, jiska naam ‘urticaria’ ho sakta hai…”*

**Scores** **for** **this** **documented** **example**: **factual** **0.00**; **hallucination** **1.00** — **the** **system** **flags** **many** **stated** **concepts** **as** **unsupported** **by** **the** **retrieved** **evidence** **bundle** **for** **that** **pair** **(the** **label** **“urticaria”** **may** **not** **appear** **in** **evidence** **under** **the** **lexicon** **rules**).

---

#### Stage E — Side-by-side comparison (what the demo emphasizes)

| Mode | Role of evidence | Typical pattern in this project’s examples |
|------|------------------|--------------------------------------------|
| **Grounded** | **Passages** **in** **prompt** **steer** **the** **model** **toward** **findings** **mentioned** **in** **cases** | **Higher** **factual** **support**, **lower** **hallucination** **on** **aggregate** (**H1**) |
| **Zero-shot** | **No** **passages**; **parametric** **knowledge** **only** | **More** **generic** **or** **label-heavy** **answers** **that** **may** **diverge** **from** **any** **specific** **retrieved** **case** |

---

#### Logical flow diagram (same pipeline, now grounded in the example above)

```
Hinglish query  ──►  LaBSE  ──►  768-d query vector
                                      │
                                      ▼
                    FAISS search on indexed MultiCaRe passages
                                      │
                                      ▼
              Ranked English case excerpts (scores + metadata)
                    │                              │
                    ▼                              │
         Prompt + evidence ──► Groq Llama-3.1-8B ──┼──► Grounded Hinglish answer
                    │                              │
                    │                              └──► (parallel) Zero-shot Hinglish answer
                    ▼
    Concatenate evidence text as scoring context
                    │
                    ▼
 For each answer: concept overlap → factual support & hallucination scores
                    │
                    ▼
           Streamlit: two columns + metrics + evidence expanders
```

---

## 11. Architecture (implemented)

The **runtime** **architecture** is a **linear** **pipeline** **with** **one** **branch** **for** **zero-shot**:

1. **Input:** **Unicode** **Hinglish** **string** from **user** **or** **dataset**.  
2. **Encoder:** **LaBSE** **CPU** **inference**.  
3. **Index** **search:** **FAISS** **`IndexFlatIP`** over **prebuilt** **vectors**.  
4. **Context** **shaping:** **Fixed** **top-k** **(UI** **slider)** **or** **adaptive** **cut**.  
5. **Generator:** **HTTP** **call** **to** **Groq** **with** **system**/**user** **prompts** **that** **embed** **evidence** **strings**.  
6. **Optional** **second** **call:** **zero-shot** **prompt**.  
7. **Scoring:** **Deterministic** **Python** **functions** **on** **strings**.  
8. **Output:** **JSON-like** **dict** **to** **UI** **or** **batch** **logger**.

**Contrast** **with** **original** **proposal** **architecture** **(high** **level).** **Open-i** **chest** **reports** **and** **BioMedCLIP** **were** **replaced** **by** **MultiCaRe** **text** **and** **LaBSE-only** **retrieval** **because** **MMCQSD** **covers** **many** **specialties** **and** **hardware** **did** **not** **support** **local** **7B+** **multimodal** **fine-tuning**. **QLoRA/DPO** **were** **not** **applied** **to** **the** **shipping** **generator**; **they** **remain** **future** **work** ([§21](#21-future-scope)).

---

## 12. Data sources and pairing pipeline

**MMCQSD** (**Multimodal** **Medical** **Code-mixed** **Question** **Summarization** **Dataset**) provides **3,015** **real** **patient-style** **queries** **in** **Hinglish** **(Roman** **script)**, **with** **associated** **metadata** **used** **for** **condition** **stratification** **and** **CMI**. **Despite** **“multimodal”** **in** **the** **name**, **the** **RAG** **milestone** **primarily** **consumes** **the** **textual** **query** **stream**.

**MultiCaRe** aggregates **open-access** **clinical** **case** **reports** **from** **PubMed** **Central**-class **sources** **(see** **dataset** **documentation** **for** **licensing)**. **The** **team** **filters** **from** **roughly** **93k+** **raw** **records** **down** **to** **61,316** **cases** **judged** **relevant** **to** **MMCQSD** **themes**, **then** **samples** **or** **truncates** **for** **encoding** **feasibility**.

**Pairing logic.** **For** **each** **MMCQSD** **row**, **the** **pipeline** **retrieves** **English** **evidence** **constrained** **by** **condition** **compatibility**, **producing** **3,015** **aligned** **pairs** **with** **mean** **LaBSE** **similarity** **≈** **0.50** **and** **roughly** **half** **of** **pairs** **above** **0.50** **similarity** **(high** **quality** **bucket** **in** **project** **reports)**.

**Eighteen** **condition** **groups** **include**, **among** **others**: **skin_rash**, **mouth_ulcers**, **neck_swelling**, **eye_redness**, **cyanosis**, **etc.** **(full** **list** **in** **`project-complete-summary.md`** **§6.3).**

**Key** **code** **entry** **points:** **`src/matching/pair_builder.py`**, **`build_index.py`**, **`src/retrieval/`**, **`data/processed/`**.

---

## 13. Critical limitation and resolution (Open-i → MultiCaRe)

**Limitation encountered.** The **first** **prototype** **aligned** **MMCQSD** **queries** **to** **Open-i** **chest** **X-ray** **reports** **using** **TF-IDF**-style **matching**. **Only** **~11** **usable** **pairs** **emerged**. **Root** **cause:** **not** **a** **bug** **in** **matching** **code** **but** **domain** **mismatch**—**most** **MMCQSD** **questions** **concern** **dermatology**, **ENT**, **oral** **medicine**, **etc.**, **while** **Open-i** **is** **radiology**-**centric** **chest** **imaging** **text**. **~96%** **of** **queries** **had** **negligible** **topical** **overlap** **with** **chest** **reports**.

**Resolution.** **Switch** **evidence** **to** **MultiCaRe** **case** **narratives** **spanning** **multiple** **specialties**, **and** **replace** **bag-of-words** **matching** **with** **LaBSE** **+** **FAISS** **semantic** **search** **plus** **condition** **filters**.

**Outcome (documented).** **3,015** **pairs** **(100%** **MMCQSD** **coverage)**, **18/18** **condition** **categories** **supported**, **~274×** **more** **pairs** **than** **the** **Open-i** **attempt**, **mean** **similarity** **~0.50**.

---

## 14. Evaluation methodology

### 14.1 Outcome metrics

**Factual support** **[0,** **1]** measures **what** **fraction** **of** **clinical** **concept** **tags** **found** **in** **the** **model** **answer** **also** **appear** **in** **the** **evidence** **text** **(after** **negation** **filtering).** **Higher** **is** **better**.

**Hallucination** **score** **[0,** **1]** measures **what** **fraction** **of** **tags** **in** **the** **answer** **do** **not** **appear** **in** **the** **evidence**. **Lower** **is** **better**.

**Derived** **quantities:** **factual** **gain** **=** **grounded** **factual** **−** **zero-shot** **factual**; **hallucination** **reduction** **=** **zero-shot** **hallucination** **−** **grounded** **hallucination**.

**Code-mixing** **index** **(CMI).** **Ratio** **of** **Hindi-origin** **tokens** **(from** **a** **curated** **~100-word** **list)** **to** **total** **Latin-alphabet** **tokens**. **Values** **near** **0** **mean** **more** **English**; **near** **1** **mean** **heavier** **Hindi** **in** **romanization**. **Queries** **are** **split** **into** **tertiles** **for** **group** **tests**.

### 14.2 Statistical procedures

- **H1:** **Wilcoxon** **signed-rank** **on** **paired** **differences**; **Cohen’s** **d**; **bootstrap**/**analytic** **95%** **CI** **on** **mean** **factual** **gain** **(documented** **interval** **[0.211,** **0.258]).**  
- **H2:** **Kruskal–Wallis** **across** **three** **CMI** **groups**; **Mann–Whitney** **U** **pairwise** **with** **Bonferroni** **correction**; **Spearman** **ρ** **for** **continuous** **CMI**.

---

## 15. Results summary

**Analytic** **sample:** **n** **=** **1,165** **paired** **observations** **after** **combining** **cleaned** **evaluation** **runs** **(see** **`project-complete-summary.md`** **§8.1** **for** **run** **ledger).** **Primary** **generator** **documented** **as** **Llama-3.1-8B-Instant** **via** **Groq**; **some** **early** **rows** **may** **use** **Llama-3.3-70B** **per** **team** **notes**—**treat** **1,165** **as** **the** **canonical** **H1/H2** **sample** **size**.

### H1 (grounding vs zero-shot)

| Metric | Zero-shot mean | Grounded mean | Delta |
|--------|----------------|---------------|-------|
| Factual support | 0.319 | 0.554 | +0.235 (~+73.5%) |
| Hallucination | 0.500 | 0.280 | −0.220 (~−44%) |

**Wilcoxon** **p** **≈** **3.09** **×** **10⁻⁶⁴** **(factual)** **and** **≈** **5.33** **×** **10⁻⁵¹** **(hallucination** **reduction).** **Cohen’s** **d** **≈** **0.576** **and** **0.492** **respectively.**

**Verdict:** **H1** **is** **strongly** **supported**.

### H2 (across CMI tertiles)

**Grounded** **factual** **means** **by** **tertile:** **~0.554,** **0.544,** **0.563** **(low** **→** **high** **CMI).** **Factual** **gain** **increases** **slightly** **from** **low** **to** **high** **CMI** **(+0.202** **→** **+0.260)** **but** **Kruskal–Wallis** **on** **gain** **gives** **H** **=** **3.879,** **p** **=** **0.144** **(not** **significant).** **Pairwise** **Mann–Whitney** **tests** **(Bonferroni)** **do** **not** **show** **significant** **pair** **differences.**

**Verdict:** **No** **statistical** **evidence** **that** **grounding** **benefits** **disappear** **at** **higher** **code-mixing**; **the** **system** **remains** **effective** **across** **tertiles** **in** **this** **dataset**.

### Communication caution

Do **not** **assert** **that** **“non-grounded** **models** **degrade** **rapidly** **as** **Hindi** **increases”** **unless** **you** **plot** **that** **explicitly** **from** **your** **CSV**. **The** **supported** **story** **is** **stable** **grounded** **performance** **and** **non-significant** **differences** **in** **gains** **across** **CMI** **bins**.

---

## 16. Phase 6 ablation (evidence format)

**Sample:** **n** **=** **401** **pairs.**

**Comparison:** **Raw** **MultiCaRe** **narrative** **snippets** **vs** **LLM-extracted** **structured** **evidence** **blocks** **fed** **into** **the** **same** **generator**.

**Results** **(means):** **Grounded** **factual** **0.571** **→** **0.639** **(+0.069);** **grounded** **hallucination** **0.240** **→** **0.196** **(−0.044).** **Both** **settings** **achieve** **highly** **significant** **Wilcoxon** **p-values**; **structured** **evidence** **increases** **Cohen’s** **d**.

**Interpretation:** **Structured** **evidence** **helps**, **but** **raw** **narrative** **alone** **already** **enables** **strong** **grounding**—consistent **with** **practical** **deployment** **where** **structuring** **may** **cost** **extra** **LLM** **calls**.

---

## 17. Implementation and reproducibility

**Environment.** **Python** **3.x** **with** **dependencies** **from** **the** **repository** **requirements** **(install** **via** **`pip`** **or** **`uv`** **as** **you** **standardize** **locally).**

**Secrets.** **`GROQ_API_KEY`** **in** **`.env`** **(loaded** **by** **`python-dotenv`** **in** **`app.py`).**

**Index** **build.** **From** **repository** **root:** **`python` `build_index.py`** **(expect** **~10–20** **minutes** **on** **CPU** **per** **team** **notes).** **Output:** **`data/faiss_index/evidence.index`** **and** **`evidence_metadata.csv`.**

**Launch** **demo.** **`streamlit` `run` `app.py`** **or** **`python` `-m` `streamlit` `run` `app.py`** **on** **Windows** **if** **`streamlit`** **is** **not** **on** **PATH.** **Non-interactive** **servers** **may** **set** **`CI=true`** **and** **`STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`** **to** **skip** **first-run** **prompts**.

**Hardware** **constraints** **documented** **in** **summaries:** **no** **CUDA**; **~7.8** **GiB** **RAM** **forces** **small** **batches** **and** **truncation**; **Groq** **free** **tier** **motivated** **multi-day** **evaluations** **with** **key** **rotation**.

---

## 18. Streamlit demo

The **demo** **implements** **the** **full** **user** **journey:** **pick** **an** **example** **Hinglish** **query** **or** **type** **your** **own**, **choose** **`top_k`**, **optionally** **enable** **zero-shot** **comparison**, **run** **inference**, **then** **inspect** **expandable** **evidence** **cards** **and** **read** **side-by-side** **answers** **with** **numeric** **metrics**.

**Example** **buttons** **in** **code** **include** **child** **rash**, **eye** **swelling**, **tonsillar** **pain**, **leg** **swelling**, **skin** **growth**, **and** **oral** **ulcers**—**everyday** **complaints** **families** **recognize**.

---

## 19. Limitations

1. **Linguistic** **generality:** **Only** **Hinglish** **is** **studied**; **other** **Indian** **code-mixed** **languages** **need** **new** **data** **and** **evaluation**.

2. **Evidence** **type** **and** **modality:** **PMC-style** **case** **text** **is** **not** **the** **same** **as** **a** **hospital** **EHR** **stream**; **no** **imaging** **in** **the** **evaluated** **RAG** **path**.

3. **Retrieval** **variance:** **Small** **rewordings** **can** **change** **which** **cases** **surface**, **altering** **both** **answers** **and** **scores**.

4. **Model** **and** **vendor** **lock-in:** **No** **in-house** **fine-tuning** **of** **the** **API** **model**; **latency**, **cost**, **and** **availability** **depend** **on** **Groq**.

5. **Metric** **validity:** **Pattern-based** **concept** **scores** **are** **rough** **proxies**; **they** **do** **not** **measure** **clinical** **correctness**, **patient** **comprehension**, **or** **harm**. **No** **prospective** **clinician** **study** **is** **included**.

6. **Clinical** **context:** **No** **longitudinal** **history**, **medications**, **allergies**, **or** **demographics** **in** **the** **demo** **pipeline** **by** **default**.

7. **Positioning:** **Outputs** **must** **not** **be** **sold** **as** **diagnosis** **or** **treatment** **plans**.

---

## 20. Impact and significance

**Scientific** **impact.** **Demonstrates** **that** **off-the-shelf** **cross-lingual** **embeddings** **plus** **dense** **retrieval** **can** **bridge** **Hinglish** **queries** **to** **English** **clinical** **narratives** **at** **scale**, **with** **large** **measured** **gains** **from** **grounding** **under** **automated** **metrics**.

**Practical** **impact.** **Offers** **a** **reproducible** **blueprint** **for** **teams** **building** **patient-facing** **assistants** **in** **code-mixed** **South** **Asian** **languages** **when** **only** **English** **reference** **text** **exists**.

**Social** **impact.** **Centers** **linguistic** **reality** **of** **Indian** **patients** **rather** **than** **assuming** **English** **fluency**—**a** **step** **toward** **more** **inclusive** **clinical** **NLP** **if** **paired** **with** **proper** **governance**.

**Limits** **of** **impact** **claims.** **Automated** **metric** **gains** **do** **not** **equal** **regulatory-grade** **safety** **or** **clinical** **utility** **until** **validated** **with** **experts** **and** **users**.

---

## 21. Future scope

- **Multilingual** **expansion** **beyond** **Hinglish** **with** **new** **corpora** **and** **CMI-like** **metrics** **per** **language**.  
- **Multimodal** **RAG** **(BioMedCLIP-class** **encoders,** **dermatology** **images,** **chest** **imaging)** **once** **data** **and** **compute** **allow**.  
- **QLoRA** **/** **LoRA** **and** **DPO** **(or** **RLHF)** **to** **specialize** **style** **and** **reduce** **unsupported** **content** **without** **full** **fine-tunes** **of** **massive** **models** **on** **consumer** **GPUs**.  
- **H3** **experiments** **contrasting** **clinical** **case** **banks** **vs** **general** **biomedical** **retrieval**.  
- **Clinician** **and** **patient** **studies** **for** **trust**, **comprehension**, **and** **workflow** **fit**.  
- **Governance** **artifacts** **(logging,** **audit** **trails,** **uncertainty** **disclosure,** **bias** **audits).**  
- **Full** **3,015-pair** **LLM** **coverage** **and** **richer** **metrics** **(e.g.** **MMFCM-class** **scores)** **if** **resources** **permit**.

---

## 22. Ethics, safety, and positioning

**Non-diagnostic** **intent.** **Documentation** **and** **demo** **are** **research** **artifacts**. **They** **do** **not** **constitute** **medical** **device** **labeling** **or** **approval**.

**Data** **ethics.** **Use** **public** **datasets** **according** **to** **their** **licenses**. **Do** **not** **paste** **real** **patient** **PHI** **into** **the** **demo**.

**Safe** **communication.** **Any** **user-facing** **deployment** **requires** **disclaimers**, **escalation** **paths** **to** **human** **care**, **and** **jurisdictional** **compliance**.

---

## 23. Repository map (key paths)

```
hinglish-grounded-medical-rag/
├── app.py                      # Streamlit entrypoint
├── build_index.py              # Builds FAISS index + metadata
├── src/
│   ├── encoding/text_encoder.py    # LaBSE wrapper
│   ├── retrieval/indexer.py        # FAISS load/search
│   ├── retrieval/retriever.py      # Adaptive truncation logic
│   ├── generation/generator.py     # Groq client + prompts
│   ├── pipeline.py                 # RAGPipeline + concept scores
│   ├── matching/pair_builder.py    # Offline pairing utilities
│   └── prototype/                  # Batch eval + stats + ablation
├── data/processed/             # Filtered CSVs (as committed/generated)
├── data/faiss_index/             # evidence.index, evidence_metadata.csv
├── research-poster-work/
│   ├── project-complete-summary.md
│   ├── PROJECT_FULL_DOCUMENTATION.md   # this file
│   └── generate_plots.py
└── results/                      # Often gitignored
```

---

## 24. Glossary

| Term | Definition |
|------|------------|
| **Code-mixing** | Using **two** **languages** **in** **one** **utterance** **(here** **Hindi** **+** **English** **in** **Roman** **script).** |
| **CMI** | **Code-mixing** **index** — **Hindi-lexicon** **token** **count** **/** **Latin-token** **count**. |
| **FAISS** | **Library** **for** **efficient** **similarity** **search** **over** **dense** **vectors**. |
| **Groq** | **Inference** **API** **provider** **used** **for** **Llama-3.1-8B-Instant**. |
| **Hinglish** | **Hindi–English** **code-switched** **colloquial** **register** **common** **in** **urban** **India**. |
| **LaBSE** | **Multilingual** **sentence** **embedding** **model** **family** **suited** **to** **cross-lingual** **retrieval**. |
| **MMCQSD** | **Dataset** **of** **code-mixed** **medical** **queries** **(and** **related** **assets).** |
| **MultiCaRe** | **Large** **clinical** **case** **report** **corpus** **used** **as** **English** **evidence**. |
| **RAG** | **Retrieval-augmented** **generation** — **retrieve** **text** **then** **condition** **an** **LLM** **on** **it**. |
| **Zero-shot** | **Generation** **without** **supplying** **retrieved** **passages** **in** **the** **prompt** **for** **that** **call**. |

---

## Appendix: numeric quick reference

| Quantity | Value |
|----------|------:|
| MMCQSD queries | 3,015 |
| Filtered MultiCaRe cases (documented) | 61,316 |
| Demo FAISS vectors (typical) | 10,000 |
| H1/H2 paired scored queries | 1,165 |
| Phase 6 ablation pairs | 401 |
| LaBSE embedding dimension | 768 |
| Default LLM id in code | llama-3.1-8b-instant |
| Mean LaBSE pair similarity (documented) | ~0.500 |

---

*End of document. Poster figure file names and additional qualitative examples: `project-complete-summary.md` §13–§16 and §15.*
