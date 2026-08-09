# Overcoming the Weak Dataset Pairing Limitation — MultiCaRe Solution Plan

## 1. The Limitation

### What is the problem

The current prototype pairs two datasets that were never designed to work together:

| Dataset | What it contains | Medical scope |
|---|---|---|
| **Open-i** | 6,687 English chest X-ray radiology reports | Chest only — lungs, heart, pleural space |
| **MMCQSD** | 3,015 Hinglish patient queries + English summaries | Multi-specialty — skin, eye, ENT, oral, musculoskeletal, etc. |

The result: only **11 usable pairs** after aggressive filtering. This is insufficient for training, evaluation, or any statistically meaningful hypothesis testing.

### Why it happens — root cause

MMCQSD's actual condition distribution:

| Condition | Count | % of dataset | Overlaps with Open-i? |
|---|---:|---:|---|
| skin rash | 1,050 | 34.8% | No |
| neck swelling | 276 | 9.2% | No |
| mouth ulcers | 196 | 6.5% | No |
| lip swelling | 193 | 6.4% | No |
| swollen tonsils | 174 | 5.8% | No |
| foot swelling | 172 | 5.7% | No |
| hand lump | 162 | 5.4% | No |
| swollen eye | 152 | 5.0% | No |
| knee swelling | 115 | 3.8% | No |
| edema | 112 | 3.7% | Partially |
| eye redness | 92 | 3.1% | No |
| skin growth | 81 | 2.7% | No |
| skin irritation | 77 | 2.6% | No |
| skin dryness | 65 | 2.2% | No |
| dry scalp | 44 | 1.5% | No |
| eye inflammation | 25 | 0.8% | No |
| cyanosis | 15 | 0.5% | Partially |
| itchy eyelid | 14 | 0.5% | No |

**Only 127 out of 3,015 MMCQSD rows (4.2%)** have even a partial overlap with Open-i's chest radiology domain. The problem is a fundamental **domain mismatch**, not a code quality issue.

### What this blocks

| Blocked area | Why |
|---|---|
| Training the RAG system | Cannot train on 11 examples |
| Evaluating H1 reliably | n=11 makes any statistical test fragile |
| Testing H2 (CMI robustness) | Need sufficient samples at each CMI level |
| Testing H3 (evidence type comparison) | Need enough pairs to split across evidence conditions |
| Streamlit demo | System cannot demonstrate meaningful results |
| Research paper | Results on 11 weakly-supervised pairs are not publishable |

---

## 2. Proposed Solution

### Core decision

Replace Open-i as the primary evidence corpus with **MultiCaRe** — a multi-specialty clinical case dataset whose medical specialties naturally align with MMCQSD's 18 condition categories.

### Why MultiCaRe

| Property | Detail |
|---|---|
| **Name** | MultiCaRe — Multi-Modal Clinical Dataset |
| **Source** | Open-access case reports from PubMed Central (1990–2023) |
| **Size** | 75,000+ articles, 93,000+ clinical cases, 130,000+ images |
| **Specialties** | Oncology, cardiology, dermatology, ophthalmology, surgery, pathology, ENT, oral, musculoskeletal, and more |
| **Image types** | 7 main types: radiology, pathology, medical_photograph, ophthalmic_imaging, endoscopy, chart, electrography |
| **Text** | Full clinical case narratives (patient presentation, findings, diagnosis, treatment) |
| **Labels** | 140+ hierarchical image classification classes |
| **License** | CC BY-NC-SA (free for academic research) |
| **Access** | Hugging Face, Zenodo, GitHub — no CITI certificate, no ethics approval needed |
| **Python library** | `multiversity` — filter and create custom subsets programmatically |

### Why MultiCaRe solves the limitation

| MMCQSD condition (3,015 queries) | MultiCaRe coverage |
|---|---|
| skin rash (1,050) | Dermatology cases (medical_photograph) |
| neck swelling (276) | Surgical / ENT cases |
| mouth ulcers (196) | Oral pathology cases |
| lip swelling (193) | Oral / dermatology cases |
| swollen tonsils (174) | ENT cases |
| foot swelling (172) | Vascular / orthopedic cases |
| hand lump (162) | Orthopedic / dermatology cases |
| swollen eye (152) | Ophthalmic cases (ophthalmic_imaging) |
| knee swelling (115) | Orthopedic cases |
| edema (112) | Cardiology / nephrology cases |
| eye redness (92) | Ophthalmic cases |
| skin growth, irritation, dryness (223) | Dermatology cases |
| dry scalp (44) | Dermatology cases |
| eye inflammation + itchy eyelid (39) | Ophthalmic cases |
| cyanosis (15) | Cardio-pulmonary cases |

**With Open-i:** 11 pairs (0.4% of MMCQSD matched).
**With MultiCaRe:** Potentially **hundreds to thousands** of pairs across all 18 conditions, because the specialties naturally align.

### Updated project scope

The project scope expands from:

> *"Evidence-grounded RAG for Hinglish clinical queries, grounded in chest radiology reports"*

To:

> *"Evidence-grounded RAG for Hinglish clinical queries, grounded in multi-specialty clinical evidence"*

This is a **stronger** research contribution. The core hypothesis (grounded > zero-shot) is the same. The architecture (LaBSE encoding → retrieval → grounded generation) is the same. Only the evidence corpus changes and broadens.

### Relationship to Open-i

| Decision | Rationale |
|---|---|
| Open-i is **not discarded** | It remains available as a supplementary chest-specific corpus |
| MultiCaRe becomes the **primary** evidence corpus | It covers all 18 MMCQSD conditions |
| Hybrid use is **optional for later** | Open-i can be re-integrated for chest-specific analysis if time permits |

---

## 3. The Evidence Transformation Pipeline

### Why transformation is needed

MultiCaRe contains **clinical case narratives** from academic publications. These are long, narrative paragraphs:

> *"A 28-year-old female presented with erythematous papular lesions on both forearms, accompanied by pruritus for 2 weeks. Dermatological examination revealed raised, well-defined plaques consistent with contact dermatitis..."*

For RAG grounding, this needs to be transformed into **structured clinical evidence** that a generator can reference:

```
Finding: Contact dermatitis (erythematous papular rash)
Location: Bilateral forearms
Symptoms: Pruritus, raised plaques
Severity: Mild-moderate
Duration: 2 weeks
```

### How the extraction works

Use a local LLM (Llama 3.1 8B via Ollama) to extract structured findings from each MultiCaRe case narrative.

**LLM prompt template:**

```
You are a medical evidence extractor. Given a clinical case report,
extract ONLY the factual clinical findings into a structured format.
Do NOT add any information not present in the case.

Case Report:
{case_narrative}

Extract in this format:
- Primary Finding: [main diagnosis or condition]
- Location: [body area affected]
- Symptoms: [patient-reported symptoms]
- Clinical Signs: [examination findings]
- Severity: [mild / moderate / severe / not specified]
- Duration: [how long symptoms have been present, if stated]
- Key Evidence: [the most important clinical sentence from the case]

Structured Evidence:
```

### Extraction statistics

| Item | Estimate |
|---|---|
| MultiCaRe cases to process | ~2,000–5,000 (filtered to MMCQSD-relevant conditions) |
| Tokens per call | ~800 (input + output) |
| Total tokens | ~2–4M |
| LLM | Llama 3.1 8B via Ollama (local, free, unlimited) |
| VRAM needed | ~6–8 GB (Q4_K_M quantization) |
| Time estimate | ~1–2 days of local processing |

---

## 4. LaBSE-Based Matching (replacing TF-IDF)

### Why LaBSE instead of TF-IDF

| Aspect | TF-IDF (current) | LaBSE (proposed) |
|---|---|---|
| Matching type | Surface token overlap | Semantic meaning |
| Cross-lingual | No — Hinglish and English share few tokens | Yes — encodes both into same vector space |
| Example: "chhati me dard" vs "chest pain" | Near-zero similarity | High similarity |
| Medical synonym handling | Misses synonyms entirely | Captures semantic equivalence |
| Model | Bag-of-words statistical | Pre-trained multilingual transformer (768-dim) |
| Already in project config | Yes (`sentence-transformers/LaBSE`) | Yes — same model |

### How the matching pipeline works

```
Step 1: Encode all extracted MultiCaRe evidence with LaBSE
        → evidence_embeddings (N × 768 matrix)

Step 2: Encode all MMCQSD Hinglish queries with LaBSE
        → query_embeddings (3015 × 768 matrix)

Step 3: Build FAISS index over evidence embeddings
        → IndexFlatIP (cosine similarity)

Step 4: For each MMCQSD query, retrieve top-k most similar evidence
        → candidate pairs with similarity scores

Step 5: Apply condition-label filter
        → MMCQSD condition label must be compatible with MultiCaRe case specialty

Step 6: Apply minimum similarity threshold
        → reject pairs below threshold (e.g., cosine < 0.3)

Step 7: Rank and select best match per query
        → final paired dataset
```

### Expected yield

| Scenario | Estimated pairs | Reasoning |
|---|---|---|
| Pessimistic | 300–500 | Only high-confidence semantic matches survive |
| Realistic | 800–1,500 | Good domain overlap + LaBSE cross-lingual alignment |
| Optimistic | 1,500–2,500 | Strong MultiCaRe coverage across all MMCQSD conditions |

Even the pessimistic scenario is **27x–45x** better than the current 11 pairs.

---

## 5. Complete Execution Plan

### Phase 0 — Setup and dependencies

| Step | Action | Output |
|---|---|---|
| 0.1 | Install `multiversity` library | MultiCaRe data access |
| 0.2 | Install Ollama + pull `llama3.1:8b` | Local LLM for evidence extraction |
| 0.3 | Install `sentence-transformers` (LaBSE) | Multilingual encoding |
| 0.4 | Verify FAISS installation | Retrieval index |

**Dependencies to add to `requirements.txt`:**
```
multiversity
ollama
```

### Phase 1 — Download and filter MultiCaRe

| Step | Action | Output |
|---|---|---|
| 1.1 | Initialize `MedicalDatasetCreator` from `multiversity` | Full MultiCaRe dataset loaded |
| 1.2 | Define condition-to-specialty mapping (MMCQSD conditions → MultiCaRe image types/labels) | Mapping table |
| 1.3 | Filter MultiCaRe cases to retain only specialties matching MMCQSD conditions | Filtered subset (~2,000–5,000 cases) |
| 1.4 | Export filtered cases with clinical text, image metadata, and condition labels | `data/raw/multicare/multicare_filtered.csv` |

**Condition-to-MultiCaRe filter mapping:**

| MMCQSD condition group | MultiCaRe filter criteria |
|---|---|
| skin rash, skin growth, skin irritation, skin dryness, dry scalp | image_type: `medical_photograph`, case_strings: skin/rash/dermatitis/lesion/eczema |
| swollen eye, eye redness, eye inflammation, itchy eyelid | image_type: `ophthalmic_imaging`, case_strings: eye/conjunctivitis/swelling/redness |
| mouth ulcers, lip swelling | case_strings: oral/mouth/ulcer/lip/stomatitis/mucosa |
| swollen tonsils, neck swelling | case_strings: tonsil/pharynx/neck/lymph/swelling/thyroid |
| foot swelling, knee swelling, hand lump | case_strings: extremity/joint/swelling/lump/edema/arthritis |
| edema | case_strings: edema/fluid/swelling/heart failure/renal |
| cyanosis | case_strings: cyanosis/hypoxia/cardiac/respiratory |

### Phase 2 — Extract structured evidence from MultiCaRe cases

| Step | Action | Output |
|---|---|---|
| 2.1 | Build the LLM extraction prompt template | Prompt for Llama 3.1 8B |
| 2.2 | Run batch extraction via Ollama for all filtered cases | Structured evidence per case |
| 2.3 | Parse and validate LLM outputs (reject malformed extractions) | Cleaned structured evidence |
| 2.4 | Save extracted evidence | `data/processed/multicare_evidence.csv` |

**Schema for extracted evidence:**

| Field | Description |
|---|---|
| `case_id` | MultiCaRe patient/case identifier |
| `article_id` | PMC article identifier |
| `condition_group` | Mapped MMCQSD condition group |
| `primary_finding` | Main diagnosis or condition |
| `location` | Body area affected |
| `symptoms` | Patient-reported symptoms |
| `clinical_signs` | Examination findings |
| `severity` | mild / moderate / severe / not specified |
| `key_evidence` | Most important clinical sentence from case |
| `full_case_text` | Original case narrative (for reference) |
| `extraction_quality` | pass / needs_review / failed |

### Phase 3 — Encode with LaBSE and build retrieval index

| Step | Action | Output |
|---|---|---|
| 3.1 | Encode all extracted evidence texts with LaBSE | `evidence_embeddings.npy` (N × 768) |
| 3.2 | Encode all MMCQSD Hinglish queries with LaBSE | `query_embeddings.npy` (3015 × 768) |
| 3.3 | Build FAISS IndexFlatIP over evidence embeddings | `multicare_evidence.index` |
| 3.4 | Validate embedding quality with sanity checks (nearest-neighbor spot checks) | Validation report |

### Phase 4 — Match MMCQSD queries to MultiCaRe evidence

| Step | Action | Output |
|---|---|---|
| 4.1 | For each MMCQSD query, retrieve top-k candidates from FAISS | Candidate pair list |
| 4.2 | Apply condition-label compatibility filter | Filtered candidates |
| 4.3 | Apply minimum similarity threshold | Thresholded candidates |
| 4.4 | Select best match per query | Matched pairs |
| 4.5 | Export paired dataset | `data/processed/mmcqsd_multicare_paired.csv` |
| 4.6 | Generate pairing summary statistics | Match rates per condition, score distributions |

**Schema for paired dataset:**

| Field | Description |
|---|---|
| `pair_id` | Unique pair identifier |
| `mmcqsd_sample_id` | MMCQSD sample identifier |
| `hinglish_query` | MMCQSD Hinglish query text |
| `english_summary` | MMCQSD English summary |
| `multicare_case_id` | Matched MultiCaRe case identifier |
| `evidence_text` | Extracted structured evidence |
| `key_evidence_sentence` | Most important clinical sentence |
| `full_case_text` | Full case narrative |
| `condition_group` | Condition category |
| `similarity_score` | LaBSE cosine similarity |
| `cmi_bucket` | Code-mixing index level (low/medium/high) |
| `match_quality` | auto_high / auto_medium / needs_review |

### Phase 5 — Human validation of a gold subset

| Step | Action | Output |
|---|---|---|
| 5.1 | Sort matched pairs by similarity score (descending) | Ranked pair list |
| 5.2 | Team reviews top 50–80 pairs: accept / reject / edit | Reviewed pairs |
| 5.3 | Annotate accepted pairs with finding labels, evidence spans | Annotated gold set |
| 5.4 | Export gold evaluation set | `data/processed/gold_evaluation_set.csv` |

**Gold set annotation fields:**

| Field | Description |
|---|---|
| `review_status` | accepted / rejected / needs_revision |
| `reviewer` | Team member name |
| `main_finding_label` | Primary finding (e.g., contact_dermatitis) |
| `secondary_finding_label` | Optional secondary finding |
| `body_region` | Body area |
| `evidence_span` | Key sentence supporting the match |
| `notes` | Optional reviewer notes |

### Phase 6 — Run prototype pipeline on new data

| Step | Action | Output |
|---|---|---|
| 6.1 | Build FAISS retrieval index over all MultiCaRe evidence | Evidence retrieval index |
| 6.2 | Run zero-shot generation for each matched query | Zero-shot outputs |
| 6.3 | Run evidence-grounded generation for each matched query | Grounded outputs |
| 6.4 | Evaluate: factual score, hallucination rate, retrieval metrics | Scored results |
| 6.5 | Run statistical tests (H1 at minimum) | Hypothesis test results |
| 6.6 | Compare results against the old Open-i prototype baseline | Improvement report |

### Phase 7 — Full evaluation and hypothesis testing

| Step | Action | Output |
|---|---|---|
| 7.1 | H1: Paired t-test / Wilcoxon — grounded vs zero-shot | p-value, effect size, CI |
| 7.2 | H2: Two-way ANOVA / Kruskal-Wallis — CMI level interaction | CMI robustness analysis |
| 7.3 | H3: Paired t-test / Wilcoxon — authoritative vs general evidence | Evidence quality comparison |
| 7.4 | Generate visualizations: robustness curves, comparison charts | Plots for paper |

---

## 6. Technical Architecture (Updated)

```
┌──────────────────────────────────────────────────────────────────┐
│                    HINGLISH PATIENT QUERY                        │
│  "Doctor, mere dono haathon pe laal daane ho gaye hain"         │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │   TEXT ENCODER (LaBSE)  │  ← Multilingual embedding
         │   Hinglish → Vector     │    (replaces TF-IDF)
         └────────────┬────────────┘
                      │
          ┌───────────▼───────────┐
          │   FAISS RETRIEVER     │  ← Cosine similarity
          │   + Condition Filter  │    + adaptive truncation
          └───────────┬───────────┘
                      │
     ┌────────────────▼────────────────┐
     │   RETRIEVED CLINICAL EVIDENCE   │
     │   • MultiCaRe case findings     │  ← Multi-specialty
     │   • Structured by LLM extract   │    (skin, eye, ENT, oral, etc.)
     │   • (Optional) case images      │
     └────────────────┬────────────────┘
                      │
         ┌────────────▼────────────┐
         │  GROUNDED GENERATOR     │  ← Evidence-first template
         │  Evidence → Hinglish    │
         │  Explanation            │
         └────────────┬────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────┐
│              GROUNDED HINGLISH EXPLANATION                       │
│  (factually consistent with retrieved clinical evidence)         │
└──────────────────────────────────────────────────────────────────┘
```

### Key architecture changes from prototype

| Component | Prototype (current) | Updated (MultiCaRe) |
|---|---|---|
| Evidence corpus | Open-i chest radiology reports | MultiCaRe multi-specialty clinical evidence |
| Evidence format | Structured radiology reports | LLM-extracted structured findings from case narratives |
| Text encoding | TF-IDF (bag-of-words) | LaBSE (multilingual dense embeddings, 768-dim) |
| Retrieval | TF-IDF similarity + concept bonus | FAISS cosine similarity + condition filter |
| Domain coverage | Chest only | Skin, eye, ENT, oral, musculoskeletal, cardiac, pulmonary |
| Paired data size | 11 weakly supervised pairs | Hundreds to thousands of matched pairs |
| Cross-lingual matching | None (same-language token overlap only) | LaBSE handles Hinglish ↔ English alignment natively |

---

## 7. File and Folder Structure (New/Changed)

```
data/
├── raw/
│   ├── openi/                         # existing (unchanged)
│   ├── mmcqsd/                        # existing (unchanged)
│   └── multicare/                     # NEW
│       └── multicare_filtered.csv     # filtered MultiCaRe cases
│
├── processed/
│   ├── openi_reports.csv              # existing (unchanged)
│   ├── mmcqsd_queries.csv             # existing (unchanged)
│   ├── multicare_evidence.csv         # NEW — LLM-extracted structured evidence
│   ├── mmcqsd_multicare_paired.csv    # NEW — matched query-evidence pairs
│   └── gold_evaluation_set.csv        # NEW — human-validated gold subset
│
├── embeddings/                         # NEW
│   ├── evidence_embeddings.npy        # LaBSE embeddings for evidence
│   └── query_embeddings.npy           # LaBSE embeddings for queries

src/
├── data/
│   ├── download_multicare.py          # NEW — download and filter MultiCaRe
│   └── extract_evidence.py            # NEW — LLM extraction pipeline
│
├── encoding/
│   └── text_encoder.py                # UPDATE — implement LaBSE encoding
│
├── retrieval/
│   ├── indexer.py                     # UPDATE — LaBSE-based FAISS index
│   └── retriever.py                   # UPDATE — cosine retrieval + condition filter
│
├── matching/                          # NEW
│   └── pair_builder.py                # LaBSE matching + condition filter + pairing
│
└── prototype/
    └── run_multicare_prototype.py     # NEW — end-to-end MultiCaRe pipeline
```

---

## 8. Implementation Priority Order

| Priority | Task | Phase | Estimated time |
|---:|---|---|---|
| 1 | Install dependencies (multiversity, ollama, LaBSE) | Phase 0 | 1 hour |
| 2 | Download and filter MultiCaRe to MMCQSD-relevant conditions | Phase 1 | 2–4 hours |
| 3 | Build and run LLM evidence extraction pipeline | Phase 2 | 1–2 days (batch runs) |
| 4 | Implement LaBSE encoding for evidence and queries | Phase 3 | 3–4 hours |
| 5 | Build FAISS index and matching pipeline | Phase 4 | 3–4 hours |
| 6 | Generate paired dataset and review statistics | Phase 4 | 2–3 hours |
| 7 | Human validation of gold subset (team effort) | Phase 5 | 1–2 days |
| 8 | Run prototype with grounded vs zero-shot comparison | Phase 6 | 4–6 hours |
| 9 | Statistical hypothesis testing | Phase 7 | 3–4 hours |

**Total estimated time: ~1.5–2 weeks**

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MultiCaRe cases too long/complex for clean extraction | Medium | Medium | Iterative prompt refinement; reject malformed outputs |
| LaBSE cross-lingual matching quality insufficient | Low | High | LaBSE is proven for Hindi-English; validate with spot checks |
| Too few MultiCaRe cases match specific MMCQSD conditions | Low | Medium | MultiCaRe has 93K+ cases across many specialties; flexible filtering |
| LLM extraction hallucinates findings not in case | Medium | High | Prompt explicitly constrains to stated findings; post-extraction validation |
| Ollama inference too slow on available hardware | Low | Low | Can supplement with free APIs (Gemini, Groq) |
| Mentor questions scope change | Medium | Medium | Frame as "addressing a documented limitation"; stronger contribution |
| Case report evidence is less structured than radiology reports | Medium | Low | LLM extraction normalizes format; structured schema enforced |

---

## 10. Updated Hypothesis Testing Plan

The three hypotheses remain unchanged. Only the underlying data improves.

| Hypothesis | Test | Current data (Open-i) | New data (MultiCaRe) |
|---|---|---|---|
| H1: Grounded > zero-shot on factuality | Paired t-test / Wilcoxon | 11 pairs (weak) | 300–1,500+ pairs (strong) |
| H2: CMI level does not degrade grounded performance | Two-way ANOVA / Kruskal-Wallis | Insufficient per-CMI-level samples | Sufficient samples across low/medium/high CMI |
| H3: Authoritative evidence > general evidence | Paired t-test / Wilcoxon | Not testable (too few pairs) | Testable by comparing evidence quality levels |

### How H3 adapts

With MultiCaRe, H3 can compare:
- **Authoritative evidence:** MultiCaRe cases with clear diagnosis and structured findings
- **General evidence:** Lower-quality or loosely matched evidence
- This comparison was previously untestable due to insufficient data

---

## 11. Success Criteria

The MultiCaRe solution is considered successful if:

| Criterion | Target |
|---|---|
| Matched pairs produced | >= 300 (minimum), 800+ (target) |
| Condition coverage | >= 12 out of 18 MMCQSD conditions represented |
| Gold validated subset | >= 50 human-reviewed pairs |
| H1 testable | Statistically significant with adequate sample size |
| Factual score improvement | Grounded > zero-shot by measurable margin |
| Hallucination reduction | Grounded < zero-shot by measurable margin |
| Retrieval quality (LaBSE) | Top-k hit rate significantly above old TF-IDF baseline |

---

## 12. What Stays the Same

| Component | Status |
|---|---|
| MMCQSD as the Hinglish query source | Unchanged — same 3,015 queries |
| LaBSE as text encoder | Already in project config — now actually implemented |
| FAISS as retrieval engine | Unchanged — same technology, better embeddings |
| Hypothesis framework (H1, H2, H3) | Unchanged — same hypotheses, better data |
| Evaluation metrics (factual consistency, hallucination rate) | Unchanged |
| Statistical testing plan | Unchanged |
| Streamlit demo architecture | Unchanged — just wired to new data |
| Project structure and module layout | Minimal changes — new files added, existing stubs implemented |

---

## 13. Mentor-Facing Summary

### What changed and why

The project encountered a documented limitation: Open-i (chest-only radiology) and MMCQSD (multi-specialty Hinglish queries) have a fundamental domain mismatch, producing only 11 usable pairs.

To overcome this, the evidence corpus is expanded from chest-only radiology to **multi-specialty clinical evidence** using MultiCaRe, a freely available dataset of 93,000+ clinical cases from PubMed Central that covers the same medical specialties as MMCQSD (dermatology, ophthalmology, ENT, oral, musculoskeletal, etc.).

Additionally, the retrieval approach is upgraded from TF-IDF (surface token matching) to **LaBSE** (multilingual dense embeddings), enabling true cross-lingual semantic matching between Hinglish patient queries and English clinical evidence.

### What this achieves

- Increases paired dataset size from 11 to potentially hundreds–thousands
- Enables statistically meaningful hypothesis testing
- Makes the system multi-specialty (stronger research contribution)
- Addresses Limitation #2 from the original proposal ("Domain & Modality Constraint")
- Uses the multilingual encoding approach (LaBSE) that was already part of the project design

### What stays unchanged

The core research question, hypotheses, statistical testing plan, and system architecture remain the same. The change is in the evidence data source and the encoding quality — both of which strengthen the project.
