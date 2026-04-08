# Prototype vs Final Development: Side-by-Side Comparison

## Purpose

This document compares **what the current real-data prototype does** with **what the main final development is supposed to do**. It serves as a single reference for: what’s done, what’s left, expectations, techniques, and next steps.

---

## 1. What the Prototype (Real HMG Dataset) Does

### 1.1 Scope and role

| Aspect | Prototype (current) |
|--------|----------------------|
| **Data** | Real **Open-i** (6,687 reports) + real **MMCQSD** (Hinglish queries) |
| **Pairing** | Weakly supervised: 11 high-confidence cardio-respiratory matched pairs after filtering |
| **Goal** | Prove that grounded generation beats zero-shot on factuality and hallucination on real data |
| **Status** | End-to-end working pipeline; results are prototype-level, not final benchmark |

### 1.2 Pipeline flow

```
Load real Open-i reports → Load real MMCQSD queries → Filter cardio-respiratory cases
→ Align query to best Open-i report → Build index → Run zero-shot and grounded outputs
→ Evaluate (retrieval, factual score, hallucination, paired significance)
```

### 1.3 What the prototype actually does

| Component | Implementation |
|-----------|----------------|
| **Evidence corpus** | Open-i reports indexed (full 6,687) |
| **Query side** | MMCQSD Hinglish queries, filtered to cardio-respiratory |
| **Matching** | Multi-stage: basic alignment → cardio-respiratory filter → shared medical concepts → contradiction-aware filter |
| **Retrieval** | Prototype-level (TF-IDF / FAISS style) |
| **Generation** | Zero-shot vs grounded (simple, generic prompts) |
| **Evaluation** | Retrieval (top-1, top-k), factual score, hallucination rate, paired t-test |

### 1.4 Prototype results (summary)

| Metric | Zero-shot | Grounded | Interpretation |
|--------|-----------|----------|----------------|
| Factual score | 0.095 | 0.412 | Strong gain from grounding |
| Hallucination rate | 1.000 | 0.000 | Strong drop in this run |
| Top-1 hit | — | 0.000 | Correct report not rank-1 |
| Top-k hit | — | 0.364 | Evidence sometimes in top-k |
| p-value (grounded vs zero-shot) | — | ~0.000000 | Statistically strong difference |

**Hypothesis H1:** Supported at **prototype level** — grounded is more factual and less hallucinatory than zero-shot. Not yet a final claim due to small n (11) and weak pairing.

### 1.5 Prototype limitations

- **Small subset:** Only 11 cleaned real pairs.
- **Weak supervision:** Open-i and MMCQSD are not naturally paired; alignment is artificial.
- **Retrieval:** Still prototype-level; top-1 hit is 0.
- **Generation:** Simple, generic; no evidence-first template or structured format.
- **Evaluation:** Lightweight; no negation/side/severity correctness, no gold benchmark.

---

## 2. What the Main Final Development Is Supposed to Do

### 2.1 High-level goal

Build a **production-ready evidence-first RAG system** for Hinglish clinical queries that:

1. Uses **Open-i** as the radiology evidence/report corpus.
2. Uses **MMCQSD** as the Hinglish query-style resource (not as a ready-made paired set).
3. Bridges them via a **human-validated gold subset** (no forced large-scale automatic pairing).
4. Delivers **grounded, factual, low-hallucination** Hinglish explanations for clinical decision support.

### 2.2 Design principle (from weak-pairing strategy)

- **Do not** treat Open-i and MMCQSD as a ready-made training pair.
- **Assign roles:** Open-i = evidence; MMCQSD = Hinglish query resource.
- **Add a third layer:** Curated paired bridge set connecting Hinglish queries to real radiology evidence.

### 2.3 Final system components (target)

| Component | Final development target |
|-----------|---------------------------|
| **Evidence** | Open-i as main report corpus; optional multimodal (e.g. BioMedCLIP) later |
| **Queries** | Hinglish from MMCQSD-style + validated bridge set |
| **Pairing** | Human-validated gold subset (30–50 first, then 100–150+); no weak-only pairing |
| **Retrieval** | Hybrid (sparse + dense), reranking, concept-aware, contradiction-aware |
| **Generation** | Evidence-first template, structured format (finding → evidence → explanation → caution), contradiction checking |
| **Evaluation** | Retrieval (top-1, top-k, rank), factuality, hallucination, negation/side/severity correctness, Hinglish quality |
| **Benchmark** | Validated bridge set as **core** benchmark; PubMedQA/MMedBench as **auxiliary** |

---

## 3. Side-by-Side Comparison

### 3.1 What’s done vs what’s left

| Area | Prototype (done) | Final development (left to do) |
|------|------------------|---------------------------------|
| **Data** | Real Open-i + MMCQSD loaded and used | Same sources; add **validated gold bridge set** (30–50 → 100–150+ pairs) |
| **Pairing** | Weak alignment, 11 cleaned pairs | **Human-validated** pairs; review workflow (candidate → review → gold) |
| **Retrieval** | Single-step, prototype-level | **Hybrid** (sparse + dense), **reranking**, clinical/negation-aware |
| **Generation** | Simple zero-shot vs grounded | **Evidence-first** template, **structured** response, contradiction checking; later fine-tuning (e.g. QLoRA/DPO) |
| **Evaluation** | Basic factual + hallucination + paired test | Full suite: retrieval, factuality, hallucination, **negation/side/severity**, Hinglish quality; parametrized tests |
| **Medical structure** | Filtering by cardio-respiratory + concepts | **Finding extraction layer** (finding, side, severity, negation, body-system) for both query and report |
| **Benchmark** | 11 pairs, weak supervision | **Gold subset** as main benchmark; PubMedQA/MMedBench auxiliary only |
| **Training/eval split** | Not clearly separated | Clear roles: bridge = core benchmark; others = support/stress-test |

### 3.2 Expectations

| Dimension | Prototype expectation | Final development expectation |
|-----------|------------------------|-------------------------------|
| **Hypothesis H1** | Grounded > zero-shot on factuality/hallucination (prototype-level support) | Same hypothesis; **reliable** support on **validated** benchmark with stronger retrieval and generation |
| **Sample size** | Small (11) acceptable for prototype | 30–50 minimum for first validated set; 100–150+ for stronger evaluation |
| **Pair quality** | “Good enough” for signal | **Defensible** for mentor review and paper; human-validated |
| **Retrieval** | Some top-k success acceptable | Top-1 and top-k both matter; hybrid + reranking expected to improve |
| **Hallucination** | Strong drop in prototype run | Sustained low hallucination with evidence-first generation and contradiction checks |
| **Claim** | “Prototype-level validated signal” | “Final system results on validated Open-i + Hinglish benchmark” |

### 3.3 Techniques: prototype vs final

| Technique | Prototype | Final development |
|-----------|-----------|--------------------|
| **Encoding** | Text-only retrieval (TF-IDF / FAISS-style) | **LaBSE** (or similar) for multilingual; keep sparse, add **dense** retrieval |
| **Retrieval** | Single-stage | **Hybrid** (sparse + dense) + **reranking**; clinical/negation constraints |
| **Generation** | Generic prompts, no template | **Evidence-first** template; **structured** format; **contradiction checking** |
| **Fine-tuning** | Not in prototype | **QLoRA**, **DPO** (anti-hallucination) as in README |
| **Multimodal** | Text-first only | Optional later: **BioMedCLIP** for images |
| **Evaluation** | Factual score, hallucination rate, paired test | Add **negation**, **side**, **severity** correctness; **MMFCM**-style where applicable; parametrized tests |
| **Medical structure** | Filtering only | **Finding extraction** (query + report) for pairing, reranking, and evaluation |

### 3.4 Datasets: roles

| Dataset | Prototype role | Final development role |
|---------|----------------|-------------------------|
| **Open-i** | Evidence corpus (6,687 reports) | Same; main evidence/report corpus |
| **MMCQSD** | Hinglish query source; weakly matched to Open-i | Hinglish query **style** resource; **validated bridge** uses both but with human curation |
| **Validated bridge** | Does not exist | **Core** training/evaluation benchmark |
| **PubMedQA** | Compared in earlier real-dataset comparison | **Auxiliary** evidence-grounded QA; does not replace bridge |
| **MMedBench** | Compared in earlier real-dataset comparison | **Auxiliary** robustness benchmark; does not replace bridge |

---

## 4. Next Steps (Prioritized)

### 4.1 Highest priority

| # | Step | Why |
|---|------|-----|
| 1 | Build **reviewable candidate pair file** (Open-i + MMCQSD suggestions) | Enables human validation instead of weak-only matching |
| 2 | **Manually validate** first 30–50 gold pairs (review workflow: accept/reject/edit, store finding labels, evidence span) | Removes main weakness; defensible benchmark |
| 3 | Use **validated pairs** for retrieval and generation evaluation | Fair comparison and tuning |
| 4 | **Improve retrieval:** hybrid (sparse + dense) + reranking, concept-aware, contradiction-aware | Better top-1/top-k and evidence quality |
| 5 | **Improve grounded generation:** evidence-first template, structured format, contradiction checking | Sustain factuality and low hallucination |

### 4.2 Phased implementation (from weak-pairing plan)

| Phase | Goal | Deliverable |
|-------|------|-------------|
| **Phase 1** | Clean candidate generation | Better real candidate pair file |
| **Phase 2** | Human validation | First gold subset (30–50) |
| **Phase 3** | Retrieval upgrade | Hybrid retrieval + reranking |
| **Phase 4** | Generation upgrade | Evidence-first grounded answers |
| **Phase 5** | Final evaluation | Validated benchmark results |

### 4.3 Lower priority (later)

- Expand gold set (100–150+).
- Add multimodal (image) support if needed.
- Supplementary reporting on PubMedQA and MMedBench.
- Full fine-tuning (QLoRA, DPO) for medical + Hinglish grounded style.

---

## 5. One-page snapshot

| | **Prototype (real HMG)** | **Final development** |
|---|--------------------------|------------------------|
| **What it does** | Runs on real Open-i + MMCQSD with 11 weakly matched pairs; compares zero-shot vs grounded; supports H1 at prototype level | Same hypothesis; validated gold bridge set; stronger retrieval and generation; full evaluation suite; defensible benchmark |
| **Done** | Real data pipeline; filtering; prototype retrieval/generation; basic evaluation; H1 signal | — |
| **Left** | — | Gold subset + review workflow; hybrid retrieval + reranking; evidence-first generation; finding extraction; full metrics; parametrized tests |
| **Expectations** | Prototype-level signal; small n; weak pairing acceptable | Reliable, validated results; human-validated pairs; stronger claims |
| **Techniques** | TF-IDF/FAISS-style retrieval; simple grounded prompt | LaBSE + dense retrieval; hybrid + reranking; evidence-first template; QLoRA/DPO later |
| **Next steps** | — | 1) Candidate pairs → 2) Validate 30–50 → 3) Evaluate on gold → 4) Retrieval upgrade → 5) Generation upgrade |

---

## 6. References (in this repo)

- `plans/review-2/revamped-real-prototype-status-report.md` — What the current real prototype does and uses.
- `plans/review-2/prototype-results-and-hypothesis-report.md` — Prototype results and H1 interpretation.
- `plans/review-2/weak-pairing-strategy-plan.md` — Final development strategy (gold set, retrieval, generation, evaluation).
- `plans/review-2/real-dataset-comparison-detailed-report.md` — Why Open-i + MMCQSD was chosen.
- `README.md` — Project goal, architecture, techniques, hypotheses.
