# Final Development Strategy Plan For Weak Dataset Pairing

## Purpose
This plan defines how the project should move from the current weakly supervised `Open-i + MMCQSD` prototype toward a stronger final development setup.

The core problem is:

> `Open-i` and `MMCQSD` are both useful, but they are not naturally paired datasets.

So the final system must be designed in a way that:
- uses both datasets effectively
- does not force an unrealistic pairing
- creates a more reliable bridge between Hinglish queries and radiology evidence

## Main strategy

### Final design principle
Do **not** treat `Open-i` and `MMCQSD` as if they were a ready-made training pair.

Instead, assign them different roles:

| Dataset | Final role |
|---|---|
| `Open-i` | radiology evidence/report corpus |
| `MMCQSD` | Hinglish query style and user-query language resource |

Then build a **third layer** that bridges them:

| New layer needed | Role |
|---|---|
| curated paired bridge set | connects Hinglish queries to real radiology evidence |

## What must be implemented for final development

## 1. Build a human-validated gold pairing subset

### Why this is necessary
This is the single most important step for overcoming weak dataset pairing.

The current prototype uses weak supervision.  
For final development, the project needs a **small but trustworthy gold subset**.

### What to implement
Create a reviewed dataset containing:

| Field | Description |
|---|---|
| `pair_id` | unique pair identifier |
| `openi_report_id` | linked real `Open-i` report |
| `report_text` | evidence report text |
| `hinglish_query` | Hinglish question aligned to the report |
| `english_reference_answer` | short medically grounded answer |
| `main_finding_label` | primary report finding |
| `secondary_finding_label` | optional secondary finding |
| `negation_flag` | whether the report is mainly negative / no-acute |
| `side_label` | right / left / bilateral / none |
| `severity_label` | mild / moderate / severe / none |
| `evidence_span` | key report sentence or phrase supporting the answer |
| `review_status` | accepted / rejected / needs revision |

### Recommended size

| Stage | Recommended size |
|---|---:|
| first validated set | `30-50` pairs |
| stronger evaluation set | `100-150` pairs |
| later extended set | `200+` pairs |

### Why this helps
- removes the strongest weakness in the current pipeline
- gives a fair evaluation benchmark
- supports better retrieval tuning
- supports better grounded generation
- becomes defendable during mentor review and paper writing

## 2. Create a structured pair-review workflow

### Why this is needed
The bridge set should not be created ad hoc.

It should follow a repeatable review workflow so the team can keep improving it over time.

### What to implement

| Step | Action |
|---|---|
| 1 | auto-suggest best `Open-i` report for candidate Hinglish query |
| 2 | show candidate pair to reviewer |
| 3 | accept / reject / edit the pair |
| 4 | save corrected query and validated answer |
| 5 | store finding labels and evidence span |

### Output files needed

| File | Purpose |
|---|---|
| `candidate_pairs.csv` | machine-suggested candidates |
| `review_sheet.csv` | human review and correction file |
| `gold_pairs.csv` | accepted final validated set |

## 3. Improve retrieval to support better pairing

### Why this matters
Even a validated subset will need strong retrieval for final development.

The current retrieval is still prototype-level.

### What to implement

| Component | Improvement |
|---|---|
| sparse retrieval | keep TF-IDF/BM25 style retrieval |
| dense retrieval | add multilingual / semantic embedding retrieval |
| hybrid retrieval | combine sparse + dense scores |
| reranking | add second-stage reranking on top candidates |
| clinical constraints | reward side/severity/finding agreement |
| negation handling | penalize contradiction such as positive query vs negative report |

### Final target
Move from:
- one-step approximate matching

To:
- hybrid retrieval
- concept-aware reranking
- contradiction-aware filtering

## 4. Build a medical finding extraction layer

### Why this is needed
Weak pairing becomes much easier to control if both query and report are mapped into the same structured clinical space.

### What to implement
For both report and query, extract:

| Field | Examples |
|---|---|
| finding | cardiomegaly, pleural effusion, opacity |
| side | right, left, bilateral |
| severity | mild, moderate, severe |
| negation | present / absent |
| body-system | chest / cardio-respiratory |

### Use in final system
- help candidate-pair generation
- help reranking
- help error analysis
- help evaluation of factual correctness

## 5. Separate training/evaluation roles clearly

### Why this matters
In final development, datasets should not all be used in the same way.

### Recommended role split

| Dataset | Final use |
|---|---|
| `Open-i` | evidence corpus and report-side grounding |
| `MMCQSD` | Hinglish query style resource |
| validated gold subset | true training/evaluation bridge |
| `PubMedQA` | supplementary text-QA comparison |
| `MMedBench` | supplementary benchmark comparison |

### Key rule
The **validated bridge subset** should become the main benchmark for the final `Open-i + Hinglish` system.

## 5A. How `PubMedQA` and `MMedBench` will be used

### Can they help weak pairing?
Yes, but indirectly.

They can improve:
- grounding behavior
- medical reasoning robustness
- evaluation coverage

They cannot directly create natural `Open-i <-> MMCQSD` pairs.

### Recommended practical usage

| Dataset | Use in final development | What it helps | What it does not solve |
|---|---|---|---|
| `PubMedQA` | auxiliary training/evaluation for evidence-grounded text QA | improves reference-following, claim support checks, and answer faithfulness | does not provide Hinglish-radiology pair supervision |
| `MMedBench` | supplementary robustness benchmark | improves structured medical QA robustness and stress-tests model behavior | does not provide report-level paired Hinglish supervision |

### Implementation rule for the team

| Rule | Decision |
|---|---|
| Core final benchmark | validated `Open-i + Hinglish` bridge subset |
| Auxiliary datasets | `PubMedQA` and `MMedBench` used for support, stress tests, and supplementary reporting |
| Final claim basis | should come primarily from the validated core benchmark, not auxiliary-only gains |

## 6. Improve grounded answer generation

### Why this is necessary
The current prototype shows that retrieval alone is not enough.
Hallucination remained high even when factuality improved slightly.

### What to implement

| Improvement | Purpose |
|---|---|
| evidence-first answer template | force answer to begin from retrieved evidence |
| structured response format | finding -> evidence -> explanation -> caution |
| sentence-level evidence insertion | use exact retrieved evidence lines |
| contradiction checking | block outputs that conflict with evidence |
| later fine-tuning | improve medical + Hinglish grounded answering style |

### Suggested final answer format
1. likely finding
2. evidence from report
3. short Hinglish explanation
4. caution if uncertainty remains

## 7. Build a stronger evaluation framework

### Why this is necessary
For final development, the project needs better evaluation than a weakly matched subset alone.

### What to implement

| Evaluation area | Metric or check |
|---|---|
| retrieval | top-1, top-k, evidence rank |
| factuality | support score against evidence/reference |
| hallucination | unsupported claim detection |
| negation correctness | positive vs negative finding correctness |
| side correctness | right/left/bilateral accuracy |
| severity correctness | mild/moderate/severe correctness |
| Hinglish quality | readability / usefulness review |

### Recommended testing style
Since the project already prefers this by default, testing should remain **parametrized**:

| Test type | Examples |
|---|---|
| retrieval tests | different findings, side labels, negative reports |
| generation tests | evidence-present vs evidence-absent |
| concept tests | negation, severity, side, overlap |
| evaluation tests | known expected scoring outcomes |

## 8. Use phased implementation instead of one-step finalization

### Recommended development phases

| Phase | Goal | Deliverable |
|---|---|---|
| Phase 1 | clean candidate generation | better real candidate pair file |
| Phase 2 | human validation | first gold subset |
| Phase 3 | retrieval upgrade | hybrid retrieval + reranking |
| Phase 4 | generation upgrade | evidence-first grounded answers |
| Phase 5 | final evaluation | validated benchmark results |

## Priority order for actual implementation

### Highest priority

| Priority | Task |
|---:|---|
| 1 | build reviewable candidate pair file |
| 2 | manually validate first `30-50` gold pairs |
| 3 | use validated pairs for retrieval and generation evaluation |
| 4 | improve retrieval with hybrid + reranking |
| 5 | improve grounded answer generation |

### Lower priority but still useful

| Priority | Task |
|---:|---|
| 6 | expand gold set size |
| 7 | add later multimodal image support |
| 8 | include supplementary benchmark reporting using `PubMedQA` and `MMedBench` |

## Final recommendation

### Most important conclusion
The final development should **not** try to force `MMCQSD` and `Open-i` into a direct large-scale automatic pairing.

Instead, the best long-term solution is:

1. keep `Open-i` as the real radiology evidence base  
2. keep `MMCQSD` as the Hinglish query-style resource  
3. build a human-validated bridge subset between them  
4. train and evaluate the final system around that validated bridge  

## Mentor-facing takeaway

- the current weak pairing is a real limitation, but it is manageable
- the proper solution is not more loose matching
- the proper solution is a **validated bridge dataset + stronger retrieval + evidence-first generation**
- this gives the project a realistic path from prototype to final development
