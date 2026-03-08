# Revamped Real Prototype Status Report

## Purpose
This report summarizes the new **real-data prototype pipeline** built after replacing the earlier synthetic setup.

The main objective of this phase was to make the prototype work on **real downloaded datasets**, while keeping it aligned with the project goal:

- use `Open-i` as the evidence/report corpus
- use `MMCQSD` as the Hinglish query side
- compare grounded generation against zero-shot generation

## Quick status

| Item | Current status |
|---|---|
| Prototype mode | running on **real data** |
| Evidence side | `Open-i` |
| Hinglish query side | `MMCQSD` |
| Pairing quality | cleaned, high-confidence matched subset |
| Pipeline status | end-to-end working |

### Important reality check

| Item | Meaning |
|---|---|
| `Open-i` and `MMCQSD` are not naturally paired | current pairing is still weakly supervised |
| Current subset is small | useful for prototype progress, not yet final-quality benchmark |

## What was changed in this revamped prototype

| Change area | What was done |
|---|---|
| Main data path | moved away from the synthetic route for the main prototype |
| Data source | switched to real downloaded `Open-i` and `MMCQSD` |
| Preparation | added real filtering, alignment, and contradiction-aware matching |
| Runner | added an end-to-end real prototype runner |

### Current workflow
`load real Open-i reports -> load real MMCQSD queries -> filter cardio-respiratory cases -> align query to best Open-i report -> build index -> run zero-shot and grounded outputs -> evaluate`

## Current real prototype data used

| Data component | Current value | What it means |
|---|---:|---|
| Total `Open-i` reports indexed | `6687` | full real evidence corpus is being used |
| Retained `MMCQSD` candidates | `11` | only stricter cardio-respiratory matches survived |
| Final matched subset used | `11` | smaller, cleaner real subset |

### Interpretation
- the subset is now smaller but more defensible
- weak and irrelevant pairs were reduced
- this is better for mentor presentation than a larger noisy subset

## Files produced in the revamped prototype

| Type | Files |
|---|---|
| Real preparation files | `data/processed/real_h1/openi_real_corpus.csv`, `data/processed/real_h1/openi_mmcqsd_real_queries.csv`, `data/processed/real_h1/openi_mmcqsd_real_prep_summary.md` |
| Prototype output files | `results/h1_real_openi_mmcqsd/h1_outputs.csv`, `results/h1_real_openi_mmcqsd/h1_scored.csv`, `results/h1_real_openi_mmcqsd/h1_summary.md` |
| Main scripts | `src/prototype/prepare_openi_mmcqsd_real.py`, `src/prototype/run_openi_mmcqsd_real_prototype.py` |

## How the current matching was improved

| Improvement stage | What changed |
|---|---|
| Stage 1 | basic real alignment between `MMCQSD` queries and `Open-i` reports |
| Stage 2 | stronger cardio-respiratory filtering on `MMCQSD` |
| Stage 3 | shared medical concept filtering between query and report |
| Stage 4 | contradiction-aware filtering to reduce clearly wrong matches |

## Current strengths of the revamped prototype

| Strength | Why it matters |
|---|---|
| Real-data based | the main prototype no longer depends on synthetic samples |
| Better role split | `Open-i` is now clearly the evidence side and `MMCQSD` the Hinglish query side |
| Reusable preparation logic | future work can improve filtering and matching without rebuilding the whole pipeline |

## Current limitations

| Limitation | Why it matters |
|---|---|
| Small subset | only `11` high-confidence real pairs remain |
| Weak supervision | `Open-i` and `MMCQSD` still have to be aligned artificially |
| Remaining mismatch | some retained pairs are still only partially aligned clinically |

## Mentor-facing interpretation

| What was achieved | What should be understood |
|---|---|
| main prototype moved to real data | this is still a working prototype, not the final strong paired system |
| dedicated `Open-i + MMCQSD` preparation pipeline built | the biggest success is the realistic pipeline foundation, not high final performance yet |
| weaker matches reduced using stricter filtering | future improvements can now build on a real-data base |

## Recommended next step

| Next step | Why |
|---|---|
| manually review the current `11` matched real pairs | improve trust in the retained subset |
| keep only the strongest real matches | reduce weak supervision noise further |
| expand into a small validated gold subset | make the final development and evaluation more reliable |
