# Real Dataset Comparison Report

## Purpose
This report summarizes the first comparison run performed on **real dataset subsets** after replacing the earlier synthetic proxy comparison.

The main aim was to identify which real dataset family is most useful for the project before moving into the main real-data Hinglish grounded prototype.

## Quick status

| Item | Status |
|---|---|
| Comparison type | real subset comparison |
| Datasets compared | `Open-i`, `MMCQSD`, `PubMedQA`, `MMedBench` |
| Samples per profile | `150` |
| Retrieval setup | TF-IDF + FAISS + reranking |
| Generation comparison | zero-shot vs grounded |
| Statistical test | Wilcoxon signed-rank |

### Important note

| Item | Meaning |
|---|---|
| `PubMedQA` subset | real labeled subset only |
| `MMedBench` subset | English subset only |
| Result type | directional prototype comparison, not full final benchmark |

## Result table

| Dataset profile | Top-1 hit | Top-k hit | Zero factual | Grounded factual | Factual gain | Zero hallucination | Grounded hallucination | Hallucination drop | p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MMCQSD | 0.693 | 0.853 | 0.091 | 0.250 | 0.159 | 1.000 | 0.980 | 0.020 | 0.0000 |
| Open-i | 0.007 | 0.027 | 0.228 | 0.333 | 0.105 | 0.993 | 0.000 | 0.993 | 0.0000 |
| MMedBench | 0.807 | 0.967 | 0.294 | 0.323 | 0.029 | 0.940 | 0.933 | 0.007 | 0.0000 |
| PubMedQA | 0.980 | 0.980 | 0.601 | 0.560 | -0.042 | 0.047 | 0.000 | 0.047 | 0.0000 |

## Ranking summary

| Rank | Dataset | Main reason |
|---:|---|---|
| 1 | `MMCQSD` | best factual gain under current setup |
| 2 | `Open-i` | strongest evidence-side value for the actual project |
| 3 | `MMedBench` | good retrieval but weaker alignment with project goal |
| 4 | `PubMedQA` | very strong retrieval, but weak fit for this task setup |

## Metric key

| Metric | Meaning |
|---|---|
| `Top-1 hit` | correct expected item retrieved first |
| `Top-k hit` | correct expected item appeared somewhere in retrieved set |
| `Factual gain` | grounded output improved factual score over zero-shot |
| `Hallucination drop` | grounded output reduced unsupported content |

## Dataset-wise interpretation

| Dataset | What worked well | Main limitation | Practical role |
|---|---|---|---|
| `MMCQSD` | strongest factual gain, strong query-style fit, good retrieval | hallucination still high, generator still weak | best **query-side** dataset |
| `Open-i` | strongest evidence/report relevance, large hallucination drop | generic prompts hurt standalone retrieval comparison | best **evidence/report-side** dataset |
| `MMedBench` | very strong retrieval | only small grounded gain, less aligned with Hinglish radiology goal | supplementary benchmark |
| `PubMedQA` | very strong retrieval | grounded output did not improve factuality | supplementary text-QA benchmark |

## What the comparison tells us

| Question | Result |
|---|---|
| Which dataset is strongest for the current query setting? | `MMCQSD` |
| Which dataset is strongest for the evidence/report side? | `Open-i` |
| Should all four datasets be treated equally in the main prototype? | No |
| Best core direction for the main prototype? | `Open-i + MMCQSD` |

## Why `Open-i + MMCQSD` was chosen

| Dataset | Best role in the project |
|---|---|
| `Open-i` | real radiology-style evidence/report corpus |
| `MMCQSD` | real Hinglish / code-mixed query side |
| `PubMedQA` | supporting text-only benchmark |
| `MMedBench` | supporting benchmark, not core pair |

## Assumptions and limitations

| Item | Current limitation |
|---|---|
| Subset size | this was a subset-level comparison, not full-scale dataset benchmarking |
| Generator | current generator is still lightweight and simple |
| Open-i prompts | many are generic, so Open-i is underrepresented on the query side |
| MMedBench | only English subset used |
| Cross-dataset comparability | `PubMedQA` and `MMedBench` are not directly equivalent to a radiology-style evidence task |

## Mentor-facing conclusion

- this real comparison is a stronger foundation than the earlier synthetic comparison
- `MMCQSD` is the best real dataset for the query side
- `Open-i` remains the most important real evidence/report dataset
- the main prototype should continue with `Open-i + MMCQSD`
- `PubMedQA` and `MMedBench` should remain supplementary comparison datasets

## Recommended next step

`Open-i` should be used as the retrieval/report corpus and `MMCQSD` should be used as the real Hinglish query source in the main real-data prototype pipeline.
