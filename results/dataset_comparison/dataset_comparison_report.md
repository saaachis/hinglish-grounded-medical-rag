# Multi-Dataset Real Subset Comparison Report

## Purpose
Compare real subsets from the downloaded datasets under one shared retrieval pipeline to estimate which dataset family is most useful for the current project objective.

## Experiment Setup
- Real samples per dataset profile: **150**
- Same retrieval/indexing and generation pipeline across all profiles
- Cross-dataset distractors added per other profile: **30**
- Same generic evaluator and paired statistical test
- `PubMedQA` comparison uses the expert-labeled subset
- `MMedBench` comparison uses the English subset for compatibility with the current text pipeline

## Dataset Profiles Compared
- Open-i real subset
- MMCQSD real subset
- PubMedQA real labeled subset
- MMedBench real English subset

## Results Table

| Profile | Top-1 Hit | Top-k Hit | Zero Factual | Grounded Factual | Factual Gain | Zero Hall. | Grounded Hall. | Hallucination Drop | p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MMCQSD (real subset) | 0.693 | 0.853 | 0.091 | 0.250 | 0.159 | 1.000 | 0.980 | 0.020 | 0.0000 |
| Open-i (real subset) | 0.007 | 0.027 | 0.228 | 0.333 | 0.105 | 0.993 | 0.000 | 0.993 | 0.0000 |
| MMedBench (real English subset) | 0.807 | 0.967 | 0.294 | 0.323 | 0.029 | 0.940 | 0.933 | 0.007 | 0.0000 |
| PubMedQA (real labeled subset) | 0.980 | 0.980 | 0.601 | 0.560 | -0.042 | 0.047 | 0.000 | 0.047 | 0.0000 |

## Ranking Summary (Most Useful First)
- `MMCQSD (real subset)`: factual gain=0.159, hallucination drop=0.020, top-k hit=0.853, p=0.0000
- `Open-i (real subset)`: factual gain=0.105, hallucination drop=0.993, top-k hit=0.027, p=0.0000
- `MMedBench (real English subset)`: factual gain=0.029, hallucination drop=0.007, top-k hit=0.967, p=0.0000
- `PubMedQA (real labeled subset)`: factual gain=-0.042, hallucination drop=0.047, top-k hit=0.980, p=0.0000

## Recommendation
- Best profile under this real-subset comparison: **MMCQSD (real subset)**
- This is still a subset-based prototype comparison, not the final full-scale benchmark.
- For the main Hinglish grounded prototype, the next focused build should still prioritize `Open-i + MMCQSD`.

## Next Step
- Use the comparison outcome to build the main real-data prototype around `Open-i + MMCQSD`.
