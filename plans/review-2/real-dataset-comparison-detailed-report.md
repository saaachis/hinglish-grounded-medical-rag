# Real Dataset Comparison Detailed Report

## Purpose
This report summarizes the first comparison run performed on **real dataset subsets**, replacing the earlier synthetic proxy comparison.

The aim of this experiment was to check which real dataset family is most useful for the current prototype setup before moving into the main real-data Hinglish grounded prototype.

## What changed from the earlier synthetic comparison
- The comparison is now based on **real downloaded datasets**
- Small, manageable **real subsets** were created from each dataset
- The same retrieval and generation pipeline was used across all dataset profiles
- Cross-dataset distractors were added into the retrieval index to make retrieval more realistic
- The result is still a **prototype-level subset comparison**, but it is more valid than the earlier synthetic setup

## Datasets included in this run
- `Open-i`
- `MMCQSD`
- `PubMedQA`
- `MMedBench`

## Real subset setup used
- Samples per dataset profile: `150`
- Cross-dataset distractors added per other dataset: `30`
- Retrieval pipeline: TF-IDF + FAISS + reranking
- Comparison style: zero-shot response vs grounded response
- Statistical test used: Wilcoxon signed-rank test

Additional subset notes:
- `PubMedQA` used the **real labeled subset**
- `MMedBench` used the **English subset** for compatibility with the current text pipeline

## Result table

| Dataset profile | Top-1 hit | Top-k hit | Zero factual | Grounded factual | Factual gain | Zero hallucination | Grounded hallucination | Hallucination drop | p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MMCQSD | 0.693 | 0.853 | 0.091 | 0.250 | 0.159 | 1.000 | 0.980 | 0.020 | 0.0000 |
| Open-i | 0.007 | 0.027 | 0.228 | 0.333 | 0.105 | 0.993 | 0.000 | 0.993 | 0.0000 |
| MMedBench | 0.807 | 0.967 | 0.294 | 0.323 | 0.029 | 0.940 | 0.933 | 0.007 | 0.0000 |
| PubMedQA | 0.980 | 0.980 | 0.601 | 0.560 | -0.042 | 0.047 | 0.000 | 0.047 | 0.0000 |

## Ranking summary
Based on the current subset comparison:

1. `MMCQSD`
2. `Open-i`
3. `MMedBench`
4. `PubMedQA`

This ranking is based mainly on:
- factual gain after grounding
- hallucination reduction
- retrieval quality under the same shared pipeline

## Dataset-wise interpretation

### 1. MMCQSD
`MMCQSD` was the strongest dataset in this comparison for the current project direction.

Why it performed well:
- It is closest in spirit to the **query side** of the project
- It contains code-mixed / user-query-like medical inputs
- Retrieval quality was reasonably strong:
  - top-1 hit: `0.693`
  - top-k hit: `0.853`
- Grounding improved factual score by `0.159`, which was the best gain among the four datasets

Important interpretation:
- Even though hallucination remained very high after grounding (`0.980`), this is not because the dataset is weak.
- It shows that the current lightweight generator is still very simple and not yet specialized for better answer control.
- In other words, `MMCQSD` is giving us the right kind of query setting, but the answer-generation component still needs improvement.

Practical value:
- This dataset is the strongest candidate for the **real Hinglish / code-mixed query side** of the main prototype.

### 2. Open-i
`Open-i` ranked second overall and remains highly important for the actual project objective.

Observed metrics:
- top-1 hit: `0.007`
- top-k hit: `0.027`
- factual gain after grounding: `0.105`
- hallucination drop: `0.993`

Why the retrieval numbers are low:
- Many `Open-i` entries in the current downloaded form use a very similar or generic prompt such as:
  - "What abnormality is present in this chest X-ray?"
- Because of that, the query side is weak and repetitive
- So the comparison does **not** show Open-i’s full value as a rich retrieval dataset

Why it still matters:
- `Open-i` gives the real radiology-style report evidence
- It is the strongest candidate for the **evidence/report side** of the main prototype
- The large hallucination drop indicates that once evidence is used, the output becomes much more grounded than the zero-shot baseline

Practical value:
- `Open-i` is not the best standalone comparison dataset under generic prompts
- But it is still essential for the final `Open-i + MMCQSD` grounded prototype

### 3. MMedBench
`MMedBench` performed reasonably well, especially on retrieval.

Observed metrics:
- top-1 hit: `0.807`
- top-k hit: `0.967`
- factual gain: `0.029`
- hallucination drop: `0.007`

Interpretation:
- Retrieval is strong because the questions and answer/rationale structure are relatively direct for text matching
- However, grounding only gave a small factual improvement
- Hallucination barely changed

What this means:
- `MMedBench` is useful as a medical benchmark-style dataset
- But it is less aligned with the exact target problem of:
  - Hinglish code-mixed medical queries
  - grounded report-based support
  - radiology-focused evidence use

Practical value:
- Good supplementary benchmark
- Not the best core dataset for the main prototype

### 4. PubMedQA
`PubMedQA` had the best retrieval numbers but the weakest comparison outcome for the current prototype objective.

Observed metrics:
- top-1 hit: `0.980`
- top-k hit: `0.980`
- zero-shot factual score: `0.601`
- grounded factual score: `0.560`
- factual gain: `-0.042`

Interpretation:
- Retrieval works very well because the dataset has very direct text alignment between question and supporting context
- But the grounded answer did not improve over zero-shot in this current setup
- This means the dataset is compatible with text retrieval, but not well aligned with the project’s intended answer format and use case

Why this likely happened:
- `PubMedQA` is abstract-based biomedical QA, not radiology-style report grounding
- The dataset’s evidence and target answer structure differ from the main project setup
- The current generic grounded response style does not fully exploit the QA format of this dataset

Practical value:
- Useful for supplementary text-only biomedical QA experiments
- Not a strong primary dataset for the main Hinglish grounded medical prototype

## Overall conclusions

### Main finding
The real comparison confirms that the best direction for the main prototype is **not** to treat all four datasets equally.

Instead, they naturally separate into roles:
- `MMCQSD` is best for the **query side**
- `Open-i` is best for the **evidence/report side**
- `PubMedQA` and `MMedBench` are useful as supporting comparison datasets, but not the best core pair for the main prototype

### Most important outcome
The comparison supports the decision to build the main real-data prototype around:

- `Open-i + MMCQSD`

This pairing is the most aligned with the project goal of:
- grounded medical support
- Hinglish/code-mixed user-facing queries
- report-backed evidence use

## Important limitations of this comparison
- This is still a **subset-level prototype comparison**, not a full benchmark on the complete datasets
- The current generator is intentionally lightweight and simple
- `Open-i` is underrepresented on the query side because many prompts are generic
- `MMedBench` was restricted to English only in this run
- `PubMedQA` and `MMedBench` are text-only benchmark styles, so they are not directly equivalent to `Open-i`

Because of these limitations, the results should be interpreted as:
- strong directional evidence for prototype planning
- not final research-paper-level conclusions yet

## Mentor-facing takeaway
This real-data comparison is an improvement over the earlier synthetic comparison and gives a clearer project direction.

The main conclusion is:
- `MMCQSD` is the strongest dataset for the current query setting
- `Open-i` remains essential as the real evidence/report source
- the main prototype should now move forward using `Open-i + MMCQSD`
- `PubMedQA` and `MMedBench` can remain supplementary comparison datasets rather than the core prototype pair

## Recommended next step
Update the main prototype pipeline so that:
- `Open-i` is used as the retrieval/report corpus
- `MMCQSD` is used as the real Hinglish query source
- the grounded vs zero-shot comparison is run on this real paired setup
