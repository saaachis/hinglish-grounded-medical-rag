# Dataset Download Summary

## Purpose
This note summarizes how the currently selected datasets were downloaded, what form they were converted into for the prototype, and what was intentionally skipped in this phase.

The goal of this step was to make all datasets lightweight, scriptable, and directly usable by the team for the text-first prototype and dataset comparison work.

## Overall approach
- All datasets were downloaded through stable Hugging Face access routes.
- The current phase is text-first, so large image binaries were not downloaded into the working CSV files.
- Wherever possible, the original text content was preserved and only reorganized into simpler CSV structures for downstream use.
- For image-based datasets, image references were preserved as lightweight path/reference fields instead of embedding image data.
- Two output forms were maintained:
  - `raw`: compact CSV export of the fetched dataset content relevant to our work
  - `processed`: cleaned/standardized CSV prepared for easier pipeline use

## Dataset-wise summary

### 1. Open-i
- Source route: `https://huggingface.co/datasets/ayyuce/Indiana_University_Chest_X-ray_Collection`
- Fallback route kept in script: `https://huggingface.co/datasets/dz-osamu/IU-Xray`
- Nature: multimodal chest X-ray dataset
- Current prototype form: text-first export with report text and image reference only
- Images downloaded now: no

Downloaded outputs:
- `data/raw/openi/openi_train_raw.csv`
- `data/processed/openi_reports.csv`

Current counts and size:
- Raw rows: `6687`
- Processed rows: `6687`
- Raw size: about `2.55 MB`
- Processed size: about `2.55 MB`

Fields kept in current CSV form:
- record/report identifier
- question or prompt if available
- report text
- image reference

Phase note:
- This dataset is originally image-related, but for the current prototype we kept only report-side text plus image references so the team can work quickly without heavy storage.

### 2. MMCQSD
- Source route: `https://huggingface.co/datasets/ArkaAcharya/MMCQSD`
- Nature: medical query dataset with code-mixed / query-target style structure
- Current prototype form: text-first export with query, target/summary, and image reference if present
- Images downloaded now: no

Downloaded outputs:
- `data/raw/mmcqsd/mmcqsd_train_raw.csv`
- `data/processed/mmcqsd_queries.csv`

Current counts and size:
- Raw rows: `3015`
- Processed rows: `3015`
- Raw size: about `2.69 MB`
- Processed size: about `2.74 MB`

Fields kept in current CSV form:
- sample identifier
- Hinglish/query text
- English summary or target
- image reference

Phase note:
- This dataset is useful because it is closer to the code-mixed medical query setting of our project, but we still kept the stored version lightweight by excluding image binaries.

### 3. PubMedQA
- Source route: `https://huggingface.co/datasets/qitai123/PubMedQA`
- Nature: text-only biomedical question answering dataset based on PubMed abstracts
- Current prototype form: text-only CSV
- Images downloaded now: not applicable

Downloaded outputs:
- `data/raw/pubmedqa/pubmedqa_all_raw.csv`
- `data/processed/pubmedqa_records.csv`

Current counts and size:
- Raw rows: `273518`
- Processed rows: `273518`
- Raw size: about `487.20 MB`
- Processed size: about `487.20 MB`

Content preserved in current CSV form:
- PubMed identifier
- subset name (`pqa_labeled`, `pqa_unlabeled`, `pqa_artificial`)
- split
- question
- abstract/context text
- long answer / rationale text
- final decision label

Phase note:
- This dataset is much larger than Open-i and MMCQSD in text volume because it is a large QA corpus, even though it has no images.

### 4. MMedBench
- Source route: `https://huggingface.co/datasets/Henrychur/MMedBench`
- Downloaded archive: `MMedBench.zip`
- Nature: text-only multilingual medical benchmark with multiple-choice questions and rationales
- Current prototype form: text-only CSV
- Images downloaded now: not applicable

Downloaded outputs:
- `data/raw/mmedbench/mmedbench_all_raw.csv`
- `data/processed/mmedbench_questions.csv`

Current counts and size:
- Raw rows: `53566`
- Processed rows: `53566`
- Raw size: about `65.29 MB`
- Processed size: about `65.29 MB`

Content preserved in current CSV form:
- generated record/sample identifier
- split
- language
- question
- options
- answer text
- answer option index
- rationale
- meta information

Important implementation note:
- The default Hugging Face dataset loader showed schema inconsistencies across files for this benchmark.
- To avoid losing data or forcing manual edits, the official zip archive was downloaded and the contained JSONL files were parsed directly.
- This still preserves the benchmark text faithfully while making it usable in our CSV-based workflow.

## What was skipped in this phase
- No image binaries were downloaded into the working dataset files.
- No image bytes were embedded inside CSVs.
- Extra heavy metadata fields that were not needed for the current prototype pipeline were not expanded into separate artifacts.

## Why this conversion is valid for the prototype
- The current prototype is text-first and focused on retrieval, grounding, and comparison behavior.
- The important textual content required for experimentation has been preserved.
- The conversion changes the storage format, not the underlying report/question text.
- This keeps the repository lighter, easier to share, and faster to use for all group members.

## Key mentor-facing takeaway
- `Open-i` and `MMCQSD` are smaller in the current repository because we intentionally kept only the text/report side and image references.
- `PubMedQA` and `MMedBench` are text-only datasets, so they appear larger in CSV size mainly because they contain many more text records.
- This staged download strategy supports fast prototype development now, while keeping room for a later multimodal phase if image-based experiments are required.
