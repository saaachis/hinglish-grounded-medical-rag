# Annotation instructions

Two sheets, two annotators. Work **independently** -- do not discuss items while
labelling, or the agreement statistic becomes meaningless. Columns starting with
`_hidden_` are for scoring afterwards: **do not read them while labelling**, and
do not edit them.

If a clinician is available, their copy is the one that validates the lexicon.
A second, non-clinician copy is still useful: it measures how reproducible the
judgments are.

## Sheet A -- answers (`annotation_sheet_A_*.csv`)

For each row, read `patient_question` and `answer_text`, then fill three columns.

1. **`wrong_concepts`** -- of the terms listed in `extractor_concepts`, write the
   ones the answer does **not** actually assert about this patient. Include any
   the answer **denies** ("rash nahi hai") or merely mentions hypothetically.
   Comma-separated, blank if all are correct.
2. **`missed_concepts`** -- clinical findings the answer **does** assert but which
   are missing from `extractor_concepts`. Use the vocabulary in
   `concept_vocabulary.csv`; if the right word is not in that list, write it
   anyway and mark it with a `*`. Blank if nothing was missed.
3. **`appropriateness`** -- how clinically useful the answer is for this patient:
   - `0` wrong, unsafe, or off-topic
   - `1` not wrong but unhelpful (vague, hedged, or refuses without direction)
   - `2` clinically appropriate and responsive

## Sheet B -- retrieved cases (`annotation_sheet_B_*.csv`)

For each row, read the patient's question and the retrieved case, then fill
**`relevance`**:
   - `0` not relevant
   - `1` same clinical area, but would not answer this patient's question
   - `2` directly relevant: a clinician could use this case to answer it

The same question appears several times with different cases, in a shuffled
order. That is intentional. Judge each row on its own.

## When finished

Save each file under its own name, keeping the `_annotatorA` / `_annotatorB`
suffix, then run:

    python -m src.evaluation.annotation_agreement
