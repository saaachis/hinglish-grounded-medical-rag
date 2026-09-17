# Validating the concept extractor without annotators

Clinician annotation was out of scope for this revision. This is what can be
measured automatically. **It does not license the phrase "clinically validated";
clinician validation remains a stated limitation.**

## Part 1 — what the 26 concepts can represent at all

| Quantity | Value |
|---|---:|
| References (gold English summaries) | 3,015 |
| References containing >= 1 lexicon concept | 98.3% |
| Evidence texts containing >= 1 lexicon concept | 95.7% |
| Mean concepts per reference | 2.76 |
| Median concepts per reference | 3 |
| References with exactly one concept | 13.9% |

A reference carrying one or two concepts gives precision and recall very few
events to work with, which is why single-reference scores are unstable and why
the paper reports paired deltas rather than absolute levels.

### Frequent clinical terms the lexicon has no word for

| Term | Occurrences | Close to a lexicon word? |
|---|---:|---|
| `multuiple` | 48 | — |
| `imner` | 46 | — |
| `diagnosis` | 44 | — |
| `peels` | 37 | — |
| `dryskin` | 27 | — |
| `discolour` | 22 | — |
| `noel` | 22 | — |
| `buldge` | 21 | — |
| `bothe` | 20 | — |
| `inwards` | 18 | — |
| `rasied` | 17 | — |
| `stye` | 16 | — |
| `bumpy` | 16 | — |
| `itichy` | 15 | `itchy` |
| `teeths` | 14 | — |
| `muliple` | 13 | — |
| `andsilvery` | 13 | — |
| `withe` | 12 | — |
| `prese` | 12 | — |
| `stomatitis` | 12 | — |

Of the 30 most frequent uncovered terms, **2 are close variants of words the lexicon already has** (the reference captions are typed free text and contain frequent misspellings). Those are extractor gaps, fixable with fuzzy matching. The rest are findings the 26-item vocabulary genuinely cannot represent.

Either way the consequence is the same for interpretation: the metric measures overlap
on a small vocabulary, not clinical correctness, and the unbiased reference is the
text most affected, since it is a short free-text caption.

## Part 2 — agreement with an independent LLM reader

Judge: `qwen/qwen3.8-27b`, 120 answers.

| Quantity | Value |
|---|---:|
| Extractor precision vs judge | 0.412 |
| Extractor recall vs judge | 0.686 |
| Extractor F1 vs judge | 0.515 |
| Cohen's kappa (per-concept decisions) | 0.494 |
| Concepts agreed / extractor-only / judge-only | 70 / 100 / 32 |

### Where they disagree most

| Concept and direction | Count |
|---|---:|
| infection (extractor only) | 16 |
| pain (judge only) | 14 |
| erythema (extractor only) | 10 |
| pruritus (extractor only) | 10 |
| swelling (judge only) | 8 |
| rash (extractor only) | 8 |
| dermatitis (extractor only) | 7 |
| malignancy (extractor only) | 6 |
| allergy (extractor only) | 6 |
| mass (extractor only) | 6 |
| pain (extractor only) | 5 |
| fungal (extractor only) | 5 |

### Agreement by arm — does the bias cancel in the paired delta?

| Arm | n | agreed | extractor only | judge only | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| grounded | 60 | 36 | 39 | 19 | 0.480 | 0.655 |
| zero_shot | 60 | 34 | 61 | 13 | 0.358 | 0.723 |

The extractor's precision against the judge differs by arm (0.480 grounded vs 0.358 zero-shot, gap +0.122), so **the bias does not cancel in the paired difference**.

Direction matters. `factual_support` is |answer concepts in reference| /
|answer concepts|, so a spurious extra concept inflates the denominator and
DEPRESSES the score. The extractor over-attributes more in the zero-shot arm,
which depresses the zero-shot score more than the grounded one and therefore
**inflates the measured grounding benefit**. Part of H01's effect may be an
extraction artefact; the direction of the bias favours the reported result, so
the paper must say so rather than leave it for a reader to find.


Agreement is **moderate** (kappa = 0.494). 
An extractor-only disagreement is a likely false positive (a mention the answer
did not assert); a judge-only disagreement is a likely miss (wording the regexes
do not cover). Report both numbers in Sect. 6.1 next to the lexicon limitation.
