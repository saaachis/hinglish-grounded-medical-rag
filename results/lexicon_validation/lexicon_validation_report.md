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

_Not run. Use `python -m src.analysis.lexicon_validation` with API access._
