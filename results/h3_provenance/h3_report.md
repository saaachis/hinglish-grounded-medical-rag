# H03: evidence provenance

n = 160 queries x 4 evidence conditions. Corpora are topically matched and equal-sized; only the evidence type changes.
Answers are scored against the UNBIASED caption reference, never against the evidence they were given.

| Evidence | mean F1 | mean precision | mean recall | refusal | n scoreable |
|---|---:|---:|---:|---:|---:|
| `multicare` | 0.3265 | 0.3254 | 0.3598 | 88.1% | 63 |
| `pubmedqa` | 0.3476 | 0.3738 | 0.3810 | 79.4% | 70 |
| `mmedbench` | 0.2659 | 0.2707 | 0.3030 | 76.2% | 66 |
| `shuffled` | 0.3186 | 0.3347 | 0.3716 | 82.5% | 61 |

## Evidence length actually supplied (confound check)

| Evidence | mean words in prompt |
|---|---:|
| `multicare` | 360 |
| `pubmedqa` | 187 |
| `mmedbench` | 280 |
| `shuffled` | 335 |

The prompt caps evidence at 400 words. Case narratives are long enough to hit that cap while abstracts and exam text are not, so MultiCaRe is handed more text. Any advantage it shows is therefore an upper bound on a provenance effect, and partly a length effect.


## Refusal rate by evidence type (all rows -- the well-powered test)

Cochran's Q = 9.092, df = 3, **p = 0.02809** (n = 160 rows).

Evidence type significantly changes how often the model REFUSES to answer.

This is the only H03 test with full power, because refusal is defined on every row whereas concept F1 is undefined when an arm asserts no concept.

## Pairwise answer quality (rows where BOTH arms produced a scoreable answer)

| Comparison | n | delta F1 | Wilcoxon p |
|---|---:|---:|---:|
| multicare − pubmedqa | 38 | -0.0553 | 0.238 |
| multicare − mmedbench | 34 | -0.0045 | 0.958 |
| multicare − shuffled | 32 | +0.0094 | 1.000 |
| pubmedqa − mmedbench | 34 | +0.0847 | 0.210 |
| pubmedqa − shuffled | 38 | +0.0693 | 0.360 |
| mmedbench − shuffled | 34 | -0.0259 | 0.550 |

## Omnibus test — UNDERPOWERED (n = 13 complete cases)

Friedman chi-square = 0.063, p = 0.9959

> **This omnibus result must not be read as a null.** It rests on 13 rows where all 4 conditions happened to produce a scoreable answer simultaneously — a joint event with probability ~0.1% given the observed refusal rates. The test has almost no power to detect an effect of any plausible size, so the correct statement is that **H03 remains undetermined for answer quality**, while §refusal above shows evidence type does affect refusal behaviour.

> Reaching ~40 complete cases would need roughly 39294 queries at the current refusal rates, or a prompt that refuses less.


## Does discourse structure matter?

MultiCaRe minus sentence-shuffled MultiCaRe: **+0.0094** (n = 32, p = 1).

Shuffling sentences does not significantly change answer quality (n = 32). This is suggestive rather than conclusive at this sample size, but if it holds it means what grounding extracts behaves like a bag of clinical terms rather than a coherent narrative.