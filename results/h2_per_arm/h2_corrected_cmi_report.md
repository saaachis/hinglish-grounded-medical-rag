# H2 under the corrected Code-Mixing Index

n = 1165 cached generations. No API calls.

## 1. Why the legacy measure was replaced

Measured on the 3,015 MMCQSD queries:

- `doctor` (English) is in the Hindi list and fires in **71.0%** of queries;
  `please` in **38.0%**.
- **32.9%** of all query tokens were unknown to both the Hindi list and an
  English vocabulary; the frequency-ranked OOV list is almost entirely
  romanised Hindi (`mein` 5,340, `mere` 3,945, `hoon` 2,572, ...).

Over-counting and under-counting both compress the score toward the middle.

## 2. Distributions

| Measure | mean | SD | min | median | max | IQR |
|---|---:|---:|---:|---:|---:|---:|
| `cmi_score` | 0.4250 | 0.0747 | 0.0000 | 0.4286 | 1.0000 | 0.0798 |
| `cmi_legacy_recomputed` | 0.4250 | 0.0747 | 0.0000 | 0.4286 | 1.0000 | 0.0798 |
| `hindi_prop_v2` | 0.6964 | 0.1146 | 0.0000 | 0.7115 | 0.9655 | 0.1119 |
| `cmi_v2_exclude` | 29.0273 | 8.7797 | 0.0000 | 28.5714 | 50.0000 | 11.0873 |
| `cmi_v2_hindi` | 28.9301 | 8.7677 | 0.0000 | 28.5714 | 50.0000 | 11.1039 |
| `cmi_v2_english` | 29.2785 | 8.7423 | 0.0000 | 28.8462 | 50.0000 | 11.0828 |

## 2b. TWO CHANGES, NOT ONE -- read this before quoting any number

Replacing the legacy measure changed two things at once:

1. **The lexicon was repaired** (contaminants removed, OOV Hindi added).
2. **The construct changed.** The legacy measure is a Hindi PROPORTION.
   Das & Gamback CMI is a mixing-BALANCE measure: maximal at a 50/50 mix and
   **zero for monolingual text in either language**. A 90%-Hindi query scores
   HIGH on proportion and LOW on CMI.

`hindi_prop_v2` isolates change (1): repaired lexicon, same construct.

| Comparison | Spearman rho | p |
|---|---:|---:|
| legacy vs Das & Gamback CMI | -0.5296 | 3.37e-85 |
| legacy vs repaired proportion | +0.5817 | 1.92e-106 |
| repaired proportion vs CMI | -0.9422 | 0 |

The strong NEGATIVE legacy-vs-CMI correlation is expected and is not evidence
that either is wrong -- it is the proportion-vs-balance distinction. Compare
the legacy row against `hindi_prop_v2` to judge the lexicon repair, and
against `cmi_v2_*` to judge the construct change.

## 3. Per-arm effects, under each ambiguity policy

`exclude` treats Hindi/English homographs (`me`, `sir`, `pet`, `pair`) as
language-independent; `hindi` and `english` force them either way. Agreement
across all three means the conclusion does not rest on that judgement.

| CMI variant | Arm | rho | 95% CI | p (BH) | Verdict |
|---|---|---:|---|---:|---|
| `cmi_score (legacy, as shipped)` | Grounded factual support | +0.0149 | [-0.043, +0.071] | 0.6119 | **flat** |
| `cmi_score (legacy, as shipped)` | Zero-shot factual support | -0.0677 | [-0.123, -0.011] | 0.0416 | **declines** |
| `cmi_score (legacy, as shipped)` | Grounded hallucination | +0.0610 | [+0.005, +0.117] | 0.0496 | **rises** |
| `cmi_score (legacy, as shipped)` | Zero-shot hallucination | +0.0812 | [+0.024, +0.138] | 0.0223 | **rises** |
| `hindi_prop_v2 (repaired lexicon, SAME construct)` | Grounded factual support | -0.0006 | [-0.058, +0.057] | 0.9829 | **flat** |
| `hindi_prop_v2 (repaired lexicon, SAME construct)` | Zero-shot factual support | -0.1155 | [-0.171, -0.059] | 0.0003 | **declines** |
| `hindi_prop_v2 (repaired lexicon, SAME construct)` | Grounded hallucination | -0.0224 | [-0.080, +0.036] | 0.5932 | **flat** |
| `hindi_prop_v2 (repaired lexicon, SAME construct)` | Zero-shot hallucination | +0.0420 | [-0.018, +0.101] | 0.3031 | **flat** |
| `cmi_v2_exclude` | Grounded factual support | +0.0266 | [-0.030, +0.084] | 0.4864 | **flat** |
| `cmi_v2_exclude` | Zero-shot factual support | +0.1143 | [+0.059, +0.171] | 0.0004 | **rises** |
| `cmi_v2_exclude` | Grounded hallucination | +0.0488 | [-0.010, +0.107] | 0.1921 | **flat** |
| `cmi_v2_exclude` | Zero-shot hallucination | -0.0176 | [-0.076, +0.041] | 0.5477 | **flat** |
| `cmi_v2_hindi` | Grounded factual support | +0.0280 | [-0.028, +0.085] | 0.4519 | **flat** |
| `cmi_v2_hindi` | Zero-shot factual support | +0.1163 | [+0.062, +0.173] | 0.0003 | **rises** |
| `cmi_v2_hindi` | Grounded hallucination | +0.0482 | [-0.010, +0.106] | 0.1996 | **flat** |
| `cmi_v2_hindi` | Zero-shot hallucination | -0.0179 | [-0.077, +0.041] | 0.5418 | **flat** |
| `cmi_v2_english` | Grounded factual support | +0.0228 | [-0.035, +0.080] | 0.5838 | **flat** |
| `cmi_v2_english` | Zero-shot factual support | +0.1095 | [+0.054, +0.166] | 0.0007 | **rises** |
| `cmi_v2_english` | Grounded hallucination | +0.0507 | [-0.008, +0.109] | 0.1673 | **flat** |
| `cmi_v2_english` | Zero-shot hallucination | -0.0157 | [-0.074, +0.043] | 0.5924 | **flat** |

## 4. Robustness

- **Grounded factual support**: STABLE across policies -> ['flat']
- **Zero-shot factual support**: POLICY-DEPENDENT across policies -> ['declines', 'rises']
- **Grounded hallucination**: STABLE across policies -> ['flat']
- **Zero-shot hallucination**: STABLE across policies -> ['flat']

Report any POLICY-DEPENDENT row as an explicit limitation; do not pick the
policy that gives the preferred answer.

## 5. Resolution -- which construct to report, and what H2 actually says

Repaired proportion and Das & Gamback CMI correlate at **rho = -0.9422** on this corpus: they are near-perfect
inverses. These queries are Hindi-dominant (mean Hindi proportion 0.696), so adding Hindi moves text AWAY from a 50/50
balance. 'Rises with CMI' and 'declines with Hindi proportion' are therefore
**the same finding stated on inverted scales**, not a contradiction.

**Report `hindi_prop_v2` as primary.** The hypothesis is whether more Hindi
degrades an English-centric pipeline, which is a question about proportion.
Report Das & Gamback CMI as the standard-metric cross-check.

The lexicon repair did not overturn the factual-support result -- it
strengthened it. On the SAME construct, the zero-shot decline goes from
rho = -0.068 (BH p = 0.042) under the contaminated lexicon to rho = -0.116
(BH p = 0.0003) under the repaired one, while the grounded arm stays flat.

The hallucination effects, however, do NOT survive the repair. Under the
legacy lexicon both arms appeared to rise; under the repaired lexicon both are
flat. Those were lexicon artefacts and must not be reported as findings.

**Defensible H2 claim:** increasing Hindi content significantly degrades
zero-shot factual support while leaving grounded factual support unchanged;
hallucination rates are unaffected by Hindi content in either arm.