# H2 Re-analysis - Per-Arm Effects of Code-Mixing

Source: `results\combined_h1h2\combined_scored.csv` (n = 1165 cached generations, no new API calls).

The original analysis tested `factual_gain` (a difference of two noisy arms)
and found nothing. Testing the arms separately is better powered.

## Per-arm correlation with CMI

| Arm | n | Spearman rho | 95% CI | p (raw) | p (BH-FDR) | Kruskal-Wallis p |
|---|---:|---:|---|---:|---:|---:|
| Grounded factual support | 1165 | +0.0149 | [-0.043, +0.072] | 0.6119 | 0.6119 | 0.7753 |
| Zero-shot factual support | 1165 | -0.0677 | [-0.123, -0.011] | 0.0208 | 0.0416 | 0.1272 |
| Grounded hallucination | 1165 | +0.0610 | [+0.005, +0.118] | 0.0372 | 0.0496 | 0.0521 |
| Zero-shot hallucination | 1165 | +0.0812 | [+0.024, +0.137] | 0.0056 | 0.0223 | 0.0341 |

## Means by CMI tertile

| Bucket | n | Mean CMI | Grounded factual support | Zero-shot factual support | Grounded hallucination | Zero-shot hallucination |
|---|---:|---:|---:|---:|---:|---:|
| high_cm | 396 | 0.4932 | 0.5631 | 0.3028 | 0.2873 | 0.5324 |
| low_cm | 385 | 0.3513 | 0.5535 | 0.3517 | 0.2458 | 0.4516 |
| medium_cm | 384 | 0.4285 | 0.5437 | 0.3029 | 0.3078 | 0.5155 |

## Reading

Bootstrap CIs (10,000 resamples) decide 'flat' vs 'real effect'; a CI that
spans zero is flat. All p-values are Benjamini-Hochberg corrected across the
four tests in this family.

- Grounded factual support: **flat (rho=+0.0149, CI spans zero)**
- Zero-shot factual support: **declines (rho=-0.0677, BH p=0.0416)**
- Grounded hallucination: **rises (rho=+0.0610, BH p=0.0496)**
- Zero-shot hallucination: **rises (rho=+0.0812, BH p=0.0223)**

**On factual support the absorption is complete**: the grounded arm is flat
while the zero-shot arm declines significantly. **On hallucination it is only
partial** -- both arms rise with code-mixing; grounding slows the rise but does
not stop it. Write the claim that way. 'Grounding is robust to code-mixing'
overstates what these numbers support.

Two cautions for the write-up:

1. Grounded hallucination is significant only marginally after correction, so
   it is the one result here that a different metric could plausibly flip.
2. Zero-shot factual support is significant on the CONTINUOUS measure
   (Spearman) but not across TERTILES (Kruskal-Wallis) -- bucketing discards
   information and costs power. Report the continuous test as primary.

> CAVEAT: `cmi_score` here is the ORIGINAL 129-token-list measure, which
> counts `doctor`, `please` and `pls` as Hindi. Those appear in nearly every
> MMCQSD query, inflating CMI and compressing its variance. Re-run this after
> the CMI repair (Tier 1.5) before quoting any number in the paper.