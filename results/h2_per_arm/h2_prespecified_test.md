# H02 under its pre-specified test

n = 1165 cached generations (llama-3.1-8b-instant, oracle evidence). No API calls.
Code-mixing levels = tertiles of `hindi_prop_v2` (repaired Hindi-word list).

## Why this exists

Reviewer 3: H02 concerns the grounded-minus-ungrounded difference but the paper tested
each arm separately. The planned test (README, commit 0917322, 9 Feb 2026) was a
two-way ANOVA / Kruskal-Wallis. With arm varying within query, the arm x level
interaction is the test of whether the paired gain changes across levels.

## Results

| Test | Role | ANOVA F (p) | Kruskal-Wallis H (p) | KW BH p (pre-specified pair) | Spearman rho [95% CI], p | Equivalence (SESOI 0.10, post hoc) |
|---|---|---|---|---:|---|---|
| H02 as worded in the paper: interaction (gain across levels) | PRIMARY (pre-specified) | 3.03 (0.0486) | 5.11 (0.0779) | 0.156 | +0.081 [+0.023, +0.138], p = 0.00581 | non-zero |
| H2 as worded 9 Feb: grounded arm across levels | pre-specified (README wording) | 0.14 (0.867) | 0.31 (0.855) | 0.855 | -0.001 [-0.058, +0.057], p = 0.983 | equivalent to zero (90% CI inside +/-0.1) |
| Zero-shot arm across levels | exploratory decomposition | 5.58 (0.00386) | 11.81 (0.00272) | — | -0.116 [-0.171, -0.060], p = 7.74e-05 | non-zero |
| Interaction, originally shipped code-mixing measure | sensitivity | 1.80 (0.165) | 3.30 (0.193) | — | +0.070 [+0.013, +0.127], p = 0.0166 | non-zero |

## Means by code-mixing tertile

| Level | n | Hindi proportion | Grounded | Zero-shot | Gain |
|---|---:|---:|---:|---:|---:|
| low | 388 | 0.582 | 0.5523 | 0.3481 | +0.2041 |
| medium | 388 | 0.711 | 0.5614 | 0.3362 | +0.2253 |
| high | 389 | 0.795 | 0.5469 | 0.2728 | +0.2741 |

## Reading

The two pre-specified tertile tests DISAGREE, so neither is reported alone:
- **Interaction (H02 as worded in the paper):** ANOVA F = 3.03, p = 0.0486; Kruskal-Wallis H = 5.11, p = 0.0779. The plan named both. The outcome is bounded and non-normal, which favours the rank test, but choosing the test after seeing both p-values is exactly what must not be done. **Evidence against H02 is borderline under the pre-specified tertile tests.**
- **Continuous version (more power, not pre-specified):** rho = +0.081 [+0.023, +0.138], p = 0.00581. The grounding gain RISES with Hindi proportion, from +0.2041 in the lowest tertile to +0.2741 in the highest.
- **Why it rises:** the grounded arm is flat -- statistically equivalent to zero effect (rho = -0.001, 90% CI [-0.048, +0.049] inside +/-0.1) -- while the zero-shot arm degrades. The gain grows because the baseline falls, not because grounding improves.
- **Grounded arm alone (H2 as worded 9 Feb):** not rejected, and equivalent to flat.
- **Sensitivity:** under the originally shipped code-mixing measure the interaction is non-significant (KW p = 0.193). The paper's statement that the difference test is non-significant reflects that measure, not the repaired one, and must be updated.

**Defensible H02 statement:** grounded factual support is stable across code-mixing intensity (equivalent to flat); the grounding gain increases with Hindi proportion on the continuous measure, with borderline support under the pre-specified tertile tests, because ungrounded performance degrades. The per-arm results are the decomposition of this test, not a replacement for it.
