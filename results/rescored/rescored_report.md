# Re-scored results (repaired lexicon)

Every cached generation re-scored with negation fixed and precision/recall/F1
reported separately. No API calls. `hallucination` is omitted throughout: it is
exactly `1 - precision`, so reporting it separately double-counts one result.

## Contrasts (grounded vs zero-shot), BH-corrected across the whole family

| Source | Arm | Reference | Metric | n | zero | grounded | delta | d | p (BH) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| gptoss120b | oracle | caption (unbiased) | f1 | 189 | 0.1999 | 0.1792 | **-0.0207** | -0.067 | 0.44 |
| gptoss120b | oracle | caption (unbiased) | precision | 214 | 0.1351 | 0.1565 | **+0.0215** | 0.078 | 0.554 |
| gptoss120b | oracle | caption (unbiased) | recall | 189 | 0.3501 | 0.2116 | **-0.1384** | -0.299 | 0.000247 |
| gptoss120b | oracle | evidence (circular) | f1 | 207 | 0.2520 | 0.3446 | **+0.0926** | 0.284 | 0.00042 |
| gptoss120b | oracle | evidence (circular) | precision | 214 | 0.2756 | 0.5364 | **+0.2608** | 0.640 | 4.25e-15 |
| gptoss120b | oracle | evidence (circular) | recall | 207 | 0.2761 | 0.2903 | **+0.0142** | 0.037 | 0.673 |
| gptoss120b | real | caption (unbiased) | f1 | 200 | 0.2058 | 0.1589 | **-0.0469** | -0.171 | 0.0106 |
| gptoss120b | real | caption (unbiased) | precision | 221 | 0.1428 | 0.1367 | **-0.0060** | -0.024 | 0.56 |
| gptoss120b | real | caption (unbiased) | recall | 200 | 0.3642 | 0.2125 | **-0.1517** | -0.372 | 6.86e-06 |
| gptoss120b | real | evidence (circular) | f1 | 210 | 0.2533 | 0.2213 | **-0.0320** | -0.118 | 0.0408 |
| gptoss120b | real | evidence (circular) | precision | 221 | 0.2845 | 0.3549 | **+0.0704** | 0.191 | 0.00747 |
| gptoss120b | real | evidence (circular) | recall | 210 | 0.2695 | 0.1826 | **-0.0869** | -0.286 | 5.99e-05 |
| gptoss20b_n268 | oracle | caption (unbiased) | f1 | 133 | 0.2555 | 0.2339 | **-0.0217** | -0.057 | 0.574 |
| gptoss20b_n268 | oracle | caption (unbiased) | precision | 140 | 0.1873 | 0.2337 | **+0.0464** | 0.119 | 0.263 |
| gptoss20b_n268 | oracle | caption (unbiased) | recall | 133 | 0.4499 | 0.2632 | **-0.1867** | -0.373 | 0.000358 |
| gptoss20b_n268 | oracle | evidence (circular) | f1 | 134 | 0.3202 | 0.3135 | **-0.0067** | -0.020 | 0.68 |
| gptoss20b_n268 | oracle | evidence (circular) | precision | 140 | 0.3317 | 0.5283 | **+0.1966** | 0.453 | 1.81e-06 |
| gptoss20b_n268 | oracle | evidence (circular) | recall | 134 | 0.3608 | 0.2666 | **-0.0943** | -0.251 | 0.00669 |
| gptoss20b_n268 | real | caption (unbiased) | f1 | 130 | 0.2776 | 0.2203 | **-0.0574** | -0.156 | 0.0912 |
| gptoss20b_n268 | real | caption (unbiased) | precision | 138 | 0.1985 | 0.2168 | **+0.0183** | 0.048 | 0.701 |
| gptoss20b_n268 | real | caption (unbiased) | recall | 130 | 0.5038 | 0.2436 | **-0.2603** | -0.552 | 1.04e-06 |
| gptoss20b_n268 | real | evidence (circular) | f1 | 130 | 0.2929 | 0.2231 | **-0.0698** | -0.256 | 0.00659 |
| gptoss20b_n268 | real | evidence (circular) | precision | 138 | 0.3054 | 0.4124 | **+0.1071** | 0.255 | 0.00235 |
| gptoss20b_n268 | real | evidence (circular) | recall | 130 | 0.3371 | 0.1758 | **-0.1614** | -0.519 | 4.07e-07 |
| llama_oracle_n1165 | grounded | caption (unbiased) | f1 | 613 | 0.1314 | 0.1935 | **+0.0620** | 0.223 | 3e-07 |
| llama_oracle_n1165 | grounded | caption (unbiased) | precision | 669 | 0.1146 | 0.1722 | **+0.0576** | 0.207 | 4.74e-07 |
| llama_oracle_n1165 | grounded | caption (unbiased) | recall | 613 | 0.1615 | 0.2389 | **+0.0774** | 0.222 | 9.29e-07 |
| llama_oracle_n1165 | grounded | evidence (circular) | f1 | 651 | 0.1630 | 0.3660 | **+0.2029** | 0.683 | 2.09e-48 |
| llama_oracle_n1165 | grounded | evidence (circular) | precision | 669 | 0.2796 | 0.5758 | **+0.2962** | 0.720 | 2.83e-49 |
| llama_oracle_n1165 | grounded | evidence (circular) | recall | 651 | 0.1331 | 0.2996 | **+0.1665** | 0.581 | 2.04e-38 |

## ⚠️ Degenerate baselines on the unbiased (caption) reference

`precision` has no recall term, so its optimum is a one-word answer. Any absolute
level below these rows is a metric failure, not a system failure.

| System | Answer | precision | n |
|---|---|---:|---:|
| `copy:reference` | <the reference verbatim> | 1.0000 | 1705 |
| `const:swelling` | swelling | 0.7132 | 1876 |
| `const:swelling+erythema` | swelling and erythema | 0.6586 | 1876 |
| `const:erythema` | erythema | 0.6039 | 1876 |
| `const:six-common` | rash swelling pain erythema infection lesion | 0.2338 | 1876 |
| `const:pain` | pain | 0.0032 | 1876 |

## Refusal / coverage -- report beside every number above

| Source | Arm | outputs asserting >=1 concept |
|---|---|---:|
| llama_oracle_n1165 | zero | 73.0% |
| llama_oracle_n1165 | grounded | 71.2% |
| gptoss20b_n268 | zero | 91.0% |
| gptoss20b_n268 | oracle | 54.5% |
| gptoss20b_n268 | real | 54.1% |
| gptoss120b | zero | 92.4% |
| gptoss120b | oracle | 47.5% |
| gptoss120b | real | 49.9% |

> Rows where an arm asserts no concept score `nan` and vanish from a naive
> mean. Coverage must be reported or the system looks healthiest where it fails.
