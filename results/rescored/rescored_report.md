# Re-scored results (repaired lexicon)

Every cached generation re-scored with negation fixed and precision/recall/F1
reported separately. No API calls. `hallucination` is omitted throughout: it is
exactly `1 - precision`, so reporting it separately double-counts one result.

## Contrasts (grounded vs zero-shot), BH-corrected across the whole family

| Source | Arm | Reference | Metric | n | zero | grounded | delta | d | p (BH) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| gptoss120b | oracle | caption (unbiased) | f1 | 129 | 0.1936 | 0.1850 | **-0.0085** | -0.027 | 0.806 |
| gptoss120b | oracle | caption (unbiased) | precision | 147 | 0.1227 | 0.1553 | **+0.0327** | 0.116 | 0.376 |
| gptoss120b | oracle | caption (unbiased) | recall | 129 | 0.3579 | 0.2261 | **-0.1318** | -0.288 | 0.00318 |
| gptoss120b | oracle | evidence (circular) | f1 | 141 | 0.2425 | 0.3504 | **+0.1080** | 0.333 | 0.000561 |
| gptoss120b | oracle | evidence (circular) | precision | 147 | 0.2500 | 0.5200 | **+0.2700** | 0.660 | 4.73e-11 |
| gptoss120b | oracle | evidence (circular) | recall | 141 | 0.2646 | 0.2970 | **+0.0323** | 0.089 | 0.558 |
| gptoss120b | real | caption (unbiased) | f1 | 139 | 0.1953 | 0.1559 | **-0.0394** | -0.145 | 0.096 |
| gptoss120b | real | caption (unbiased) | precision | 153 | 0.1320 | 0.1382 | **+0.0062** | 0.024 | 0.965 |
| gptoss120b | real | caption (unbiased) | recall | 139 | 0.3633 | 0.2050 | **-0.1583** | -0.396 | 7.63e-05 |
| gptoss120b | real | evidence (circular) | f1 | 146 | 0.2517 | 0.2217 | **-0.0299** | -0.107 | 0.132 |
| gptoss120b | real | evidence (circular) | precision | 153 | 0.2738 | 0.3478 | **+0.0740** | 0.203 | 0.0238 |
| gptoss120b | real | evidence (circular) | recall | 146 | 0.2627 | 0.1818 | **-0.0809** | -0.281 | 0.00136 |
| gptoss20b_n268 | oracle | caption (unbiased) | f1 | 133 | 0.2555 | 0.2339 | **-0.0217** | -0.057 | 0.596 |
| gptoss20b_n268 | oracle | caption (unbiased) | precision | 140 | 0.1873 | 0.2337 | **+0.0464** | 0.119 | 0.263 |
| gptoss20b_n268 | oracle | caption (unbiased) | recall | 133 | 0.4499 | 0.2632 | **-0.1867** | -0.373 | 0.000418 |
| gptoss20b_n268 | oracle | evidence (circular) | f1 | 134 | 0.3202 | 0.3135 | **-0.0067** | -0.020 | 0.706 |
| gptoss20b_n268 | oracle | evidence (circular) | precision | 140 | 0.3317 | 0.5283 | **+0.1966** | 0.453 | 1.81e-06 |
| gptoss20b_n268 | oracle | evidence (circular) | recall | 134 | 0.3608 | 0.2666 | **-0.0943** | -0.251 | 0.00669 |
| gptoss20b_n268 | real | caption (unbiased) | f1 | 130 | 0.2776 | 0.2203 | **-0.0574** | -0.156 | 0.1 |
| gptoss20b_n268 | real | caption (unbiased) | precision | 138 | 0.1985 | 0.2168 | **+0.0183** | 0.048 | 0.751 |
| gptoss20b_n268 | real | caption (unbiased) | recall | 130 | 0.5038 | 0.2436 | **-0.2603** | -0.552 | 1.04e-06 |
| gptoss20b_n268 | real | evidence (circular) | f1 | 130 | 0.2929 | 0.2231 | **-0.0698** | -0.256 | 0.00659 |
| gptoss20b_n268 | real | evidence (circular) | precision | 138 | 0.3054 | 0.4124 | **+0.1071** | 0.255 | 0.0025 |
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
| `copy:reference` | <the reference verbatim> | 1.0000 | 1586 |
| `const:swelling` | swelling | 0.7144 | 1744 |
| `const:swelling+erythema` | swelling and erythema | 0.6611 | 1744 |
| `const:erythema` | erythema | 0.6078 | 1744 |
| `const:six-common` | rash swelling pain erythema infection lesion | 0.2352 | 1744 |
| `const:pain` | pain | 0.0029 | 1744 |

## Refusal / coverage -- report beside every number above

| Source | Arm | outputs asserting >=1 concept |
|---|---|---:|
| llama_oracle_n1165 | zero | 73.0% |
| llama_oracle_n1165 | grounded | 71.2% |
| gptoss20b_n268 | zero | 91.0% |
| gptoss20b_n268 | oracle | 54.5% |
| gptoss20b_n268 | real | 54.1% |
| gptoss120b | zero | 91.5% |
| gptoss120b | oracle | 46.0% |
| gptoss120b | real | 48.8% |

> Rows where an arm asserts no concept score `nan` and vanish from a naive
> mean. Coverage must be reported or the system looks healthiest where it fails.
