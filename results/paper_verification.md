# Paper numbers vs committed results

36 claims checked. **33 verified, 2 mismatched, 1 not found.**

A number that is not VERIFIED must not appear in the camera-ready until it is either
recomputed or corrected. Tolerance is 5e-4 unless stated.

| Status | Section | Claim in the paper | Claimed | Recomputed | Source |
|---|---|---|---:|---:|---|
| **MISMATCH** | Table 3 | gpt-oss-120b oracle delta +0.2659 | 0.2659 | 0.2608 | `contrasts.csv` |
| **MISMATCH** | Table 3 | gpt-oss-120b real retrieval delta +0.2238 | 0.2238 | 0.0704 | `contrasts.csv` |
| **NOT FOUND** | Sect. 5.3 | oracle beats real retrieval by +0.0875 (p = 0.041) | 0.0875 | — | `NOT FOUND in any committed file` |
| ok | Sect. 5.3 | llama oracle F1, circular +0.203 | 0.2030 | 0.2029 | `contrasts.csv` |
| ok | Sect. 5.3 | llama oracle F1, unbiased +0.062 | 0.0620 | 0.0620 | `contrasts.csv` |
| ok | Sect. 5.3 | gpt-oss-120b oracle F1, circular +0.093 | 0.0930 | 0.0926 | `contrasts.csv` |
| ok | Sect. 5.3 | gpt-oss-120b oracle F1, unbiased -0.021 | -0.0210 | -0.0207 | `contrasts.csv` |
| ok | Sect. 5.3 | gpt-oss-120b real F1, circular -0.032 | -0.0320 | -0.0320 | `contrasts.csv` |
| ok | Sect. 5.3 | gpt-oss-120b real F1, unbiased -0.047 | -0.0470 | -0.0469 | `contrasts.csv` |
| ok | Table 1 | Hybrid (RRF) Hinglish 0.1751 | 0.1751 | 0.1751 | `retrieval_v2_metrics.csv` |
| ok | Table 4 | zero-shot factual support rho = -0.116 | -0.1160 | -0.1155 | `h2_corrected_cmi_stats.csv` |
| ok | Table 4 | grounded hallucination rho = -0.022 | -0.0220 | -0.0224 | `h2_corrected_cmi_stats.csv` |
| ok | Table 4 | zero-shot hallucination rho = +0.042 | 0.0420 | 0.0420 | `h2_corrected_cmi_stats.csv` |
| ok | Table 5 | MultiCaRe refusal 88.1% | 0.8810 | 0.8812 | `h3_summary.csv` |
| ok | Table 5 | MMedBench refusal 76.2% | 0.7620 | 0.7625 | `h3_summary.csv` |
| ok | Table 4 | grounded factual support rho = -0.001 | -0.0010 | -0.0006 | `h2_corrected_cmi_stats.csv` |
| ok | Sect. 5.3 | grounding reduces concept recall by 0.138 | -0.1380 | -0.1384 | `contrasts.csv` |
| ok | Table 3 | llama oracle, repaired lexicon, d = 0.720 | 0.7200 | 0.7202 | `contrasts.csv` |
| ok | Table 3 | llama oracle, repaired lexicon, delta +0.2962 | 0.2962 | 0.2962 | `contrasts.csv` |
| ok | Table 1 | Hybrid (RRF) English 0.1973 | 0.1973 | 0.1973 | `retrieval_v2_metrics.csv` |
| ok | Table 1 | LaBSE Hinglish 0.1280 | 0.1280 | 0.1280 | `retrieval_v2_metrics.csv` |
| ok | Table 1 | LaBSE English 0.1486 | 0.1486 | 0.1486 | `retrieval_v2_metrics.csv` |
| ok | Table 1 | BM25 Hinglish 0.0935 | 0.0935 | 0.0935 | `retrieval_v2_metrics.csv` |
| ok | Table 1 | BM25 English 0.1847 | 0.1847 | 0.1847 | `retrieval_v2_metrics.csv` |
| ok | Table 1 | TF-IDF Hinglish 0.0842 | 0.0842 | 0.0842 | `retrieval_v2_metrics.csv` |
| ok | Table 1 | TF-IDF English 0.1529 | 0.1529 | 0.1529 | `retrieval_v2_metrics.csv` |
| ok | Table 1 | Hybrid penalty +0.0222 | 0.0222 | 0.0222 | `h4_v2_tests_by_depth.csv` |
| ok | Table 1 | LaBSE penalty +0.0206 | 0.0206 | 0.0206 | `h4_v2_tests_by_depth.csv` |
| ok | Table 1 | BM25 penalty +0.0912 | 0.0912 | 0.0912 | `h4_v2_tests_by_depth.csv` |
| ok | Sect. 5.1 | BM25/dense penalty ratio ~4.4x | 4.4000 | 4.4355 | `derived from h4_v2_tests_by_depth.csv` |
| ok | Sect. 5.1 | MuRIL Hinglish 0.0640 (chance) | 0.0640 | 0.0640 | `h4_baselines.csv` |
| ok | Sect. 5.1 | MuRIL English 0.1821 | 0.1821 | 0.1821 | `h4_baselines.csv` |
| ok | Table 2 | constant 'swelling' 0.7132 | 0.7132 | 0.7132 | `degenerate_baselines.csv` |
| ok | Table 2 | verbatim copy 1.0000 | 1.0000 | 1.0000 | `degenerate_baselines.csv` |
| ok | Table 5 | MultiCaRe mean F1 0.3265 | 0.3265 | 0.3265 | `h3_summary.csv` |
| ok | Table 5 | PubMedQA mean F1 0.3476 | 0.3476 | 0.3476 | `h3_summary.csv` |

## Every claim that did not verify

### Table 3 — gpt-oss-120b oracle delta +0.2659
- paper says **0.2659**
- recomputed: **0.2608** (from `contrasts.csv`)
- Paper also states n = 223, d = 0.662. contrasts.csv has n = 214, d = 0.640; h1_real_120b reports +0.2849 (n = 124). No committed file matches the paper.

### Table 3 — gpt-oss-120b real retrieval delta +0.2238
- paper says **0.2238**
- recomputed: **0.0704** (from `contrasts.csv`)
- Paper states n = 237, d = 0.616. contrasts.csv has +0.0704 (n = 221); h1_real_120b reports +0.2347 (n = 131). No committed file matches the paper.

### Sect. 5.3 — oracle beats real retrieval by +0.0875 (p = 0.041)
- paper says **0.0875**
- recomputed: **not found** (from `NOT FOUND in any committed file`)
- CONTRADICTED. h1_real_120b: +0.0748, n = 76, p = 0.226. h1_real_retrieval: +0.0755, n = 125, p = 0.106. Both NON-significant. The paper reports this as 'small but significant'. Must be corrected.
