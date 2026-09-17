# Retrieval v2 -- passage chunking, matched content, hybrid fusion

n = 3015 query pairs over 10000 cases. CPU only, no API calls.

Every system reads the SAME full case text: lexical natively, dense via passages.

## Metrics by system and query language

Success@k = any relevant case in the top k (a hit rate; earlier files called it recall@k).
nDCG@10 uses the corpus-level ideal DCG. Graded nDCG uses the automatic proxy:
grade 2 = same condition group and >= 1 shared lexicon concept with the gold English
question; grade 1 = same group only. It is not a clinical relevance judgment.

| System | Query | S@1 | S@3 | S@5 | S@10 | MRR@10 | nDCG@10 | graded nDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Hybrid-RRF` | Q1_hinglish | 0.1751 | 0.3396 | 0.4604 | 0.6564 | 0.2987 | 0.1364 | 0.1068 |
| `LaBSE-passages` | Q1_hinglish | 0.1280 | 0.3075 | 0.4299 | 0.6325 | 0.2584 | 0.1204 | 0.0947 |
| `BM25-full` | Q1_hinglish | 0.0935 | 0.2408 | 0.3844 | 0.6116 | 0.2173 | 0.1012 | 0.0715 |
| `TFIDF-full` | Q1_hinglish | 0.0842 | 0.2222 | 0.3357 | 0.5599 | 0.1972 | 0.1018 | 0.0745 |
| `Hybrid-RRF` | Q2_english_question | 0.1973 | 0.4053 | 0.5347 | 0.7393 | 0.3438 | 0.1692 | 0.1439 |
| `LaBSE-passages` | Q2_english_question | 0.1486 | 0.3260 | 0.4604 | 0.6783 | 0.2835 | 0.1257 | 0.0964 |
| `BM25-full` | Q2_english_question | 0.1847 | 0.4036 | 0.5499 | 0.7373 | 0.3385 | 0.1745 | 0.1537 |
| `TFIDF-full` | Q2_english_question | 0.1529 | 0.3187 | 0.4292 | 0.6153 | 0.2729 | 0.1475 | 0.1220 |
| *random floor* | both | *0.0626* | | | | | | |

## Code-mixing penalty (English - Hinglish) at every depth

Success@k: exact McNemar. MRR / nDCG: Wilcoxon signed-rank. CIs: paired bootstrap, 10,000 resamples.

| System | Metric | Hinglish | English | Penalty | 95% CI | p | Test |
|---|---|---:|---:|---:|---|---:|---|
| `Hybrid-RRF` | success@1 | 0.1751 | 0.1973 | **+0.0222** | [+0.0036, +0.0408] | 0.0184 | McNemar (exact) |
| `Hybrid-RRF` | success@3 | 0.3396 | 0.4053 | **+0.0657** | [+0.0428, +0.0889] | 2.07e-08 | McNemar (exact) |
| `Hybrid-RRF` | success@5 | 0.4604 | 0.5347 | **+0.0743** | [+0.0507, +0.0982] | 8.6e-10 | McNemar (exact) |
| `Hybrid-RRF` | success@10 | 0.6564 | 0.7393 | **+0.0829** | [+0.0607, +0.1055] | 5.95e-13 | McNemar (exact) |
| `Hybrid-RRF` | MRR@10 | 0.2987 | 0.3438 | **+0.0451** | [+0.0285, +0.0620] | 2.71e-09 | Wilcoxon signed-rank |
| `Hybrid-RRF` | nDCG@10 | 0.1364 | 0.1692 | **+0.0328** | [+0.0257, +0.0399] | 6.07e-19 | Wilcoxon signed-rank |
| `Hybrid-RRF` | graded_nDCG@10 | 0.1068 | 0.1439 | **+0.0371** | [+0.0309, +0.0433] | 1.95e-31 | Wilcoxon signed-rank |
| `LaBSE-passages` | success@1 | 0.1280 | 0.1486 | **+0.0206** | [+0.0040, +0.0375] | 0.0168 | McNemar (exact) |
| `LaBSE-passages` | success@3 | 0.3075 | 0.3260 | **+0.0186** | [-0.0033, +0.0408] | 0.108 | McNemar (exact) |
| `LaBSE-passages` | success@5 | 0.4299 | 0.4604 | **+0.0305** | [+0.0070, +0.0544] | 0.0132 | McNemar (exact) |
| `LaBSE-passages` | success@10 | 0.6325 | 0.6783 | **+0.0458** | [+0.0232, +0.0683] | 8.75e-05 | McNemar (exact) |
| `LaBSE-passages` | MRR@10 | 0.2584 | 0.2835 | **+0.0251** | [+0.0099, +0.0405] | 0.0016 | Wilcoxon signed-rank |
| `LaBSE-passages` | nDCG@10 | 0.1204 | 0.1257 | **+0.0053** | [-0.0007, +0.0114] | 0.0146 | Wilcoxon signed-rank |
| `LaBSE-passages` | graded_nDCG@10 | 0.0947 | 0.0964 | **+0.0016** | [-0.0034, +0.0067] | 0.203 | Wilcoxon signed-rank |
| `BM25-full` | success@1 | 0.0935 | 0.1847 | **+0.0912** | [+0.0743, +0.1085] | 9.9e-26 | McNemar (exact) |
| `BM25-full` | success@3 | 0.2408 | 0.4036 | **+0.1629** | [+0.1403, +0.1851] | 3.3e-44 | McNemar (exact) |
| `BM25-full` | success@5 | 0.3844 | 0.5499 | **+0.1655** | [+0.1423, +0.1897] | 1.16e-40 | McNemar (exact) |
| `BM25-full` | success@10 | 0.6116 | 0.7373 | **+0.1257** | [+0.1038, +0.1486] | 7.13e-28 | McNemar (exact) |
| `BM25-full` | MRR@10 | 0.2173 | 0.3385 | **+0.1212** | [+0.1054, +0.1368] | 4.4e-54 | Wilcoxon signed-rank |
| `BM25-full` | nDCG@10 | 0.1012 | 0.1745 | **+0.0733** | [+0.0662, +0.0804] | 6.93e-82 | Wilcoxon signed-rank |
| `BM25-full` | graded_nDCG@10 | 0.0715 | 0.1537 | **+0.0822** | [+0.0760, +0.0882] | 7.53e-140 | Wilcoxon signed-rank |
| `TFIDF-full` | success@1 | 0.0842 | 0.1529 | **+0.0687** | [+0.0541, +0.0839] | 1.86e-19 | McNemar (exact) |
| `TFIDF-full` | success@3 | 0.2222 | 0.3187 | **+0.0965** | [+0.0769, +0.1161] | 7.65e-22 | McNemar (exact) |
| `TFIDF-full` | success@5 | 0.3357 | 0.4292 | **+0.0935** | [+0.0716, +0.1151] | 4.09e-17 | McNemar (exact) |
| `TFIDF-full` | success@10 | 0.5599 | 0.6153 | **+0.0554** | [+0.0318, +0.0779] | 2.2e-06 | McNemar (exact) |
| `TFIDF-full` | MRR@10 | 0.1972 | 0.2729 | **+0.0756** | [+0.0620, +0.0897] | 3.74e-25 | Wilcoxon signed-rank |
| `TFIDF-full` | nDCG@10 | 0.1018 | 0.1475 | **+0.0457** | [+0.0392, +0.0522] | 1.99e-34 | Wilcoxon signed-rank |
| `TFIDF-full` | graded_nDCG@10 | 0.0745 | 0.1220 | **+0.0475** | [+0.0419, +0.0532] | 3.36e-57 | Wilcoxon signed-rank |

Penalty tests significant at p < 0.05: **26 of 28**; penalty positive (English better): **28 of 28**.

## Relevance diagnostics

- Relevant cases per query (same condition group): median 597, min 102, max 731.
- Queries whose gold English question contains >= 1 lexicon concept: 90.6% (the rest can only reach grade 1).
- Grade-2 cases per query: median 303.
