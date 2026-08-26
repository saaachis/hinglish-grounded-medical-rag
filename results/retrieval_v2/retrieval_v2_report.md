# Retrieval v2 -- passage chunking, matched content, hybrid fusion

n = 3015 queries over 10000 cases (41746 passages, 4.17 per case). CPU only.

Every system now reads the SAME full case text: lexical natively, dense via its
passages. The previous Table 1 gave BM25 the full 200 words while LaBSE saw ~85
tokens, so that comparison measured the configuration, not the method.

## Recall@1

| System | Q1 Hinglish | Q2 English |
|---|---:|---:|
| `BM25-full` | 0.0935 | 0.1847 |
| `Hybrid-RRF` | 0.1751 | 0.1973 |
| `LaBSE-passages` | 0.1280 | 0.1486 |
| `TFIDF-full` | 0.0842 | 0.1529 |
| *random floor* | *0.0626* | *0.0626* |

### For reference, the OLD (truncated, unmatched) numbers

| System | Q1 Hinglish |
|---|---:|
| LaBSE @128 tok, ~85 words | 0.1144 |
| LaBSE @256 tok, ~170 words | 0.1310 |
| BM25 @200 words | 0.1343 |

## H04 re-tested per system

| System | Q1 | Q2 | Q2-Q1 | 95% CI | McNemar p |
|---|---:|---:|---:|---|---:|
| `LaBSE-passages` | 0.1280 | 0.1486 | **+0.0206** | [+0.0040, +0.0375] | 0.01683 |
| `BM25-full` | 0.0935 | 0.1847 | **+0.0912** | [+0.0743, +0.1085] | 9.896e-26 |
| `Hybrid-RRF` | 0.1751 | 0.1973 | **+0.0222** | [+0.0036, +0.0408] | 0.01843 |

> H04 is re-tested on EVERY system because the code-mixing penalty is a
> difference between two arms, and fixing truncation moved those arms in opposite
> directions. A penalty that holds across retrieval methods is a property of
> code-mixing; one that appears only under a particular configuration is not.
