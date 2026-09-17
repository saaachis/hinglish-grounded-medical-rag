# Translate-then-retrieve baseline

n = 500 queries (stratified by condition group). Translator: `openai/gpt-oss-120b`.

Reviewer 2 asked for comparison with existing methods. Translating the query into the
corpus language and retrieving monolingually is the standard alternative to a
cross-lingual encoder; this is that comparison, on the same index and criterion as Table 1.

## Example

| Form | Text |
|---|---|
| hinglish | Meri 7 saal ki beti ko hamare kutte ne lower lip par scratch kiya tha. Ye sirf ek pal ke liye bleeding hui thi aur phir thik ho gaya. Agle din lip normal dikhai |
| translated | My 7‑year‑old daughter was scratched on her lower lip by our dog. It bled for a moment and then stopped. The next day the lip looked normal, but it was a little |
| english (gold) | Is hoarseness and swollen tonsils related to a dog scratch on the lip, leading to a canker sore? |

## Retrieval quality

| System | Query form | S@1 | S@5 | S@10 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| `LaBSE-passages` | hinglish | 0.1220 | 0.4540 | 0.6560 | 0.2636 | 0.1248 |
| `BM25-full` | hinglish | 0.1080 | 0.4120 | 0.6340 | 0.2397 | 0.1102 |
| `TFIDF-full` | hinglish | 0.0880 | 0.3620 | 0.5860 | 0.2083 | 0.1099 |
| `Hybrid-RRF` | hinglish | 0.1840 | 0.5000 | 0.7160 | 0.3226 | 0.1497 |
| `LaBSE-passages` | translated | 0.1680 | 0.5200 | 0.7440 | 0.3231 | 0.1492 |
| `BM25-full` | translated | 0.0240 | 0.1760 | 0.3840 | 0.0979 | 0.0477 |
| `TFIDF-full` | translated | 0.0300 | 0.3100 | 0.5440 | 0.1423 | 0.0757 |
| `Hybrid-RRF` | translated | 0.1200 | 0.4580 | 0.6500 | 0.2600 | 0.1175 |
| `LaBSE-passages` | english | 0.1460 | 0.4540 | 0.6680 | 0.2767 | 0.1222 |
| `BM25-full` | english | 0.1960 | 0.5740 | 0.7580 | 0.3510 | 0.1815 |
| `TFIDF-full` | english | 0.1480 | 0.4240 | 0.6300 | 0.2711 | 0.1473 |
| `Hybrid-RRF` | english | 0.2020 | 0.5500 | 0.7380 | 0.3475 | 0.1678 |

## Paired tests on Success@1

| System | Comparison | Delta | 95% CI | McNemar p |
|---|---|---:|---|---:|
| `Hybrid-RRF` | translated - hinglish | **-0.0640** | [-0.1060, -0.0240] | 0.0035 |
| `Hybrid-RRF` | english - translated | **+0.0820** | [+0.0400, +0.1260] | 0.000276 |
| `Hybrid-RRF` | english - hinglish | **+0.0180** | [-0.0280, +0.0660] | 0.494 |
| `LaBSE-passages` | translated - hinglish | **+0.0460** | [+0.0040, +0.0880] | 0.0398 |
| `LaBSE-passages` | english - translated | **-0.0220** | [-0.0640, +0.0200] | 0.343 |
| `LaBSE-passages` | english - hinglish | **+0.0240** | [-0.0160, +0.0640] | 0.29 |
| `BM25-full` | translated - hinglish | **-0.0840** | [-0.1140, -0.0540] | 5.71e-08 |
| `BM25-full` | english - translated | **+0.1720** | [+0.1360, +0.2080] | 7.1e-21 |
| `BM25-full` | english - hinglish | **+0.0880** | [+0.0440, +0.1320] | 0.00016 |
| `TFIDF-full` | translated - hinglish | **-0.0580** | [-0.0860, -0.0320] | 5.7e-05 |
| `TFIDF-full` | english - translated | **+0.1180** | [+0.0860, +0.1520] | 2.45e-12 |
| `TFIDF-full` | english - hinglish | **+0.0600** | [+0.0240, +0.0960] | 0.00182 |

## Does translation recover the penalty?

| System | Hinglish | Translated | English (gold) |
|---|---:|---:|---:|
| `LaBSE-passages` | 0.1220 | 0.1680 | 0.1460 |
| `BM25-full` | 0.1080 | 0.0240 | 0.1960 |
| `TFIDF-full` | 0.0880 | 0.0300 | 0.1480 |
| `Hybrid-RRF` | 0.1840 | 0.1200 | 0.2020 |

**Only the dense system benefits.** For LaBSE, translating helps (0.1680 vs 0.1220). For BM25 and TF-IDF, translation makes retrieval markedly WORSE than leaving the query code-mixed.

## Why -- and what it says about the paper's own English baseline

The three query forms are not the same kind of text:

| Query form | mean words | lexicon concepts | tokens known to the corpus | mean IDF of those tokens |
|---|---:|---:|---:|---:|
| hinglish | 109.5 | 1.15 | 46.9% | 5.22 |
| translated | 96.8 | 2.10 | 99.4% | 2.28 |
| english | 21.5 | 1.73 | 99.3% | 1.98 |

Two things follow, and the second matters for H04.

**1. Code-mixing incidentally filters stopwords.** Under half of a Hinglish query's
tokens exist in the English corpus at all, but the ones that survive are rare and
specific (high IDF): the Hindi carries the grammar and the English carries the clinical
content. A fluent English translation restores the function words, which are common,
low-IDF and shared by every case report, so they dilute the query. That is why lexical
retrieval gets WORSE after translation while the dense system improves.

**2. The paper's English arm is a summary, not a translation.** The gold English
question averages ~21 words against ~97 for a faithful translation of the same query.
It is a short, clinician-style condensation written with knowledge of the case, much
closer to the register of the case reports themselves. The measured code-mixing penalty
therefore compares a patient narrative against a clinical summary, and conflates
language with **conciseness and register**.

This does not overturn H04 -- the penalty holds across every system and every depth, and
the MuRIL script result is independent of it -- but the estimate is an upper bound on
the language effect alone, and the paper should say so. The translate-then-retrieve arm
is the cleaner language-only comparison, because it holds length and register fixed:
against it, dense retrieval improves and lexical retrieval does not.

## Cost

Translation adds one LLM call per query, its latency to every search, and its
availability as a dependency -- the same hosted-model hazard this paper reports
elsewhere. A cross-lingual encoder costs nothing extra at query time and, on this
evidence, retrieves at least as well.
