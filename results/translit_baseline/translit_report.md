# Transliteration-aware retrieval baseline

n = 3015 queries. CPU only, no API calls.

**Transliteration source:** hand-verified lexicon (87% of Hindi tokens) + ITRANS rules for the remainder; English tokens left in Latin script (25% of tokens)

The paper reports MuRIL at the random floor on romanised Hinglish and attributes
the failure to script mismatch. This tests that claim directly by giving MuRIL the
same queries in its own script.

## Example

| Query form | Text |
|---|---|
| romanised | Meri 11 saal ki beti ko 13 din pehle MRSA+ diagnose kiya gaya tha. Wo 5 din se high fever hai jisko hum Advil se kam kar rahe hain kyun ki wo bhi unwe |
| devanagari | मेरी 11 साल की बेटी को 13 दिन पहले MRSA+ diagnose किया गया था. वो 5 दिन से high fever है जिसको हम Advil से कम कर रहे हैं क्यों की वो भी unwell feel कर |
| english | What could be the cause of high fever, swollen tonsils with white spots, severe fatigue, and cold sweats with low body temperature in an 11-year-old g |

## Retrieval quality

| System | Query form | S@1 | S@5 | S@10 | MRR@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| `MuRIL` | romanised | 0.0640 | 0.2985 | 0.5174 | 0.1704 | 0.0705 |
| `MuRIL` | devanagari | 0.0716 | 0.3274 | 0.5396 | 0.1852 | 0.0782 |
| `MuRIL` | english | 0.1821 | 0.5108 | 0.7111 | 0.3224 | 0.1496 |
| *random floor (depth-adjusted)* | — | *0.0626* | *0.2755* | *0.4739* | | |

The floor rises with k: with ~600 relevant cases in 10,000 a random ranker hits one
47.4% of the time by rank 10. MuRIL on romanised input tracks that floor at
every depth, which is what "at chance" means here.

## Paired tests (Success@1: exact McNemar; MRR/nDCG: Wilcoxon; CIs bootstrap)

| System | Comparison | Metric | Delta | 95% CI | p |
|---|---|---|---:|---|---:|
| `MuRIL` | devanagari - romanised | success@1 | **+0.0076** | [-0.0046, +0.0196] | 0.242 |
| `MuRIL` | devanagari - romanised | MRR@10 | **+0.0148** | [+0.0030, +0.0264] | 0.00961 |
| `MuRIL` | devanagari - romanised | nDCG@10 | **+0.0077** | [+0.0039, +0.0115] | 0.000821 |
| `MuRIL` | english - devanagari | success@1 | **+0.1104** | [+0.0942, +0.1267] | 1.67e-38 |
| `MuRIL` | english - devanagari | MRR@10 | **+0.1372** | [+0.1218, +0.1525] | 6.93e-67 |
| `MuRIL` | english - devanagari | nDCG@10 | **+0.0714** | [+0.0653, +0.0776] | 2.6e-101 |
| `MuRIL` | english - romanised | success@1 | **+0.1181** | [+0.1018, +0.1340] | 1.42e-47 |
| `MuRIL` | english - romanised | MRR@10 | **+0.1520** | [+0.1369, +0.1673] | 7.82e-83 |
| `MuRIL` | english - romanised | nDCG@10 | **+0.0791** | [+0.0731, +0.0852] | 1.43e-124 |

## Reading

- Romanised 0.0640 (floor 0.0626) -> Devanagari 0.0716 -> English 0.1821.
- Transliteration recovers **6.5%** of the romanised-to-English gap (McNemar p = 0.242).

- **Sanity check:** MuRIL on English scores 0.1821 here against 0.1821 in `h4_baselines.csv` (drift 0.0000).

- On nDCG@10 the same comparison moves +0.0077 (p = 0.000821).

**Script mismatch is real but secondary.** Transliteration gives a small, reliable
improvement (nDCG@10 +0.0077, p = 0.000821) and recovers only
**6.5%** of the romanised-to-English gap; the Success@1 change is not
significant on its own (p = 0.242). Writing the query in MuRIL's script
therefore does not restore its English performance.

Two readings remain open, and the paper should give both:
1. Script is not the binding constraint -- the encoder's Hindi representations are
   themselves weaker than its English ones on this clinical content.
2. The transliteration is imperfect (a hand-verified lexicon plus rules, not a
   trained transliterator), so this is a LOWER bound on what script conversion
   could achieve.

Either way the practical advice is unchanged and now tested rather than asserted:
transliterating romanised input into an Indic encoder's script is not a substitute
for a cross-lingual encoder.
