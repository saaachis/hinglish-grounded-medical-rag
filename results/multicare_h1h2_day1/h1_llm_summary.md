# H1 Results: Real LLM Generation (Groq)

## Setup
- Generator: **Llama-3.1-8B-Instant** via Groq API (pairs 1-70: Llama-3.3-70B)
- Evidence: **MultiCaRe** raw clinical cases (no LLM extraction)
- Queries: **MMCQSD** Hinglish patient queries
- Matching: **LaBSE** + FAISS
- Evaluated pairs: **374** (proportionally sampled across 18 conditions)

## Key Results

| Metric | Zero-Shot | Grounded | Delta |
|---|---:|---:|---:|
| **Factual support** | 0.3138 | 0.5475 | **+0.2337** |
| **Hallucination score** | 0.4977 | 0.2860 | **+0.2117** |

## Statistical Significance (H1)
- Test: **wilcoxon_signed_rank**
- Statistic: **4589.5000**
- p-value: **3.60e-20**
- Effect size (Cohen's d): **Medium (0.553)**
- 95% CI for factual gain: **[0.1908, 0.2767]**
- Verdict: **HIGHLY significant (p < 0.001)**

## Per-Condition Results

| Condition | N | Zero Factual | Grounded Factual | Gain | Zero Halluc | Grounded Halluc |
|---|---:|---:|---:|---:|---:|---:|
| cyanosis | 3 | 0.000 | 0.167 | +0.167 | 1.000 | 0.333 |
| dry_scalp | 6 | 0.347 | 0.458 | +0.111 | 0.278 | 0.417 |
| edema | 15 | 0.444 | 0.600 | +0.156 | 0.156 | 0.100 |
| eye_inflammation | 4 | 0.125 | 0.562 | +0.438 | 0.500 | 0.250 |
| eye_redness | 13 | 0.212 | 0.609 | +0.397 | 0.385 | 0.218 |
| foot_swelling | 20 | 0.321 | 0.296 | -0.025 | 0.492 | 0.442 |
| hand_lump | 27 | 0.281 | 0.635 | +0.354 | 0.302 | 0.143 |
| itchy_eyelid | 1 | 1.000 | 1.000 | +0.000 | 0.000 | 0.000 |
| knee_swelling | 15 | 0.450 | 0.528 | +0.078 | 0.300 | 0.122 |
| lip_swelling | 28 | 0.232 | 0.542 | +0.310 | 0.634 | 0.190 |
| mouth_ulcers | 26 | 0.311 | 0.522 | +0.212 | 0.487 | 0.218 |
| neck_swelling | 33 | 0.311 | 0.503 | +0.192 | 0.439 | 0.316 |
| skin_dryness | 12 | 0.312 | 0.250 | -0.062 | 0.375 | 0.250 |
| skin_growth | 8 | 0.369 | 0.500 | +0.131 | 0.350 | 0.312 |
| skin_irritation | 7 | 0.357 | 0.560 | +0.202 | 0.429 | 0.333 |
| skin_rash | 139 | 0.313 | 0.551 | +0.238 | 0.595 | 0.309 |
| swollen_eye | 15 | 0.306 | 0.528 | +0.222 | 0.544 | 0.272 |
| swollen_tonsils | 27 | 0.361 | 0.590 | +0.229 | 0.500 | 0.271 |

## Sample Outputs

### skin_rash (sim=0.384)
**Query**: Hi, Hamare kashtkar Canadian winters mein, mere pichwade par kuch samasya ho gayi hai, jyadatar, kuch pait ke hisse aur kuch jaanghon par. Raat ko son...
**Zero-shot**: Arre bhai, aapka description sunke lagta hai ki aapko ek chhote chhote funsiyon ke saath ek problem hai, jise "chilblain" kaha jata hai. Ye ek common problem hai Canadian winters mein, jahan aapko gar...
**Grounded**: Bhai, aapke symptoms aur tasveer ke anusaar, maine dekha hai ki aapke pichwade par chote chote funsi jaise daane hain jin par ek chota sa chhala hota hai. Ye symptoms hydrocele ke liye hi hain, jo ek ...
- Factual: 0.25 → 0.25 | Halluc: 0.00 → 0.00

### skin_rash (sim=0.496)
**Query**: Mere left small toe par dry skin red pimply rash shuru hui hai aur foot ke top par phail rahi hai. Shuru me mujhe athletes foot lag raha tha par spray...
**Zero-shot**: Arre bhai, aapka rash kuch chhote chhote pimple jaise dikh raha hai, aur yeh dry skin ke saath ho raha hai. Yeh possible hai ki yeh athlete's foot ke symptoms hain, lekin spray kaam nahi kar raha tha,...
**Grounded**: Bachchi, aapke symptoms ke liye main aapko kuch sujhav de sakta hoon. Aapka small toe par dry skin, red pimply rash, aur foot ke top par phail rahi hai, yeh symptoms athlete's foot ke liye lagbhag mat...
- Factual: 0.00 → 0.50 | Halluc: 1.00 → 0.50

### edema (sim=0.534)
**Query**: Mujhe lipo ki problem hai aur affected area ke neeche bahut saare nodules develop ho gaye hain. Par aaj meri therapist ne meri ribs par moveable chhot...
**Zero-shot**: Arre bhai, main samajh gaya. Aapka anxiety to hai, lekin main aapko thoda relax karne ka prayaas karta hoon. Jab aapka therapist ne ribs par ek moveable lump notice kiya hai, to yeh kuchh chizein soch...
**Grounded**: Arre bhai, maine aapki problem dekhi hai. Aapke paas ek cyst hai, jo ganglion cyst kehte hain. Yeh cyst fibula ke neeche develop hota hai. 

Ganglion cysts fibula ke neeche develop karne ke liye ek co...
- Factual: 0.33 → 1.00 | Halluc: 0.67 → 0.00

---
*Generated using Llama-3.3-70B via Groq API on 374 sampled pairs*
