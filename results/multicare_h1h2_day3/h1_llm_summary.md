# H1 Results: Real LLM Generation (Groq)

## Setup
- Generator: **Llama-3.1-8B-Instant** via Groq API (pairs 1-70: Llama-3.3-70B)
- Evidence: **MultiCaRe** raw clinical cases (no LLM extraction)
- Queries: **MMCQSD** Hinglish patient queries
- Matching: **LaBSE** + FAISS
- Evaluated pairs: **356** (proportionally sampled across 18 conditions)

## Key Results

| Metric | Zero-Shot | Grounded | Delta |
|---|---:|---:|---:|
| **Factual support** | 0.3246 | 0.5485 | **+0.2239** |
| **Hallucination score** | 0.4900 | 0.2893 | **+0.2007** |

## Statistical Significance (H1)
- Test: **wilcoxon_signed_rank**
- Statistic: **3528.5000**
- p-value: **9.51e-20**
- Effect size (Cohen's d): **Medium (0.555)**
- 95% CI for factual gain: **[0.1818, 0.2660]**
- Verdict: **HIGHLY significant (p < 0.001)**

## Per-Condition Results

| Condition | N | Zero Factual | Grounded Factual | Gain | Zero Halluc | Grounded Halluc |
|---|---:|---:|---:|---:|---:|---:|
| cyanosis | 1 | 0.000 | 0.667 | +0.667 | 1.000 | 0.333 |
| dry_scalp | 5 | 0.300 | 0.450 | +0.150 | 0.550 | 0.100 |
| edema | 14 | 0.464 | 0.619 | +0.155 | 0.214 | 0.060 |
| eye_inflammation | 2 | 0.250 | 0.500 | +0.250 | 0.000 | 0.125 |
| eye_redness | 13 | 0.410 | 0.506 | +0.096 | 0.474 | 0.090 |
| foot_swelling | 23 | 0.353 | 0.569 | +0.216 | 0.354 | 0.203 |
| hand_lump | 25 | 0.391 | 0.550 | +0.159 | 0.279 | 0.240 |
| itchy_eyelid | 1 | 0.000 | 0.000 | +0.000 | 1.000 | 1.000 |
| knee_swelling | 11 | 0.591 | 0.518 | -0.073 | 0.409 | 0.141 |
| lip_swelling | 23 | 0.297 | 0.667 | +0.370 | 0.507 | 0.138 |
| mouth_ulcers | 24 | 0.286 | 0.587 | +0.301 | 0.464 | 0.194 |
| neck_swelling | 48 | 0.359 | 0.561 | +0.202 | 0.406 | 0.236 |
| skin_dryness | 7 | 0.119 | 0.500 | +0.381 | 0.667 | 0.071 |
| skin_growth | 9 | 0.333 | 0.685 | +0.352 | 0.333 | 0.148 |
| skin_irritation | 8 | 0.573 | 0.698 | +0.125 | 0.240 | 0.208 |
| skin_rash | 136 | 0.293 | 0.454 | +0.161 | 0.603 | 0.369 |
| swollen_eye | 18 | 0.375 | 0.523 | +0.148 | 0.333 | 0.102 |
| swollen_tonsils | 31 | 0.194 | 0.372 | +0.179 | 0.661 | 0.386 |

## Sample Outputs

### skin_rash (sim=0.514)
**Query**: Mere haath ki kichad jaisi khujli arm ke elbow area me shuru hui, do din baad wo aur bhi zyada bura ho gaya aur dusre arm par bhi tha lekin itni buri ...
**Zero-shot**: Arre bhai, main samajh gaya. Ek photo allergic reaction bahut hi common hai, lekin yeh kuch aur bhi ho sakta hai. Aapke symptoms bahut hi widespread hain, jaise ki gardan, peeth, pet, pairon, haath ki...
**Grounded**: Arre beta, maine dekha hai ki aapke haath ki khujli aur kichad jaisi cheez ek photo allergic reaction hai. Iska matlab hai ki aapke skin ne suraj ki wajah se reaction di hai. Is reaction ne aapke face...
- Factual: 0.00 → 0.00 | Halluc: 1.00 → 1.00

### edema (sim=0.487)
**Query**: Hi doctor,
Lagbhag 3 hafte pehle, mere dermatologist ne mere maathe par ek anokhe breakout ke liye Retin-A 0.5% aur Acticlate prescribe kiya tha. Anae...
**Zero-shot**: Arre bhai, main samajh gaya hai ki tumhara chehra bahut kharab lag raha hai. Chemical peel karne ke baad skin jalan aur kharab hone ka sawal hai, lekin dard nahi hona achha hai. Ye peel karne ke baad ...
**Grounded**: Arre bhai, maine dekha hai ki tumhare chehre par chemical peel kiya gaya hai. Iske baad aapko thodi si laalima aayi hai, aur aapko pata hai ki aapka chehra 2nd to 3rd-degree jaise dikhta hai. Lekin, a...
- Factual: 0.25 → 0.25 | Halluc: 0.00 → 0.00

### skin_growth (sim=0.498)
**Query**: Hi doctor, mujhe ek halka, komal gaanth ya phir ek tarah ka jatka ya fir ek tarah ka nod lagbhag skin ke andar, deltoid muscle par hai. Yeh bahut mush...
**Zero-shot**: Beta, main tumhe thoda relax karne ka prayaas karta hoon. Deltoid muscle par ek nod ya jatka hone ka matlab yeh nahi hai ki tumhara sarcoma hai. Sarcoma ke liye humein kai symptoms aur factors dekhne ...
**Grounded**: Bhai, main aapko yeh batana chahta hoon ki aapki jaankari ke anusaar, aapke paas ek tumor hai jo LGMFS (Low-Grade Myxofibrosarcoma) ka naam hai. Yeh ek prakaar ka sarcoma hai jo aapke first metatarsal...
- Factual: 0.25 → 1.00 | Halluc: 0.00 → 0.00

---
*Generated using Llama-3.3-70B via Groq API on 356 sampled pairs*
