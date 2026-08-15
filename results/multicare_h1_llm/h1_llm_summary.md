# H1 Results: Real LLM Generation (Groq Llama-3.3-70B)

## Setup
- Generator: **Llama-3.3-70B-Versatile** via Groq API
- Evidence: **MultiCaRe** raw clinical cases (no LLM extraction)
- Queries: **MMCQSD** Hinglish patient queries
- Matching: **LaBSE** + FAISS
- Evaluated pairs: **73** (proportionally sampled across 18 conditions)

## Key Results

| Metric | Zero-Shot | Grounded | Delta |
|---|---:|---:|---:|
| **Factual support** | 0.3148 | 0.5559 | **+0.2411** |
| **Hallucination score** | 0.6235 | 0.3311 | **+0.2925** |

## Statistical Significance (H1)
- Test: **wilcoxon_signed_rank**
- Statistic: **141.5000**
- p-value: **4.41e-06**
- Effect size (Cohen's d): **Medium (0.605)**
- 95% CI for factual gain: **[0.1481, 0.3341]**
- Verdict: **HIGHLY significant (p < 0.001)**

## Per-Condition Results

| Condition | N | Zero Factual | Grounded Factual | Gain | Zero Halluc | Grounded Halluc |
|---|---:|---:|---:|---:|---:|---:|
| cyanosis | 3 | 0.250 | 0.250 | +0.000 | 0.000 | 0.000 |
| dry_scalp | 4 | 0.250 | 0.250 | +0.000 | 0.000 | 0.000 |
| edema | 11 | 0.273 | 0.250 | -0.023 | 0.045 | 0.000 |
| eye_inflammation | 3 | 0.250 | 0.250 | +0.000 | 0.000 | 0.000 |
| eye_redness | 9 | 0.306 | 0.250 | -0.056 | 0.111 | 0.000 |
| foot_swelling | 17 | 0.304 | 0.250 | -0.054 | 0.078 | 0.000 |
| hand_lump | 16 | 0.297 | 0.250 | -0.047 | 0.094 | 0.000 |
| itchy_eyelid | 3 | 0.250 | 0.250 | +0.000 | 0.000 | 0.000 |
| knee_swelling | 11 | 0.341 | 0.250 | -0.091 | 0.045 | 0.000 |
| lip_swelling | 19 | 0.211 | 0.250 | +0.039 | 0.158 | 0.000 |
| mouth_ulcers | 19 | 0.250 | 0.250 | +0.000 | 0.158 | 0.000 |
| neck_swelling | 27 | 0.244 | 0.250 | +0.006 | 0.062 | 0.000 |
| skin_dryness | 6 | 0.256 | 0.250 | -0.006 | 0.119 | 0.000 |
| skin_growth | 8 | 0.250 | 0.250 | +0.000 | 0.000 | 0.000 |
| skin_irritation | 7 | 0.298 | 0.250 | -0.048 | 0.167 | 0.000 |
| skin_rash | 104 | 0.308 | 0.465 | +0.157 | 0.476 | 0.232 |
| swollen_eye | 15 | 0.367 | 0.250 | -0.117 | 0.033 | 0.000 |
| swollen_tonsils | 17 | 0.235 | 0.250 | +0.015 | 0.147 | 0.000 |

## Sample Outputs

### skin_rash (sim=0.474)
**Query**: Meri beti ko ek rash hai jo theek nahi ho raha hai. Hydrocortisone cream help nahi karti. Pehle humne socha ki ye soap se contact dermatitis hai, leki...
**Zero-shot**: Aapki beti ki rash ke liye pehle to aapko doctor ki salah leni chahiye, unhone kuchh karan socha hoga ki Pancreatic Cancer ke liye blood test karwaya jaye. Lekin aapko tension lena chahiye ya nahi, ye...
**Grounded**: Aapki beti ki rash ke liye hum koi specific karan nahi bata sakte kyunki humare paas uske bare mein koi information nahi hai. Lekin hum yeh keh sakte hain ki scabies ek aam parasitic infection hai jo ...
- Factual: 0.50 → 0.75 | Halluc: 0.50 → 0.25

### skin_rash (sim=0.589)
**Query**: Meri lower legs par horrible red raised flaky itchy rash shuru hui, jisse hive like rash on my thighs par progress hua, phir hives ne upper arms, back...
**Zero-shot**: Aapko 3 mahine se ye rash hai, aur wo hive like rash thighs se lekar upper arms, back, aur forearms tak spread ho gaya hai, aur aapko khujli hoti hai. Aapka ANA test negative aaya hai, isliye lupus ka...
**Grounded**: Aapki symptoms ko dekhkar, mujhe lagta hai ki aapki bimari chikungunya fever ki wajah se ho sakti hai, jaise ki clinical evidence mein bataya gaya hai. Lekin, aapke symptoms mein kuchh alag baat hai, ...
- Factual: 1.00 → 1.00 | Halluc: 0.00 → 0.00

### skin_rash (sim=0.455)
**Query**:  Meri hatheli par ek daana hai jo exczema ki tarah lag raha hai, mujhe iss daane ko ek mahine ya do mahine se hai. Kya ye HIV rash ho sakta hai? Maine...
**Zero-shot**: Aapka daana exczema ki tarah lagta hai, lekin HIV rash bhi ek sambhavana hai. Lekin sirf ek daana hone se HIV rash ka pata nahin chalta, iske liye doctor se milna zaroori hai. Aapne jo creams aur hydr...
**Grounded**: Aapki skin par jo daana hai, uske baare mein hum confirm nahi kar sakte ki ye HIV rash hai ya nahi, kyunki clinical evidence mein iske baare mein koi jankari nahi di gayi hai. Aapke symptoms, jaise ki...
- Factual: 0.00 → 0.00 | Halluc: 1.00 → 1.00

---
*Generated using Llama-3.3-70B via Groq API on 73 sampled pairs*
