# H1 Results: Real LLM Generation (Groq)

## Setup
- Generator: **Llama-3.1-8B-Instant** via Groq API (pairs 1-70: Llama-3.3-70B)
- Evidence: **MultiCaRe** raw clinical cases (no LLM extraction)
- Queries: **MMCQSD** Hinglish patient queries
- Matching: **LaBSE** + FAISS
- Evaluated pairs: **362** (proportionally sampled across 18 conditions)

## Key Results

| Metric | Zero-Shot | Grounded | Delta |
|---|---:|---:|---:|
| **Factual support** | 0.3197 | 0.5642 | **+0.2445** |
| **Hallucination score** | 0.4876 | 0.2555 | **+0.2321** |

## Statistical Significance (H1)
- Test: **wilcoxon_signed_rank**
- Statistic: **3411.0000**
- p-value: **2.28e-23**
- Effect size (Cohen's d): **Medium (0.615)**
- 95% CI for factual gain: **[0.2034, 0.2856]**
- Verdict: **HIGHLY significant (p < 0.001)**

## Per-Condition Results

| Condition | N | Zero Factual | Grounded Factual | Gain | Zero Halluc | Grounded Halluc |
|---|---:|---:|---:|---:|---:|---:|
| dry_scalp | 4 | 0.312 | 0.542 | +0.229 | 0.500 | 0.083 |
| edema | 21 | 0.579 | 0.598 | +0.018 | 0.063 | 0.081 |
| eye_inflammation | 4 | 0.250 | 0.458 | +0.208 | 0.375 | 0.167 |
| eye_redness | 15 | 0.367 | 0.644 | +0.278 | 0.533 | 0.156 |
| foot_swelling | 13 | 0.256 | 0.462 | +0.205 | 0.513 | 0.077 |
| hand_lump | 14 | 0.381 | 0.613 | +0.232 | 0.298 | 0.226 |
| itchy_eyelid | 1 | 0.000 | 0.250 | +0.250 | 1.000 | 0.000 |
| knee_swelling | 19 | 0.333 | 0.605 | +0.272 | 0.430 | 0.118 |
| lip_swelling | 22 | 0.220 | 0.530 | +0.311 | 0.576 | 0.129 |
| mouth_ulcers | 29 | 0.259 | 0.433 | +0.174 | 0.483 | 0.334 |
| neck_swelling | 35 | 0.321 | 0.592 | +0.270 | 0.357 | 0.237 |
| skin_dryness | 10 | 0.200 | 0.450 | +0.250 | 0.350 | 0.175 |
| skin_growth | 16 | 0.594 | 0.578 | -0.016 | 0.125 | 0.188 |
| skin_irritation | 14 | 0.417 | 0.637 | +0.220 | 0.262 | 0.149 |
| skin_rash | 149 | 0.300 | 0.493 | +0.193 | 0.620 | 0.306 |
| swollen_eye | 15 | 0.283 | 0.594 | +0.311 | 0.367 | 0.156 |
| swollen_tonsils | 18 | 0.292 | 0.611 | +0.319 | 0.583 | 0.306 |

## Sample Outputs

### eye_redness (sim=0.542)
**Query**: Hello doctor, mujhe eye ke aas paas bahut dard ho raha hai jab mujhe toy gun ki goli se unexpected hit laga. Cornea mein thodi si redness hai aur eye ...
**Zero-shot**: Arre bhai, aapka dard bahut ho sakta hai. Kya aapne toy gun se goli aapke eye par direct se laga hai ya phir aapke chehre par? Aapke symptoms ke according to mujhe lagta hai ki aapke eye mein ek chhot...
**Grounded**: Arre bhai, main dekh raha hoon ki tumne cricket ball se injury liya tha. Aapke paas photophobia aur dull ache hai, jisse main samajh raha hoon ki aapke right eye mein traumatic iritis ho sakti hai. Is...
- Factual: 0.25 → 1.00 | Halluc: 0.00 → 0.00

### lip_swelling (sim=0.526)
**Query**:  Hi, mere gaal sujan aati hai (ya toh right ho jata hai ya left, kabhi dono nahi, aur sirf cheek bone par). Please doctor, dekhiye is vartamaan condit...
**Zero-shot**: Arre bhai, mere gaal sujan aati hai, yeh ek common problem hai. Aapka description sunke lagta hai ki aap Trigeminal Neuralgia ke symptoms dekh rahe hain, jismein cheekbone par sujan aati hai. Yeh ek n...
**Grounded**: Arre, beta, maine dekha hai ki aapke gaal mein sujan aati hai. Kya aapne kaha ki aapke gaal sujan aati hai ya sirf cheek bone par? 

Aapke case mein, CBCT (Vatech) ne dekha ki aapke left upper canine ...
- Factual: 0.00 → 0.25 | Halluc: 1.00 → 0.00

### skin_rash (sim=0.580)
**Query**: Mujhe ek samasya hai mere skin me jo mere legs par develop hui aur ab mere arms aur wrists par hai, woh bahut khujali kar sakti hai, woh lal dots jais...
**Zero-shot**: Bhai, mujhe lagta hai ki aapko ek skin problem hai, jiska naam 'urticaria' ho sakta hai. Iska matlab hai ki aapko itni khujali hoti hai aur lal dots dikhte hain. Lekin, aapki blood report ko dekhte hu...
**Grounded**: Didi, aapki samasya ka sabse pehle kya hai, yehi dekhna hai. Aapki blood report mein neutrophil count 9.9 hai, jo thoda high hai, aur WBC count 12.4 hai, jo normal hai. Lekin aapki skin par lal dots a...
- Factual: 0.00 → 1.00 | Halluc: 1.00 → 0.00

---
*Generated using Llama-3.3-70B via Groq API on 362 sampled pairs*
