# Aims, Deliverables & Challenges

> What we are building, what we will deliver, and what might get in the way.

---

## Clear Aim

**Build and evaluate an evidence-grounded RAG system that can take informal Hinglish medical queries and generate clinically safe, factually consistent Hinglish explanations — backed by real medical evidence, not hallucinated from model memory.**

This is NOT:
- A translation system (Hinglish → English)
- A diagnostic tool (we don't diagnose)
- A general chatbot (we generate evidence-grounded explanations only)

This IS:
- A clinical decision support tool
- An evidence-first reasoning system
- A research contribution to safe multilingual medical AI

---

## Concrete Deliverables

### D1. HMG Dataset
- ~4,000 triplets: `{X-ray, English Report, Hinglish Query}`
- With CMI (Code-Mixing Index) scores bucketed into low / medium / high
- Constructed via synthetic-to-real pipeline (Llama-3 + MMCQSD templates)

### D2. Evidence-Grounded RAG Pipeline
- End-to-end system: Hinglish query → evidence retrieval → grounded explanation
- Text encoding via LaBSE (multilingual)
- FAISS-based retrieval with adaptive context selection
- Grounded generation via LLaVA-v1.5 with QLoRA fine-tuning

### D3. Hallucination Control via DPO
- Preference-tuned model that prefers evidence-grounded responses
- Measurable reduction in hallucination rate vs zero-shot baselines

### D4. Statistical Evaluation of Three Hypotheses
- **H1:** RAG vs zero-shot LLM (grounding effect)
- **H2:** Robustness across code-mixing levels
- **H3:** Authoritative vs general evidence quality
- With proper statistical tests, p-values, effect sizes, confidence intervals

### D5. Streamlit Demo Interface
- Interactive demo showing query → retrieval → explanation flow
- Visual evidence display (reports + optional X-ray)

### D6. Research Paper
- All development feeds into the final research paper for Research Discourse
- Reproducible results, clear methodology, statistical rigor

---

## Challenges We Are Facing / Might Face

### Linguistic Challenges

| Challenge | Impact | Our Mitigation |
|---|---|---|
| **No standard Hinglish grammar** | LLMs may misparse queries | Use MMCQSD as realistic linguistic templates |
| **Regional variation** in Hinglish | Model might not generalize across dialects | Diversify synthetic generation with varied prompting |
| **Medical terms in Hinglish are indirect** (e.g., "pani bhar gaya" for pleural effusion) | Semantic gap in retrieval | LaBSE handles cross-lingual alignment; CMI-stratified evaluation |
| **Synthetic vs real Hinglish gap** | Generated queries may lack naturalness | Use MMCQSD patterns as template; plan quality validation |

### Technical Challenges

| Challenge | Impact | Our Mitigation |
|---|---|---|
| **Compute constraints** — fine-tuning 7B models | Slow training, OOM errors | QLoRA (4-bit quantization) — designed for consumer GPUs |
| **FAISS retrieval sensitivity** | Minor phrasing changes → different results | Adaptive truncation (similarity-score-based) filters noise |
| **Cross-modal alignment** (text ↔ image) | Hinglish text must "find" English report in vector space | BioMedCLIP + LaBSE encode into aligned spaces |
| **Hallucination despite grounding** | Model might still generate unsupported claims | DPO training with preference pairs |
| **Evaluation of factuality** is non-trivial | Standard metrics (BLEU) don't capture clinical safety | MMFCM metric + factual consistency scoring |

### Research / Process Challenges

| Challenge | Impact | Our Mitigation |
|---|---|---|
| **No existing Hinglish radiology dataset** | Must build from scratch | Synthetic-to-real pipeline (our core contribution) |
| **Limited team experience** with full ML pipelines | Learning curve for some members | Collaborative work, skill sharing, structured tasks |
| **Balancing dev quality with timeline** | Risk of scope creep or delays | Clear phased plan, regular progress checks |
| **Offline evaluation only** | No clinician-in-the-loop validation | Acknowledged as limitation; rigorous offline metrics instead |

---

## Success Criteria

We consider the project successful if:

1. HMG dataset is constructed with ~4,000 valid triplets
2. RAG pipeline produces Hinglish explanations grounded in retrieved evidence
3. H1 is statistically supported (RAG > zero-shot on factuality)
4. Hallucination rate is measurably lower with DPO
5. Results are reproducible and documented for the research paper
6. All three team members have contributed to and learned from the project
