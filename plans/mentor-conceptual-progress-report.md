# Mentor Conceptual Progress Report

> Project: **Grounded Multimodal RAG for Hinglish Clinical Decision Support**  
> Focus of this update: conceptual progress, key findings, and dataset direction (non-code summary)

---

## 1) Objective of Today’s Work

We aimed to validate, at prototype level, whether **evidence grounding** improves reliability under Hinglish clinical queries.

### Hypothesis in focus
**H1:** Grounded response generation should be more factually reliable than non-grounded (zero-shot) generation.

---

## 2) Conceptual Strategy Used

We treated the problem as a **reasoning alignment** challenge, not pure translation:

- Informal Hinglish query intent
- Mapped against formal clinical evidence
- Explanation generated with and without evidence grounding
- Compared using factuality and hallucination behavior

### Comparison Modes

| Mode | Description | Expected behavior |
|---|---|---|
| Zero-shot | Response without retrieved evidence grounding | More fluent but less constrained |
| Grounded | Response constrained by retrieved evidence | Better factual consistency, lower hallucination |

---

## 3) Evaluation Philosophy (What we measured)

| Metric family | What it tells us | Better direction |
|---|---|---|
| Retrieval quality (Top-1 / Top-k hits) | How often relevant evidence is found | Higher |
| Factual gain | Improvement from grounded over zero-shot | Higher |
| Hallucination drop | Reduction in unsupported claims after grounding | Higher |
| p-value (paired test) | Whether differences are statistically meaningful | Lower (< 0.05) |

---

## 4) H1 Prototype Outcome (Core Result)

The prototype-level H1 comparison showed a positive grounding effect:

- Grounded mode improved factual reliability over zero-shot
- Grounded mode improved hallucination behavior in the main prototype run
- Paired statistical testing indicated meaningful directional support for H1

### Mentor-facing interpretation
**Evidence grounding improves response reliability for this Hinglish clinical reasoning setting (at MVP scale).**

---

## 5) Dataset Comparison Insight (Important Clarification)

We also ran a synthetic profile-level comparison across potential dataset families for early directional insight.

### Simple interpretation rule
Synthetic score and final dataset choice are not the same thing.

| What we look at | Simple meaning | How we use it |
|---|---|---|
| Synthetic score | How a dataset behaves in this practice setup | Early hint only |
| Objective fit | How well dataset matches our real problem (Hinglish + radiology) | Final decision basis |

So even if one dataset scores high in synthetic testing, we still choose datasets based on our actual project goal.

---

## 6) Final Dataset Direction (Objective-Aligned)

Despite synthetic ranking differences, we retain this final practical direction:

1. **Open-i** (radiology evidence foundation)  
2. **MMCQSD-guided Hinglish patterning** (language realism)  
3. **HMG construction** on top of (1) + (2)

### Why this remains correct

| Dataset role | Why it matters for our objective |
|---|---|
| Open-i | Best fit for radiology-grounded evidence retrieval |
| MMCQSD | Best fit for realistic Hinglish/code-mixed query behavior |
| HMG | Bridges formal report evidence with informal Hinglish user intent |

---

## 7) What This Progress Demonstrates

- We now have a validated evidence-first conceptual pipeline
- We have early statistical support for H1 under prototype conditions
- We established a disciplined interpretation framework (proxy signal vs objective fit)
- We have a clear path to move from synthetic proxy to real subset evaluation

---

## 8) Next Step (Immediate)

To increase external validity:

- Run the same evaluation logic on a real Open-i subset
- Keep MMCQSD-guided Hinglish generation behavior
- Preserve the same H1 testing structure for continuity

---

## 9) Visual Summary

```text
Hinglish Query
      |
      v
Retrieve Evidence  -------->  Zero-shot Output (no evidence)
      |                                |
      v                                v
Grounded Output (with evidence)   Compare both outputs
                 \                 /
                  \               /
                   v             v
            Factuality + Hallucination + Statistical Test (H1)
```

---

## 10) One-Line Mentor Conclusion

We now have early, statistically supported prototype evidence that grounding improves reliability, and we have aligned dataset decisions to the actual Hinglish-radiology objective rather than synthetic ranking artifacts.

