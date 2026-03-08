# Prototype Results And Hypothesis Report

## Purpose
This report summarizes the latest results from the **real `Open-i + MMCQSD` prototype** and explains how those results should be interpreted with respect to the project hypothesis.

The goal here is to present:
- what was tested
- what the prototype produced
- what the numbers mean
- what can and cannot be concluded at this stage

## Hypothesis being checked
The core hypothesis for this prototype phase is:

> Grounded generation using retrieved evidence should perform better than zero-shot generation in terms of factual support and reduced hallucination.

In simple terms:
- `zero-shot` = answer without retrieved evidence
- `grounded` = answer after retrieving supporting report evidence

## Experimental setup

### Data used
- Evidence/report corpus: `Open-i`
- Hinglish query source: `MMCQSD`
- Final cleaned real matched subset: `11` query-report pairs

### Prototype pipeline used
1. retrieve relevant `Open-i` evidence for each `MMCQSD` query
2. generate a zero-shot answer
3. generate a grounded answer using retrieved evidence
4. compare zero-shot vs grounded outputs

### Prototype flow
`MMCQSD Hinglish query -> retrieve Open-i evidence -> generate zero-shot answer -> generate grounded answer -> compare outputs`

### Evaluation style
The current evaluation is still prototype-level and lightweight.

It measures:
- retrieval success
- factual support score
- hallucination behavior
- paired significance between grounded and zero-shot outputs

## Final result summary

### Main metrics

| Metric | Value | Quick meaning |
|---|---:|---|
| Samples evaluated | `11` | very small cleaned real subset |
| Retrieval top-1 hit rate | `0.000` | correct paired report was not rank-1 |
| Retrieval top-k hit rate | `0.364` | correct paired report appeared in top-k in some cases |
| Mean factual score, zero-shot | `0.095` | baseline factual support is low |
| Mean factual score, grounded | `0.119` | grounding improves factual support slightly |
| Hallucination rate, zero-shot | `1.000` | unsupported content is still very high |
| Hallucination rate, grounded | `1.000` | grounding did not reduce hallucination yet |
| Statistical test | `Wilcoxon signed-rank` | paired comparison between zero-shot and grounded |
| p-value | `0.000977` | early directional difference is detectable |

### Grounded vs zero-shot snapshot

| Measure | Zero-shot | Grounded | Change |
|---|---:|---:|---:|
| Factual score | `0.095` | `0.119` | `+0.024` |
| Hallucination rate | `1.000` | `1.000` | `0.000` |
| Retrieval top-k hit rate | - | `0.364` | evidence sometimes appears in top-k |

### Metric key

| Metric | Meaning |
|---|---|
| `Top-1 hit` | correct report was retrieved first |
| `Top-k hit` | correct report appeared somewhere in retrieved set |
| `Factual score` | answer matched the reference/evidence better |
| `Hallucination rate` | answer contained unsupported content |
| `p-value` | grounded vs zero-shot difference was statistically detectable |

## Quick interpretation

| Area | What the result says |
|---|---|
| Factuality | grounded is slightly better than zero-shot (`0.119` vs `0.095`) |
| Retrieval | top-k shows some useful evidence retrieval (`0.364`), but top-1 is still weak (`0.000`) |
| Hallucination | no improvement yet (`1.000` vs `1.000`) |
| Statistics | the grounded vs zero-shot difference is detectable (`p = 0.000977`) |

### Bottom line
- grounding helps factuality a little
- retrieval is partially working, but not strong enough yet
- hallucination control is still not improved
- this is an early real-data signal, not a final strong result

## What improved compared to the earlier weaker real matching
The latest cleaned real subset is smaller, but better aligned than the earlier weakly filtered version.

Compared with the earlier real attempt:

| Measure | Earlier weaker real subset | Current cleaned real subset | Change |
|---|---:|---:|---:|
| Top-k retrieval | `0.175` | `0.364` | `+0.189` |
| Grounded factual score | `0.090` | `0.119` | `+0.029` |

This is important because it shows:
- stricter real matching improved the usefulness of the subset
- cleaner data helped the prototype more than simply keeping more noisy rows

## What these results mean for the hypothesis

| Item | Statement |
|---|---|
| `Hypothesis tested` | grounded generation should outperform zero-shot generation on factual support and hallucination reduction |
| `Expected pattern` | grounded should increase factual score and reduce hallucination |
| `Observed result` | factual score improved slightly, hallucination did not improve |
| `Current verdict` | **partial support only** |

### Why the hypothesis is only partially supported
- grounded factual score improved
- statistical comparison shows a detectable difference
- but hallucination did not decrease
- and retrieval is still weak at rank 1

### Correct final reading
> The current real prototype gives early support that grounding can improve factuality, but it does not yet show the full expected benefit of grounded medical RAG.

## Assumptions and limitations

| Item | Current assumption / limitation |
|---|---|
| Pairing | `Open-i` and `MMCQSD` are not naturally paired, so current matches are weakly supervised |
| Sample size | only `11` cleaned real pairs were retained |
| Retrieval | current retrieval stack is still prototype-level |
| Generation | grounded answer generation is still simple and generic |

## Mentor-facing conclusion

- the prototype now runs on real data
- grounding gives a small factual improvement over zero-shot
- stricter real matching improved the subset quality
- but the result should still be presented as an **early real prototype result**, not a final validated system

## Practical takeaway

- the system is operational on real data
- a cleaner matched subset improved the prototype
- evidence grounding is showing a small positive effect
- stronger final results now depend on better pair quality, retrieval quality, and grounded generation quality

## Recommended next step

The best next move is:
- manually review and validate the current `11` real matched pairs
- build a stronger gold subset from them
- then rerun the same hypothesis evaluation on the validated set

That will make the next round of results much more reliable for both mentor review and final project development.
