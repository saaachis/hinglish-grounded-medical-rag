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
| Mean factual score, grounded | `0.412` | grounding improves factual support strongly |
| Hallucination rate, zero-shot | `1.000` | unsupported content is still very high |
| Hallucination rate, grounded | `0.000` | grounding removes unsupported content in this run |
| Statistical test | `paired_t_test` | paired comparison between zero-shot and grounded |
| p-value | `0.000000` | strong detectable grounded vs zero-shot difference |

### Grounded vs zero-shot snapshot

| Measure | Zero-shot | Grounded | Change |
|---|---:|---:|---:|
| Factual score | `0.095` | `0.412` | `+0.317` |
| Hallucination rate | `1.000` | `0.000` | `-1.000` |
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
| Factuality | grounded is much better than zero-shot (`0.412` vs `0.095`) |
| Retrieval | top-k shows some useful evidence retrieval (`0.364`), but top-1 is still weak (`0.000`) |
| Hallucination | strong drop in this run (`1.000` -> `0.000`) |
| Statistics | the grounded vs zero-shot difference is strongly detectable (`p = 0.000000`) |

### Bottom line
- grounding improves factuality strongly
- retrieval is partially working, but not strong enough yet
- hallucination control improved strongly in this prototype run
- this is a strong prototype-level signal, but still not final-system evidence due to sample size

## What improved compared to the earlier weaker real matching
The latest cleaned real subset is smaller, but better aligned than the earlier weakly filtered version.

Compared with the earlier real attempt:

| Measure | Earlier weaker real subset | Current cleaned real subset | Change |
|---|---:|---:|---:|
| Top-k retrieval | `0.175` | `0.364` | `+0.189` |
| Grounded factual score | `0.090` | `0.412` | `+0.322` |
| Grounded hallucination rate | `1.000` | `0.000` | `-1.000` |

This is important because it shows:
- stricter real matching improved the usefulness of the subset
- cleaner data helped the prototype more than simply keeping more noisy rows

## What these results mean for the hypothesis

| Item | Statement |
|---|---|
| `Hypothesis tested` | grounded generation should outperform zero-shot generation on factual support and hallucination reduction |
| `Expected pattern` | grounded should increase factual score and reduce hallucination |
| `Observed result` | factual score increased strongly and hallucination dropped strongly in this prototype run |
| `Current verdict` | **H1 supported at prototype level** |

### Why H1 is supported at prototype level
- grounded factual score improved clearly (`0.095` -> `0.412`)
- grounded hallucination reduced (`1.000` -> `0.000`)
- statistical comparison shows a strong detectable difference
- the result direction matches the expected H1 pattern

### Correct final reading
> The current real prototype supports H1 at prototype level: grounded outputs are more factual and less hallucinatory than zero-shot outputs in this run. However, this should still be interpreted with caution due to small sample size and weakly supervised dataset pairing.

## Assumptions and limitations

| Item | Current assumption / limitation |
|---|---|
| Pairing | `Open-i` and `MMCQSD` are not naturally paired, so current matches are weakly supervised |
| Sample size | only `11` cleaned real pairs were retained |
| Retrieval | current retrieval stack is still prototype-level |
| Generation | grounded answer generation is still simple and generic |

## Mentor-facing conclusion

- the prototype now runs on real data
- grounding gives a strong factual improvement over zero-shot in this run
- stricter real matching improved the subset quality
- H1 is satisfied at prototype level in this run
- but the result should still be presented as a **prototype-level validated signal**, not a final system claim

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
