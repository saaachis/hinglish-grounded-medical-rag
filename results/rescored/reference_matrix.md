# The grounding effect under both metrics and both references

Delta = grounded - zero-shot, within query, same rows scored both ways.
Significance is BH-corrected across the whole re-scoring family.

| Condition | precision vs evidence | F1 vs evidence | F1 vs unbiased |
|---|---|---|---|
| llama-3.1-8b-instant, oracle evidence | +0.2962 ***<br><sub>n=669, d=+0.72</sub> | +0.2029 ***<br><sub>n=651, d=+0.68</sub> | +0.0620 ***<br><sub>n=613, d=+0.22</sub> |
| gpt-oss-120b, oracle evidence | +0.2608 ***<br><sub>n=214, d=+0.64</sub> | +0.0926 ***<br><sub>n=207, d=+0.28</sub> | -0.0207 n.s.<br><sub>n=189, d=-0.07</sub> |
| gpt-oss-120b, real retrieval | +0.0704 **<br><sub>n=221, d=+0.19</sub> | -0.0320 *<br><sub>n=210, d=-0.12</sub> | -0.0469 *<br><sub>n=200, d=-0.17</sub> |
| gpt-oss-20b, oracle evidence | +0.1966 ***<br><sub>n=140, d=+0.45</sub> | -0.0067 n.s.<br><sub>n=134, d=-0.02</sub> | -0.0217 n.s.<br><sub>n=133, d=-0.06</sub> |
| gpt-oss-20b, real retrieval | +0.1071 **<br><sub>n=138, d=+0.25</sub> | -0.0698 **<br><sub>n=130, d=-0.26</sub> | -0.0574 n.s.<br><sub>n=130, d=-0.16</sub> |

`***` BH p < 0.001 · `**` < 0.01 · `*` < 0.05 · `n.s.` not significant

> **Absolute levels are not quality.** Against the unbiased reference a constant one-word answer ("swelling") scores 0.7132 on precision, above every real system. Only the paired deltas in this table are interpretable, and only within a column.

## Reading

- Against the evidence the grounded arm was given, the effect is positive in 5 of 5 conditions on precision.
- Against a reference neither arm saw, it is positive in 1 of 5 conditions on F1.
- The benefit is therefore **reference-dependent**: real under the protocol in common
  use, and much smaller or absent under one that removes the circularity. The abstract
  must state both, in that order.
