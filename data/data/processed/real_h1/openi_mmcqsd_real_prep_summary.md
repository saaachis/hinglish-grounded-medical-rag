# Open-i + MMCQSD Real Preparation Summary

- Total Open-i reports available: **6687**
- MMCQSD cardio-respiratory candidates found: **11**
- Final aligned query set retained: **11**
- Mean alignment score: **0.2823**

## Notes
- MMCQSD was filtered to retain cardio-respiratory style cases most compatible with a chest-report corpus.
- Each MMCQSD query was aligned to the best-matching Open-i report using TF-IDF similarity plus lightweight concept-overlap bonuses.
- This is a weakly supervised real pairing, used to make the prototype pipeline operational on real data.
