> **NOTE (2026-10-23): this split is UNUSED.** Per Yaswant's decision, OpenDecisions is not split; all 2,352 items in `raw/opendecisions.jsonl` are the evaluation set. The train/validation/test files are kept for reference only.

# OpenDecisions split

- Source: https://huggingface.co/datasets/jgeuter/OpenDecisions
- Version / revision: ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe
- License: CC BY-SA 4.0
- Total items: 2352  (groups: 2014)
- Split (seed 42, 70/15/15): train 1646, validation 353, test 353
- item_id: native `id` field (unique)
- Stratification key: family x question type (choice/noul/score); tier, use_case and topic follow proportionally
- Grouping (never split across sets): native `group` field (paired/metamorphic variants: fairness, negation pairs, perturbations, option order) unioned with identical `state` text
- Notes: Source is itself the published validation set; we re-split it for our evaluation protocol.
- Files: train/train.jsonl, validation/validation.jsonl, test/test.jsonl (all original fields + item_id); split_manifest.csv
- Regenerate: `.venv\Scripts\python scripts\split_datasets.py`

| stratum | train | validation | test |
|---|---|---|---|
| calibration_gen|choice | 17 | 4 | 3 |
| calibration_gen|noul | 28 | 6 | 6 |
| calibration_gen|score | 21 | 5 | 4 |
| calibration|choice | 7 | 1 | 2 |
| calibration|noul | 10 | 3 | 2 |
| calibration|score | 8 | 1 | 1 |
| composite_gen|choice | 25 | 5 | 6 |
| composite_gen|noul | 28 | 6 | 6 |
| composite_gen|score | 21 | 5 | 4 |
| fairness|choice | 67 | 16 | 14 |
| fairness|noul | 47 | 9 | 10 |
| fairness|score | 51 | 10 | 12 |
| jaggedness|choice | 49 | 11 | 10 |
| jaggedness|noul | 35 | 7 | 8 |
| jaggedness|score | 30 | 6 | 6 |
| llm_router|choice | 266 | 57 | 57 |
| llm_router|noul | 84 | 18 | 18 |
| llm_router|score | 136 | 29 | 29 |
| multilingual|choice | 166 | 36 | 36 |
| multilingual|noul | 59 | 13 | 13 |
| multilingual|score | 69 | 15 | 15 |
| negation_pair_gen|choice | 50 | 10 | 10 |
| negation_pair_gen|noul | 62 | 14 | 14 |
| option_order|choice | 84 | 18 | 18 |
| perturb_pair|choice | 34 | 6 | 8 |
| perturb_pair|noul | 22 | 4 | 4 |
| perturb_pair|score | 22 | 6 | 4 |
| usecase|choice | 59 | 13 | 13 |
| usecase|noul | 35 | 7 | 8 |
| usecase|score | 54 | 12 | 12 |

