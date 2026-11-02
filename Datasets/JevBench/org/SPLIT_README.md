# JevBench/org split

- Source: https://huggingface.co/datasets/JevBench/jevbench (code: https://github.com/JevBench/jevbench)
- Version / revision: cfd3325a6ff0752cda83f799c507ce81c4d0c58d
- License: Apache-2.0
- Total items: 2240  (groups: 240)
- Split (seed 42, 70/15/15): train 1575, validation 330, test 335
- item_id: jevbench_<case_id>_<question_key> (pair is unique)
- Stratification key: domain (12 domains); each case's questions are kept together, so type mix (noul/choice/score) follows the case
- Grouping (never split across sets): case_id: all questions of a case (same state text) are in one split
- Notes: Unit = one question from data/questions.jsonl (2,240). rewrites.jsonl is NOT split: it has no case_id and its `source` matches question-instruction TEMPLATES shared by 3-20 cases (684 of 1,697 match no question text verbatim), so rewrites cannot be assigned to a case without merging cases. gold.jsonl duplicates the `gold` field already present in questions. 302 questions have gold 'unknown'. Original raw files untouched in JevBench/raw/JevBench-org-hf. The old fstandhartinger files in JevBench/raw are not split.
- Files: train/train.jsonl, validation/validation.jsonl, test/test.jsonl (all original fields + item_id); split_manifest.csv
- Regenerate: `.venv\Scripts\python scripts\split_datasets.py`

| stratum | train | validation | test |
|---|---|---|---|
| agent_logs | 120 | 25 | 24 |
| content_moderation | 168 | 36 | 36 |
| customer_support | 168 | 36 | 36 |
| ecommerce_returns | 137 | 29 | 28 |
| education_grading | 112 | 24 | 24 |
| finance_ops | 134 | 28 | 28 |
| healthcare_admin | 112 | 24 | 24 |
| hr_recruiting | 117 | 24 | 24 |
| it_incidents | 123 | 25 | 24 |
| legal_contracts | 119 | 24 | 32 |
| logistics | 120 | 25 | 25 |
| security_ops | 145 | 30 | 30 |
