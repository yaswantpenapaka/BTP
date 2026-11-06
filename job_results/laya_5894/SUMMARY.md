# laya job 5894

- partition: gpu-A100
- node: gpu-A100-01
- nodelist: gpu-A100-01
- H100 not used (btech qos)
- SIF: /userhome/btech/p.yaswant/containers/vllm-openai.sif
- import check:
  - torch 2.13.0+cu130
  - cuda True
  - laya 0.4.15
  - device NVIDIA A100-PCIE-40GB
  - finetune finetune TrainConfig TrainConfig
- Hub id: convaiinnovations/laya
- Hub revision: 7b928d828b7b0e022f929d9bd2e44165aa270148
- English root only: checkpoints/laya-base (multilingual/ and typed-decisions/ not used)
- CUDA patch: eval/laya.py LayaPlugin uses device=os.environ.get("LAYA_DEVICE", "cuda")
- build-data JSONL: ensure_ascii=True
- TrainConfig: epochs=4, eval_data=<dataset>_validation.jsonl; other fields are laya 0.4.15 defaults
- finetune(..., device="cuda") from checkpoints/laya-base for each dataset. No OpenDecisions fine-tune. No sequential transfer.
- stage seconds:
  - import 16
  - ft_jevbench 212
  - infer_jevbench 48
  - ft_fastdecisions 343
  - infer_fastdecisions 44
  - infer_opendecisions 107
- outputs:
  - /userhome/btech/p.yaswant/BTP/Decision_Models/eval/outputs/raw/laya__jevbench.jsonl
  - /userhome/btech/p.yaswant/BTP/Decision_Models/eval/outputs/raw/laya__fastdecisions.jsonl
  - /userhome/btech/p.yaswant/BTP/Decision_Models/eval/outputs/raw/laya__opendecisions.jsonl
  - /userhome/btech/p.yaswant/BTP/Decision_Models/checkpoints/laya-base-ft-jevbench
  - /userhome/btech/p.yaswant/BTP/Decision_Models/checkpoints/laya-base-ft-fastdecisions
- row counts:
- laya__jevbench.jsonl: rows=335 errors=0
- laya__opendecisions.jsonl: rows=2352 errors=0
- laya__fastdecisions.jsonl: rows=255 errors=0

Checked after the job exited 0:0 on gpu-A100-01 (started 2026-11-03 07:31 IST, elapsed 12m51s, MaxRSS about 2.8G).

Held-out validation inside `finetune` (not the test sets): JevBench accuracy 0.6634 to 0.8170. Fast Decisions accuracy 0.5631 to 0.7021.

Test rows, `model=laya`:

- JevBench `laya-base-ft-jevbench`: 335/335, 0 errors, 248/285 correct (50 rows have no gold).
- Fast Decisions `laya-base-ft-fastdecisions`: 255/255, 0 errors, 104/255 correct.
- OpenDecisions untuned `laya-base`: 2352/2352, 0 errors, 1027/2352 correct. No fine-tune.

Library warnings only, in `laya_ft_infer_5894.err`: learning-rate scheduler order, JevBench temperature clamped at 5, conflicting question schemas while writing `questions.json`, and the base checkpoint's temperature clamped on OpenDecisions inference. Other models' raw files were not rewritten. Slurm logs: `job_results/laya_ft_infer_5894.out` and `.err`.
