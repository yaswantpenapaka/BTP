# Cluster jobs (open-weight models)

All successful BTP open-weight runs used the **gpu-A100** partition under the btech QoS. H100 was rejected for that QoS. Shared runtime pieces:

- Singularity image: `~/containers/vllm-openai.sif` (tag `docker://vllm/vllm-openai:v0.29.0`)
- Weights on the cluster under `~/BTP/models/` (not in this repo)
- Raw predictions written to `Decision_Models/eval/outputs/raw/<model>__<dataset>.jsonl`

This folder ships the SemIf runner used on the cluster. AnyJev and Laya were driven by short-lived job scripts on the cluster itself; the facts below are taken from the job summaries under `job_results/`.

## SemIf (direct) — job 5539
- Scripts in this folder: `semif_direct.py`, `submit_semif_direct_h100.sh` (targets A100 despite the name)
- SemIf commit `23cf1f39`, model `Qwen3.5-4B`, mode `semif-score --mode direct`
- User venv with torch 2.10.0+cu128 and transformers 5.17.0 (image pins did not match SemIf)
- Fast Decisions: 14 items with more than 26 options were not sent to SemIf (same letter-readout limit as AnyJev)
- Summary: `job_results/semif_direct_5539/SUMMARY.md`

## AnyJev L0 on Qwen3-4B — job 5473
- Library: AnyJev commit `35311c58`, `Decider(VLLMBackend(...), level="L0")`
- Model: `Qwen3-4B`
- vLLM serve with `--enable-prefix-caching --logprobs-mode processed_logprobs`
- One `Decider.decide` per item; choice / noul / score as in AnyJev `scripts/eval_tacit.py`
- Fast Decisions: 14 item errors (`choice supports at most 26 options in letter readout`); rows stored
- Summary: `job_results/anyjev_l0_5473/SUMMARY.md`

## AnyJev L0 on Qwen3-8B — job 5558
- Same AnyJev / vLLM setup as 5473, model `Qwen3-8B`
- Same 14 Fast Decisions letter-readout errors
- Summary: `job_results/anyjev_l0_qwen3_8b_5558/SUMMARY.md`

## Laya — job 5894
- Hub id `convaiinnovations/laya` (English root only; no multilingual / typed-decisions)
- Entry point in the repo: `Decision_Models/eval/laya.py` (`LayaPlugin`, device from `LAYA_DEVICE`, default `cuda`)
- Fine-tune from `checkpoints/laya-base` on JevBench and Fast Decisions (`TrainConfig` epochs=4, eval on each validation split)
- OpenDecisions: untuned base inference only (no fine-tune, no sequential transfer)
- Checkpoints stay on the cluster under `Decision_Models/checkpoints/`
- Summary: `job_results/laya_5894/SUMMARY.md`
