# anyjev_l0_qwen3_8b job 5558

- partition: gpu-A100
- node: gpu-A100-02
- nodelist: gpu-A100-02
- smoke: 0
- prior job: 5556
- SIF: /userhome/btech/p.yaswant/containers/vllm-openai.sif
- SIF tag: docker://vllm/vllm-openai:v0.29.0
- import check:
  - anyjev 0.3.0
  - vllm 0.29.0
  - cuda True
  - device NVIDIA A100-PCIE-40GB
- AnyJev commit: 35311c58955716cb8955ce1368a665988bc63735
- model: /userhome/btech/p.yaswant/BTP/models/Qwen3-8B
- hub revision: b968826d9c46dd6066d109eabc6255188de91218
- imports: Decider(VLLMBackend(url, Qwen3-8B, tokenizer_name=weights), level="L0")
- server: vllm serve --enable-prefix-caching --logprobs-mode processed_logprobs
- questions: one Decider.decide per item; choice / noul / score as in AnyJev scripts/eval_tacit.py
- line counts:
-     255 /userhome/btech/p.yaswant/BTP/Decision_Models/eval/outputs/raw/anyjev_l0_qwen3_8b__fastdecisions.jsonl
-     335 /userhome/btech/p.yaswant/BTP/Decision_Models/eval/outputs/raw/anyjev_l0_qwen3_8b__jevbench.jsonl
-    2352 /userhome/btech/p.yaswant/BTP/Decision_Models/eval/outputs/raw/anyjev_l0_qwen3_8b__opendecisions.jsonl
-    2942 total
- error counts:
- anyjev_l0_qwen3_8b__jevbench.jsonl: rows=335 errors=0
- anyjev_l0_qwen3_8b__opendecisions.jsonl: rows=2352 errors=0
- anyjev_l0_qwen3_8b__fastdecisions.jsonl: rows=255 errors=14

Checked after the job exited 0:0 on gpu-A100-02 (started 2026-10-28 14:12 IST, elapsed 13m04s, MaxRSS about 4.9G). Memory request on the job was 48G.

- JevBench: 335/335 rows, 0 errors, 335 answered. 50 rows have no gold. Accuracy on the other 285 is 239/285.
- OpenDecisions: 2352/2352 rows, 0 errors, 2352 answered. Accuracy 1467/2352.
- Fast Decisions: 255/255 rows, 241 answered, 14 item errors. Accuracy on the answered rows is 98/241.
- Those 14 errors are `choice supports at most 26 options in letter readout` from AnyJev `Question.choice`. The rows are stored.
- Smoke job 5556 produced the first 2 rows of each file. This job skipped them and wrote the rest.
- Qwen3-4B `anyjev_l0__*` sha256 values are unchanged.
