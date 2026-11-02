# anyjev_l0 job 5473

- partition: gpu-A100
- smoke: 0
- prior job: 5469
- SIF: /userhome/btech/p.yaswant/containers/vllm-openai.sif
- SIF tag: docker://vllm/vllm-openai:v0.29.0
- AnyJev commit: 35311c58955716cb8955ce1368a665988bc63735
- model: /userhome/btech/p.yaswant/BTP/models/Qwen3-4B
- imports: Decider(VLLMBackend(url, Qwen3-4B, tokenizer_name=weights), level="L0")
- server: vllm serve --enable-prefix-caching --logprobs-mode processed_logprobs
- questions: one Decider.decide per item; choice / noul / score as in AnyJev scripts/eval_tacit.py
- line counts:
-     255 /userhome/btech/p.yaswant/BTP/Decision_Models/eval/outputs/raw/anyjev_l0__fastdecisions.jsonl
-     335 /userhome/btech/p.yaswant/BTP/Decision_Models/eval/outputs/raw/anyjev_l0__jevbench.jsonl
-    2352 /userhome/btech/p.yaswant/BTP/Decision_Models/eval/outputs/raw/anyjev_l0__opendecisions.jsonl
-    2942 total

Checked after the job exited 0:0 on gpu-A100-01 (elapsed 9m53s, MaxRSS about 5.4G).

- JevBench: 335/335 rows, 0 errors, 335 answered. 50 rows have no gold. Accuracy on the other 285 is 239/285.
- OpenDecisions: 2352/2352 rows, 0 errors, 2352 answered. Accuracy 1434/2352.
- Fast Decisions: 255/255 rows, 241 answered, 14 item errors. Accuracy on the answered rows is 96/241.
- Those 14 errors are `choice supports at most 26 options in letter readout` from AnyJev `Question.choice`. The rows are stored; the other heads were not invented around the limit.
- Smoke job 5469 produced the first 2 rows of each file. This job skipped them and wrote the rest.
