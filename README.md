# Statistical evaluation of LLM decision models
**B.Tech Project (BTP-I), Department of CSE, IIT Guwahati**

Decision models that return a probability distribution over options are often compared by accuracy alone. Two models can post nearly the same accuracy and still disagree on many individual items, or place very different mass on the same options. Accuracy does not say whether those models are interchangeable. This project compares six such models pairwise on three decision datasets, using agreement, Cohen's kappa, rank and linear correlation of option probabilities, probability-space distances (TVD, JSD, KL), and McNemar's paired test on 0/1 correctness, reported next to gold accuracy.

## Models
| Key | Model | Access |
|---|---|---|
| `jev` | Jev (`jev-1.13.0`) | API |
| `luna` | Luna (`gpt-6-luna`, OpenAI Decisions) | API |
| `anyjev_l0` | AnyJev L0 on Qwen3-4B | open weights, gpu-A100 (vLLM) |
| `anyjev_l0_qwen3_8b` | AnyJev L0 on Qwen3-8B | open weights, gpu-A100 (vLLM) |
| `semif` | SemIf (direct mode) on Qwen3.5-4B | open weights, gpu-A100 |
| `laya` | Laya (English base; fine-tuned per dataset on JevBench and Fast Decisions; untuned base on OpenDecisions) | open weights, gpu-A100 |

Model weights and fine-tune checkpoints live on the GPU cluster and are not in this repository.

## Datasets and splits
Splits are produced by `Datasets/scripts/split_datasets.py` (seed 42). Details are in each `SPLIT_README.md`.

| Dataset | Source size | Split | Evaluated on |
|---|---|---|---|
| JevBench (org) | 2240 | 70/15/15, seed 42 | test, 335 items (285 with non-`unknown` gold used for accuracy) |
| Fast Decisions | 1700 (public DEVELOPMENT) | 70/15/15 | test, 255 items |
| OpenDecisions | 2352 | 70/15/15 files exist but are unused for transfer eval | full 2352 |

## Metrics
Implemented in `Decision_Models/eval/metrics.py` and unit-tested in `eval/tests/`:

- Agreement and Cohen's kappa on chosen options
- Spearman / Pearson within an item (option probabilities, k ≥ 3) and across items (on confidence)
- Total variation distance (TVD), Jensen–Shannon divergence (JSD, base 2), KL divergence in bits (ε = 1e-6)
- McNemar on 0/1 item correctness (exact binomial if b+c < 25, else χ² with continuity correction)
- Gold accuracy per model

No multiple-testing correction is applied. A Fast Decisions item contributes one unit per question; an item is correct only if every head matches.

## Headline accuracy
From `Decision_Models/eval/outputs/analysis/accuracy.csv`:

| Model | JevBench | OpenDecisions | Fast Decisions |
|---|---|---|---|
| Jev | 94.7% | 83.5% | 42.7% |
| Luna | 93.0% | 81.9% | 42.4% |
| SemIf | 90.2% | 67.9% | 39.0% |
| Laya | 87.0% | 43.7% | 40.8% |
| AnyJev L0 Qwen3-8B | 83.9% | 62.4% | 40.7% |
| AnyJev L0 Qwen3-4B | 83.9% | 61.0% | 39.8% |

All pairwise matrices for the six models are in [`Decision_Models/eval/outputs/analysis/matrices_jev_luna_anyjev_l0_semif_qwen3_8b_laya.xlsx`](Decision_Models/eval/outputs/analysis/matrices_jev_luna_anyjev_l0_semif_qwen3_8b_laya.xlsx). Per-metric CSV matrices live under `eval/outputs/analysis/{jevbench,opendecisions,fastdecisions}/`.

## Repository layout
```
README.md
.gitignore
Decision_Models/                 code for the earlier Jev vs Luna classification study
                                 and the BTP evaluation package
  clients/                       HTTP clients for Jev and Luna
  eval/                          BTP evaluation package (run from Decision_Models/)
    datasets.py models.py        loaders and model plug-ins
    records.py metrics.py        output schema and pairwise metrics
    run.py                       API inference -> outputs/raw/<model>__<dataset>.jsonl
    laya.py                      local Laya fine-tune / infer entry point
    analyze.py                   rebuild matrices in outputs/analysis/
    diagnostics/                 Fast Decisions per-domain notes
    outputs/raw/                 per-item predictions and option probabilities
    outputs/analysis/            accuracy.csv, summary.csv, matrix_*.csv, xlsx workbooks
    tests/
  tests/                         unit tests for the earlier study
  requirements.txt
Datasets/
  scripts/split_datasets.py      70/15/15 split generation (seed 42)
  JevBench/org/                  train / validation / test + SPLIT_README.md
  FastDecisions/                 same
  OpenDecisions/                 same (eval uses the full raw file for transfer)
cluster/                         SemIf runner + SLURM submit script; README for all cluster jobs
job_results/
  anyjev_l0_5473/SUMMARY.md
  anyjev_l0_qwen3_8b_5558/SUMMARY.md
  semif_direct_5539/SUMMARY.md
  laya_5894/SUMMARY.md
```

## Reproduce
```powershell
cd Decision_Models
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# API keys only as environment variables in the current shell
# $env:TYPESAFE_API_KEY = ...
# $env:OPENAI_API_KEY   = ...

python -m eval.run --help
python -m eval.run --models jev luna --datasets jevbench opendecisions fastdecisions
python -m eval.analyze
python -m unittest discover -s eval\tests -t . -v
python ..\Datasets\scripts\split_datasets.py
```

Open-weight models (AnyJev L0 4B/8B, SemIf, Laya) were run on the **gpu-A100** partition (jobs 5473, 5539, 5558, 5894). See `cluster/README.md` and the matching `job_results/*/SUMMARY.md`. The SLURM submit script filename still says `h100` for historical reasons; the live jobs used A100 because the btech QoS does not include H100.

To rebuild the six-model workbook after changing raw outputs:

```powershell
cd Decision_Models
.\.venv\Scripts\Activate.ps1
python -m eval.analyze --models jev luna anyjev_l0 semif anyjev_l0_qwen3_8b laya
python eval\outputs\analysis\make_matrices_laya.py
```

## Security
API keys are read from environment variables only. `.env` files, model checkpoints, Hugging Face caches, virtual environments, and the local `_local_notes/` folder are git-ignored and are not part of this artifact.
