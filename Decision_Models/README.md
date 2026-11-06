> **BTP update:** the main B.Tech Project evaluation (JevBench, OpenDecisions, Fast Decisions; Jev, Luna, AnyJev L0 4B/8B, SemIf, Laya) is in [eval/](eval/README.md). See the [root README](../README.md) for the overview and results. The earlier classification study below is kept as is.

# Decision models: Jev vs OpenAI Decisions

This folder compares two decision APIs on four public classification datasets. The question is whether a small gap in accuracy and cost is enough to treat the models as interchangeable. Agreement, calibration, and paired tests are part of the answer. Similar accuracy is reported next to those other quantities rather than used as a substitute for them.

The models are `jev-1.13.0` (TypeSafe System One) and `gpt-6-luna` (OpenAI Decisions). Both receive the same state JSON, the same instruction, and the same option labels and descriptions. Those words live only in `tasks.py`. Do not edit a template after a run has started. A changed instruction is a different experiment.

No GPU is required. The calls are ordinary HTTP requests from a laptop. The completed runs below cost about $0.17 in total.

| CSV | Dataset | Split | Items | API options |
|---|---|---|---|---|
| `outputs/sst2.csv` | `stanfordnlp/sst2` | validation | 872 | negative, positive |
| `outputs/rotten_tomatoes.csv` | `cornell-movie-review-data/rotten_tomatoes` | test | 1066 | negative, positive |
| `outputs/emotion.csv` | `dair-ai/emotion` (`split`) | test | 2000 | sadness, joy, love, anger, fear, surprise |
| `outputs/financial_phrasebank.csv` | `atrost/financial_phrasebank` | test | 970 | negative, neutral, positive |

Rotten Tomatoes stores its labels as `neg` and `pos`. The loader checks that those are the Hub names, in that order, and then maps index 0 to `negative` and index 1 to `positive`. The prompt text stays `negative` / `positive`. If the Hub names differ, the run stops.

## API keys were entered in the terminal

The runs already stored in `outputs/` were made with keys typed into the current PowerShell session only:

```powershell
$env:TYPESAFE_API_KEY = "paste-key-here"
$env:OPENAI_API_KEY = "paste-key-here"
```

Those two assignments are how the keys were supplied. They were not written into a file in this folder, not saved in `.env`, and not printed in the logs. This repository does not contain either key. Closing the terminal clears them. ChatGPT Plus is not API credit.

- `TYPESAFE_API_KEY` is the TypeSafe key for Jev (`jev-1.13.0`).
- `OPENAI_API_KEY` is the OpenAI key for Decisions (`gpt-6-luna`). OpenAI prepaid credit is required.

Set both variables again in a new terminal before any new paid command. `run.py` reads them from the process environment and refuses to start if a selected key is missing. It does not log the value.

Jev is priced at $0.042 per 1M input tokens. OpenAI Decisions is priced at $0.10 per 1M input tokens. Output tokens are free on both. The default cap is $2.00 per model per process. The four full runs stayed under that cap.

The CSV and analysis files already in `outputs/` can be read, rescored, and reanalyzed with no key and no API call.

## Setup

From a clone of this repository:

```powershell
git clone https://github.com/yaswantpenapaka/BTP.git
cd BTP\Decision_Models
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On this machine the folder is `C:\Users\yaswa\OneDrive\BTP\Decision_Models`. Python 3.10 or newer is required. The tested environment was Python 3.13.

Dependencies are `requests`, `datasets`, `pandas`, `numpy`, `scikit-learn`, `scipy`, and `tqdm`. The OpenAI SDK is not used. Both APIs are called with `requests`.

Check the local tests before any paid call. The tests do not use the network:

```powershell
python -m unittest discover -s tests -v
```

## Smoke test

One call to each selected API, using the SST-2 template and the sentence `The film was dull and too long.` Both should return `negative`. Nothing is written to a dataset CSV. If a response is missing the expected fields, the raw body is saved under `logs/` with any Bearer token removed, and the run stops so the parser can be fixed before a batch.

```powershell
$env:TYPESAFE_API_KEY = "paste-key-here"
$env:OPENAI_API_KEY = "paste-key-here"
python run.py --smoke
```

`--smoke` does not ask for confirmation. If the first API returns an unexpected shape, the second API is still called.

## 20-row SST-2 pilot

```powershell
python run.py --dataset sst2 --model both --limit 20
```

`--limit` does not ask for confirmation. The default concurrency is 4. A completed `(item_id, model)` pair already in the CSV is skipped, so stopping the process does not pay for that pair again. The full SST-2 run in this folder was started after this pilot. It resumed at item 20 and did not send the pilot rows a second time.

## Full runs

A run without `--limit` prints the item count and an estimated cost, then waits until the terminal receives exactly `yes`. The estimate assumes 1000 input tokens per call. It is an estimate, not a quote. If stdin is not a terminal, the run stops and sends nothing. The Hugging Face split is downloaded and checked before that prompt and before any paid call. The loader stops if the split length or the Hub label names differ from `tasks.py`.

The emotion option descriptions, and the financial instruction, still carry TODO notes in `tasks.py`. The emotion descriptions are one line per label. The paper's released code passes no description for those labels. The financial instruction in this project has no backticks around the word sentence. The paper's code puts backticks around `` `sentence` ``. The runs below used the frozen text in `tasks.py`. Check that text against arXiv:2609.37647 (DeuÃŸer, Sparrenberg & Sifa, "Evaluating and Benchmarking the System One Model Jev") before treating the emotion and financial numbers as a direct copy of the paper. Do not change the strings between runs once they are confirmed.

```powershell
python run.py --dataset sst2 --model both
python run.py --dataset rotten_tomatoes --model both
python run.py --dataset emotion --model both
python run.py --dataset financial_phrasebank --model both
```

Other useful flags:

```powershell
python run.py --dataset sst2 --model jev --limit 20
python run.py --dataset sst2 --model openai --limit 20
python run.py --dataset sst2 --model both --max-cost-usd 2
```

`--model` accepts `jev`, `openai`, or `both`.

## Scores without new calls

```powershell
python score.py --dataset sst2
python analyze.py
```

`score.py` reads one dataset CSV and its probability file and rewrites that file's score block. It does not call an API.

`analyze.py` reads all four CSVs and the matching `outputs/<dataset>_probs.jsonl` files. It writes `outputs/analysis_summary.csv` and one `outputs/analysis_<dataset>.txt` report. It does not call an API and it does not modify the source rows. The default equivalence margin is 0.02 accuracy points:

```powershell
python analyze.py --margin 0.02
```

For each dataset the report contains:

- the 2Ã—2 counts: both right, only Jev right, only Luna right, both wrong
- agreement, shared mistakes, and both-wrong items with different labels
- accuracy gap `delta = accuracy(Jev) - accuracy(gpt-6-luna)`
- Miller paired 95% CI, using the sample standard deviation of the per-item correctness difference, and the unpaired SE beside it
- exact two-sided McNemar test (`scipy.stats.binomtest` on the discordant counts)
- TOST at the chosen margin: `p_lower`, `p_upper`, `p_TOST`, and whether the 90% paired CI lies inside the margin
- ECE (10 equal-width bins on the predicted-label probability), multiclass Brier, and log-loss (`-log(p_gold)`, floored at `1e-12`)
- paired bootstrap 95% CIs (10,000 resamples, seed 0) for the Brier difference, the log-loss difference, and the ECE difference
- Wilcoxon signed-rank tests on the per-item Brier and log-loss differences
- confusion matrices and the most common shared wrong labels

Across the four datasets, Holmâ€“Bonferroni is applied separately to the four McNemar p-values and to the four Brier Wilcoxon p-values.

## Results already in `outputs/`

These are the completed full splits. Refusal rate, missing probabilities, and missing token counts are 0 on every model and dataset. `agree` is the share of items where both models gave the same label.

| Dataset | n | Jev accuracy | gpt-6-luna accuracy | delta | 95% paired CI | McNemar p | Equivalent at Â±0.02 | Agreement |
|---|---:|---:|---:|---:|---|---:|---|---:|
| SST-2 | 872 | 0.9633 | 0.9461 | +0.0172 | [0.0047, 0.0297] | 0.0107 | no | 0.9644 |
| Rotten Tomatoes | 1066 | 0.9353 | 0.9212 | +0.0141 | [0.0005, 0.0277] | 0.0581 | no | 0.9484 |
| Emotion | 2000 | 0.5875 | 0.5660 | +0.0215 | [0.0084, 0.0346] | 0.0016 | no | 0.8630 |
| Financial PhraseBank | 970 | 0.7361 | 0.7784 | âˆ’0.0423 | [âˆ’0.0638, âˆ’0.0208] | 0.0002 | no | 0.8784 |

Holmâ€“Bonferroni at Î± = 0.05 keeps the McNemar results for SST-2, Emotion, and Financial PhraseBank. Rotten Tomatoes does not survive: the exact p-value is 0.058, while the paired normal interval only just clears zero.

Jev's ECE is lower on all four datasets. The ECE bootstrap interval excludes zero on SST-2, Rotten Tomatoes, and Emotion. On Financial PhraseBank it includes zero. The Brier Wilcoxon test survives Holm on Emotion (Jev lower) and on Financial PhraseBank (gpt-6-luna lower). The two sentiment Brier tests do not. On Emotion the largest shared mistakes are joy predicted as sadness (91), anger predicted as sadness (60), and sadness predicted as fear (59). On Financial PhraseBank the largest shared mistake is neutral predicted as positive (123).

Recorded input-token cost for the four full runs:

| Dataset | Jev | gpt-6-luna |
|---|---:|---:|
| SST-2 | $0.01302 | $0.01354 |
| Rotten Tomatoes | $0.01597 | $0.01677 |
| Emotion | $0.03571 | $0.03936 |
| Financial PhraseBank | $0.01662 | $0.01915 |

## CSV

Item rows come first, two per item when both models have answered:

`item_id,model,gold,pred,correct,prob_pred,prob_gold,confidence,latency_ms,input_tokens,cost_usd,error`

`prob_pred` is the probability of the predicted label. `prob_gold` is the probability of the gold label. `cost_usd` is input tokens times the model price. If the response has no `usage.input_tokens`, those two cells are blank.

The full option distribution for each row is in `outputs/<dataset>_probs.jsonl`. Those files are committed with the CSVs so `score.py` and `analyze.py` can be run again from the clone.

After a blank line, the score block is:

`metric,jev-1.13.0,gpt-6-luna`

Rows: `n`, `accuracy`, `macro_f1`, `ece_10bin`, `brier`, `refusal_rate`, `mean_latency_ms`, `p50_latency_ms`, `total_input_tokens`, `total_cost_usd`, `excluded_no_probs`, `input_tokens_missing`, and `exact_agreement`. The agreement value is only in the Jev column. It is the fraction of shared items where both predictions are non-empty and equal.

Errors and refusals count as wrong for accuracy and macro-F1. ECE uses 10 equal-width bins on `prob_pred`, and the last bin includes 1.0. Brier is the mean squared error of the full option distribution against a one-hot gold label. `excluded_no_probs` is the number of rows left out of both ECE and Brier because the full distribution did not come back. `input_tokens_missing` is the number of blank token cells. `total_cost_usd` does not include those rows.

`outputs/analysis_summary.csv` has one row per dataset with the paired statistics. `outputs/analysis_<dataset>.txt` is the readable report, including the 2Ã—2 table, the confusion matrices, and a one-line verdict for each test.

## Resume, retries, and locks

- A row already in the CSV is not sent again. Delete that row first if you want to send it again.
- HTTP 429 and 5xx are retried up to 5 times (6 attempts), using `Retry-After` when it is present, otherwise delays of 1, 2, 4, 8, and 16 seconds (capped at 60). After that, the error is written and the run moves on.
- A connection error is not written. The next run sends that item again.
- A read timeout is not retried, because the server may already have billed it. The error row is written.
- HTTP 401, 403, and other 4xx responses stop the run. The rejected item is not written. The key is not logged.
- Each call reserves an estimated cost before it is sent. The process stops a model when that model's counted spend for this process reaches `--max-cost-usd`. Unsent items stay out of the CSV.
- If the CSV is open in Excel or locked by OneDrive, the writer waits and tells you to close it. Close the file and let the retry finish. Do not start a second run while the first one still has the file open.

Logs go to `logs/run_<timestamp>.log`. Schema mismatches also save the raw response under `logs/`. Bearer tokens are stripped from those files.

## What git keeps

Committed: the Python project, the tests, this README, every file in `outputs/` (the four result CSVs, the probability JSONL files, `analysis_summary.csv`, and the four analysis reports).

Ignored, because they are caches or local secrets: `.env`, `.venv/`, `__pycache__/`, the Hugging Face download under `data/`, Arrow and dataset caches, and `logs/`. The root of the BTP repository ignores every other local folder, so this remote contains `Decision_Models` only.
