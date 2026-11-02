# eval/ — Jev vs Luna (and future models) on the BTP evaluation sets

Run every command from `Decision_Models\` with the venv active. Keys are read only from
environment variables: `TYPESAFE_API_KEY` (Jev, jev-1.13.0) and `OPENAI_API_KEY` (Luna, gpt-6-luna).

```powershell
cd C:\Users\yaswa\OneDrive\BTP\Decision_Models
.\.venv\Scripts\Activate.ps1
$env:TYPESAFE_API_KEY = Read-Host -MaskInput "Jev key"
$env:OPENAI_API_KEY   = Read-Host -MaskInput "OpenAI key"

python -m unittest discover -s eval\tests -t . -v                 # tests, no network
python -m eval.run --limit 3                                       # smoke: 3 items x 3 datasets x 2 models
python -m eval.run --models jev luna --datasets jevbench opendecisions fastdecisions   # full run (resumable)
python -m eval.run --retry-errors                                  # re-send items whose last row errored
python -m eval.analyze                                             # metrics, offline
```

## Evaluation sets
| dataset | file | items |
|---|---|---|
| jevbench | Datasets\JevBench\org\test\test.jsonl | 335 |
| opendecisions | Datasets\OpenDecisions\raw\opendecisions.jsonl (all, unsplit) | 2,352 |
| fastdecisions | Datasets\FastDecisions\test\test.jsonl | 255 |

## Adapter decisions (eval/datasets.py — one adapter per dataset, shared by all models)
- Every question is sent to every model as a single-choice question, options in the dataset's
  original order, option ids as given. `state` is the item's input object (`{"text": ...}` when it is a string).
- **JevBench**: choice -> its 4 options; score -> options "0".."K-1" described by the levels (gold = level index);
  noul -> options `true` / `false`. Gold `unknown` means the fact is absent from the text; it is not an answer
  option of a Noul, so those items have **no usable gold** (correct = null), but they still count for agreement /
  distribution metrics.
- **OpenDecisions**: gold is `expected`; choice options = `labels` with `criteria` descriptions; noul options
  `no`/`yes` (labels order) described by criteria `false`/`true`; score options "0".."K-1".
- **Fast Decisions**: single-label head -> one choice question over its candidate labels (as-is);
  multi-label head -> one yes/no question per candidate label. All questions of an item go in ONE request
  (same for every model). Item correct = 1 only if every head matches exactly (predicted label set = gold set).
  Predicted label sets are saved in `fd_pred`.

## Output rows (eval/outputs/raw/<model>__<dataset>.jsonl)
item_id, dataset, model, model_version, model_reported, chosen, probs (normalised, dataset option order; missing
options set to 0 then renormalised, noted in questions.*.notes), confidence (= max prob), gold, correct (0/1/null),
answered, error, refusal, latency_ms, input_tokens, cost_usd, prompt_hash (hash of the shared item content —
identical across models), timestamp (UTC), questions (per-question detail incl. raw API probabilities and API confidence).
Last row per item_id wins.

Neither API exposes a temperature/seed parameter in the bodies we send, so none is set; both return probabilities
directly (no sampling on our side).

## Metrics (eval/analyze.py, eval/metrics.py)
Per model: accuracy over gold items. Pairwise matrices (cell (Mi,Mj), diagonal '—'), over units both models answered:
agreement, Cohen's kappa (pooled over option ids), Spearman/Pearson intra-instance (across options, units with k>=3,
undefined (constant) vectors skipped, then averaged), Spearman/Pearson cross-instance (on confidence),
TVD, JSD (base 2), KL(Mi||Mj) in bits with epsilon 1e-6 added to every entry and renormalised,
McNemar on 0/1 item correctness: b = Mi right & Mj wrong, c = Mi wrong & Mj right, exact binomial if b+c<25 else
chi-square with continuity correction. No multiple-testing correction.
A Fast Decisions item contributes one unit per question.

## Adding a model
Write a class in eval/models.py with `name`, `version`, `env_var`, `decide(item) -> Response`, register it in
`MODELS` / `MODEL_ORDER`. Local models can return the same Response without HTTP.
