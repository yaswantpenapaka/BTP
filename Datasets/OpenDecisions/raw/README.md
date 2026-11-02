---
license: cc-by-sa-4.0
pretty_name: OpenDecisions
language:
- en
- pl
- pt
- es
- de
- ja
- da
- it
- fr
- hi
- ar
- zh
- ko
- uk
- el
- tr
- nl
- cs
- fi
- 'no'
- sv
- id
task_categories:
- multiple-choice
- text-classification
tags:
- decision-model
- jev
- jevbench
- calibration
- evaluation
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: test
    path: opendecisions.jsonl
---

# OpenDecisions

OpenDecisions (version 1.0.1) is a validation set of 2,352 Jev-style decisions: a state, a typed question, and its options. A decision
model returns one probability per option. The set is built to compare decision models such as Jev and its open
alternatives. Its composition follows JevBench v1.6 (question types, difficulty tiers, item families and languages),
but it contains no JevBench items. Please do not use for training.

## At a glance

| | |
|---|---|
| Decisions | 2,352 |
| Question types | choice 1,178, yes/no 586, score (ordered levels) 588 |
| Difficulty tiers | easy 443, standard 407, judge 581, hard 921 |
| Languages | English 1,842; 510 items in 21 other languages or mixed-language inputs |
| Labels | computed by code 599; written by an LLM author and confirmed by a labeler panel 1,443; real user requests labeled by the panel 310 |

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("jgeuter/OpenDecisions", split="test")
item = ds[0]
print(item["state"])                     # the input
print(item["question"])                  # type, instructions and criteria (the options)
print(item["labels"], item["expected"])  # the option ids and the correct one
```

A model's predictions go into a JSONL file with one line per item, for example
`{"id": "jb16-usecase-842d6d23b1", "probs": {"lower_volume": 0.02, "same_volume": 0.05, "higher_volume": 0.9, "not_mentioned": 0.03}}`,
and `python score.py predictions.jsonl` scores them (details below).

## Files

| File | Content |
|---|---|
| `opendecisions.jsonl` | the 2,352 items |
| `score.py` | the scorer: Intelligence, Calibration and Capability (Python standard library only) |
| `predict_letters.py` | predictions from any Hugging Face chat model with a letter readout (one forward pass per item) |

## How to evaluate

Write one prediction per item to a JSONL file, `{"id": "<item id>", "probs": {...}}`:

- choice: one probability per option id in `labels`
- yes/no: `{"yes": p, "no": 1 - p}`
- score: one probability per level `"0"`, `"1"`, ...

The probabilities of an item must sum to 1. Items without a prediction count as wrong. Then run:

```
python score.py predictions.jsonl
```

To get predictions from a Hugging Face chat model:

```
pip install torch transformers accelerate
python predict_letters.py --model google/gemma-4-31B-it --out predictions.jsonl
python score.py predictions.jsonl
```

`predict_letters.py` shows each item as a multiple-choice prompt with lettered options, runs the model once and reads
the probabilities of the option letters. `--temperatures` divides the letter logits by one temperature per question
type (choice, yes/no, score). Fit temperatures on your own data, never on OpenDecisions.

As a check of your setup: google/gemma-4-31B-it scores 72.3 without temperatures and 79.3 with
`--temperatures 5.96,7.2444,7.2444` (the temperatures we fitted on separate data). Our paper reports 79.6 for the same
model; we ran many items per batch, which changes the top answer on 0.6% of items.

## Format

One JSON object per line, in JevBench's task format. A short example:

```json
{"id": "jb16-usecase-842d6d23b1", "family": "usecase", "tier": "easy", "language": "en",
 "state": "CRM note, 2025-03-04, account Hillcrest Bakeries\nCall with Dana (purchasing). They open two new shops in May and asked us to double their weekly flour order from the first week of May onward.",
 "question": {"type": "choice",
              "instructions": "What does this note say about Hillcrest's flour orders from May?",
              "criteria": {"lower_volume": "Hillcrest will order less flour than now.",
                           "same_volume": "Hillcrest will keep ordering about the same amount.",
                           "higher_volume": "Hillcrest will order more flour than now.",
                           "not_mentioned": "The note says nothing about future order volume."}},
 "labels": ["lower_volume", "same_volume", "higher_volume", "not_mentioned"],
 "expected": "higher_volume"}
```

| Field | Content |
|---|---|
| `state` | the input: text or a JSON object |
| `question` | `type` (`choice`, `noul` = yes/no, `score`), `instructions`, and `criteria`: one description per option (choice), the meaning of `true` and `false` (yes/no), or the ordered levels (score) |
| `labels`, `expected` | the option ids and the correct one; yes/no items use `no` / `yes`; score items give the correct level as an index from 0 |
| `family`, `tier`, `language`, `use_case`, `topic` | what kind of item it is |
| `group` | items that belong together (pairs that differ in one detail, the same item with reordered options); empty otherwise |
| `label_source` | `programmatic`, `author+panel` or `panel` (see below) |
| `trap_answer` | for authored items, the tempting wrong answer the author built the item around; empty otherwise |
| `target_probs` | for the 129 calibration items, the exact probability of each option implied by the state (`expected` is its most likely option); empty otherwise |
| `panel` | the answers of the three labeler models, their mean probability per option, and whether all three or two of three agreed (`status`) |
| `provenance` | origin and licence of the item, and the reason for its label |

Item IDs (`jb16-...`) keep the set's working name from when it was built.

## Item families

| Family | Items | What it tests |
|---|---:|---|
| llm_router | 694 | routing requests: task type, needed capability, model choice, difficulty |
| multilingual | 422 | decisions on inputs in 21 languages other than English, or mixed |
| fairness | 236 | variants that change only a protected attribute; the answer must stay the same |
| usecase | 213 | everyday business decisions (support, compliance, extraction, and similar) |
| jaggedness | 162 | easy-looking questions where models fail unevenly: counting, decimals vs. version numbers, dates, small sums |
| negation_pair_gen | 160 | pairs whose answer flips because one fact or the question is negated |
| option_order | 120 | the same item with its options reordered |
| perturb_pair | 110 | pairs that change only an irrelevant detail; the answer must stay the same |
| composite_gen | 106 | policies with 2 to 5 rules that need small computations |
| calibration_gen, calibration | 129 | the state fixes the answer distribution exactly (rates among past cases, dice, cards, draws) |

## How the labels were made

- **Computed (599 items).** Written by seeded generators; the label is computed, so it is exact.
- **Authored (1,443 items).** Written by an LLM agent (Claude Opus), with a label and a rationale.
  Kept only if at least two of three open models (Qwen3.8-27B, gpt-oss-120b, GLM-4.7-Flash) agreed with the label:
  all three for 1,273 items, two of three for 170.
- **Real requests (310 items).** User prompts from public datasets (Chatbot Arena 55k 288, Dolly 17, GSM8K 4,
  MATH-500 1), each paired with a routing question. The same panel as above was used for labeling, and a sample was kept only if all three models agreed.
- Items that belong together are kept only if every member passes.
- Every candidate was compared with our model training data and with the 231 public JevBench items; 43 of 3,297
  candidates were dropped for overlap.

## Scoring

`score.py` computes Capability, the mean of Intelligence and Calibration, with one
change for yes/no questions.

- Intelligence: accuracy corrected for chance, per question type and difficulty tier (tier weights: easy 0.1,
  standard 0.2, judge 0.3, hard 0.4; the three question types weigh equally).
- Calibration: an expected-calibration-error score per question type.
- Yes/no answers: "yes" if P(yes) is at least 0.5, otherwise "no".
- Also reported, not part of Capability: for the 129 calibration items, the mean distance between the predicted and
  the true probabilities (`target_probs`), measured as total variation distance (0 = exact, 1 = completely wrong).


## Leaderboard (October 15, 2026)

Capability scores of open and closed decision models, scored with `score.py` (see Scoring above). Each system was evaluated the way its authors serve it: their released weights, server code, prompt, and
temperatures (for Jev, its API). TVD (total variation distance) is the distance to the true probabilities on the 129 calibration
items (lower is better); it is not part of Capability. Differences of less than 2 points are not statistically significant.

| Rank | System | Base model | Capability | Intelligence | Calibration | TVD |
|---:|---|---|---:|---:|---:|---:|
| 1 | [Jeff-1.0-Large (ours)](https://huggingface.co/jgeuter/Jeff-1.0-Large) | Gemma 4 31B | 85.1 | 76.9 | 93.2 | 0.184 |
| 1 | [metask-jev-rain-31B-trained](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained) <sup>1</sup> | Gemma 4 31B | 85.1 | 80.2 | 89.9 | 0.241 |
| 3 | [vLLM-SR d3](https://huggingface.co/vllm-sr/d3) | Qwen3.8-27B | 84.5 | 77.9 | 91.1 | 0.185 |
| 4 | [Amos-1-Large](https://huggingface.co/mukaddamzaid/amos-1-large) | Qwen3.8-27B | 84.1 | 78.5 | 89.6 | 0.191 |
| 5 | [metask-jev-rain-31B-trained, one pass](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained/tree/da56acc0bb607cc214c184ad9f1e9231a0d63dee) <sup>1</sup> | Gemma 4 31B | 84.0 | 77.3 | 90.7 | 0.252 |
| 6 | [Eikos-27B](https://huggingface.co/caiovicentino1/Eikos-27B) | Qwen3.8-27B | 83.1 | 75.7 | 90.5 | 0.141 |
| 7 | [Quyet-1.0-Large](https://huggingface.co/chinhnc/Quyet-1.0-Large) | Gemma 4 31B | 81.9 | 76.7 | 87.0 | 0.189 |
| 8 | [Jeff-1.0-Qwen-27B (ours)](https://huggingface.co/jgeuter/Jeff-1.0-Qwen-27B) | Qwen3.6-27B | 81.6 | 73.5 | 89.6 | 0.199 |
| 9 | [AutoJev-27B](https://huggingface.co/denis-pplx/autojev-27b) | Qwen3.8-27B | 80.5 | 70.8 | 90.2 | 0.208 |
| 10 | Jev 1.13.0 (TypeSafe API) <sup>2</sup> | closed | 80.1 | 71.9 | 88.3 | 0.212 |
| 11 | [deck31b](https://github.com/krishna-gogineni-765/deck31b) | Gemma 4 31B | 78.1 | 76.9 | 79.2 | 0.302 |
| 12 | [RSI-Jev v6.1-VL-27B](https://huggingface.co/shgao/rsi-jev-v6.1-vl-27b) | Qwen3.8-27B | 77.9 | 71.2 | 84.6 | 0.121 |
| 13 | [Wald-4B v2.0](https://huggingface.co/org2ai/Wald-4B/tree/919a9d72a65379271005ed0945bfc72742dad3af) <sup>1</sup> | Qwen3.5-4B | 77.1 | 67.0 | 87.3 | 0.181 |
| 14 | [Torchcast-decision-12B](https://huggingface.co/torchcast-ai/torchcast-decision-12b) | Gemma 4 12B | 75.6 | 69.6 | 81.6 | 0.298 |
| 15 | [Jeff-1.0-Medium (ours)](https://huggingface.co/jgeuter/Jeff-1.0-Medium) | Gemma 4 12B | 75.4 | 65.2 | 85.5 | 0.204 |
| 16 | [Cygnet](https://github.com/blockbrain-ai/cygnet-recipe) | Gemma 4 12B | 75.3 | 65.8 | 84.7 | 0.248 |
| 17 | [Clef](https://huggingface.co/Cloudflare/clef) | Qwen3.8-27B | 74.1 | 65.8 | 82.3 | 0.245 |
| 18 | [Wald-4B v2.1](https://huggingface.co/org2ai/Wald-4B) | Qwen3.5-4B | 74.0 | 60.8 | 87.1 | 0.164 |
| 19 | [Torchcast-decision-4B v1.1](https://huggingface.co/torchcast-ai/torchcast-decision-4b) | Qwen3.5-4B | 73.5 | 63.7 | 83.2 | 0.169 |
| 20 | [Plumb-4B](https://huggingface.co/crh225/plumb-4b) | Qwen3.5-4B | 73.1 | 60.4 | 85.8 | 0.184 |
| 21 | [Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B) | Gemma 4 12B | 72.0 | 68.0 | 76.0 | 0.258 |
| 22 | [Open-Jev-27B v1.1](https://huggingface.co/ZefanCai/Open-Jev-27B-v1.1) | Qwen3.8-27B | 71.3 | 59.0 | 83.7 | 0.244 |
| 23 | [decider-4b v2](https://huggingface.co/Mapika/decider-4b) | Qwen3.5-4B | 70.8 | 56.1 | 85.5 | 0.160 |
| 24 | [Jeff-1.0-Qwen-4B (ours)](https://huggingface.co/jgeuter/Jeff-1.0-Qwen-4B) | Qwen3.5-4B | 67.7 | 54.1 | 81.4 | 0.178 |
| 25 | [SemIf (no training)](https://github.com/TheoLeeCJ/SemIf) | Qwen3.5-4B | 58.9 | 46.8 | 71.1 | 0.183 |

<sup>1</sup> Writes a short reasoning text before answering on some decisions: the current metask service (v3, revision
`2feff076`, October 13) on 137 of the 2,352 items (at most 256 tokens), Wald-4B v2.0 on 557 items (at most 512 tokens).
"metask-jev-rain-31B-trained, one pass" is the same model with its authors' earlier service (revision `da56acc0`,
October 11), which only runs a single forward pass per decision. All other open systems also read each decision in a single
pass.<br>
<sup>2</sup> Closed API.

Some numbers differ from Table 1 of our paper: there, our models are the mean of three training runs (we release one),
and some public systems use temperatures we fitted.

## Limitations

- Authored items may differ in style from real requests, and some labels may be wrong.
- Compared with JevBench v1.6, our easy and standard tiers are easier, and our hard tier is harder.
- Non-English items were written natively by the agents but not checked by native speakers.
- 78% of the items are English.
- All inputs are short (at most about 10,000 characters), so the set does not test long inputs.

## Changes

- **1.0.1** (October 15, 2026): removed stray fields; added `trap_answer`; documented `target_probs`; `score.py`
  also reports the distance to the true probabilities on the calibration items; quick start added; the split is
  named `test` instead of `train`; leaderboard added. Items, labels, and scores are unchanged.

## Licence

CC BY-SA 4.0. Items built from real requests keep the terms of their source (Chatbot Arena 55k: Apache-2.0; Dolly:
CC BY-SA 3.0; GSM8K and MATH-500: MIT); each item names its source and licence in `provenance`. The code (`score.py`,
`predict_letters.py`) is under the MIT License. This dataset is not
affiliated with TypeSafe or Benchmark Heaven.

## Citation

If you find this dataset helpful, please consider citing our paper:
```bibtex
@misc{geuter2026releasinginnerjevsimple,
      title={Releasing the Inner Jev: A Simple Recipe for Open Decision Models}, 
      author={Jonathan Geuter and Andrew Cheng and David Alvarez-Melis},
      year={2026},
      eprint={2610.17205},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.17205}, 
}
```
