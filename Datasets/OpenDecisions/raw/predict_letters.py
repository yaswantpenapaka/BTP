"""Zero-shot letter readout of a Hugging Face chat model on OpenDecisions; writes predictions for score.py.

    pip install torch transformers accelerate
    python predict_letters.py --model google/gemma-4-31B-it --out predictions.jsonl
    python predict_letters.py --model google/gemma-4-31B-it --out predictions.jsonl --temperatures 5.96,7.24,7.24
    python score.py predictions.jsonl

Each item becomes a multiple-choice prompt with lettered options (the prompt of SemIf, github.com/TheoLeeCJ/SemIf):
a fixed system message and a JSON user message {"evidence": state, "criterion": question, "options": [{"letter",
"description"}]}, rendered with the model's chat template with thinking disabled. One forward pass gives the
next-token logits of the option letters A, B, C, ...; a softmax over the item's K letters gives the probabilities.
Option descriptions are "<id>: <text>"; yes/no questions have the options "true" (listed first) and "false"; score
levels are listed from lowest to highest. This is how JevBench's semif_direct adapter maps a question
(github.com/fstandhartinger/jevbench, MIT).

--temperatures choice,yesno,score divides the letter logits by one temperature per question type (default 1, 1, 1).
Our paper fits them by NLL on separate data, never below 1 (for example 5.96, 7.24, 7.24 for Gemma 4 31B).
Items above --max-tokens prompt tokens (default 8192) are skipped and count as wrong in score.py. MIT License.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
          "Respond with only its uppercase letter, with no explanation or reasoning.")
LETTERS = "ABCDEFGHIJKLMNOP"


def options(question: dict) -> list[dict]:
    """Option ids and descriptions of one question, in prompt order."""
    qtype, criteria = question["type"], question["criteria"]
    if qtype == "noul":
        opts = [(key, criteria.get(key, f"The proposition is {key}.")) for key in ("true", "false")]
    elif qtype == "choice" and isinstance(criteria, dict):
        opts = [(key, text or key) for key, text in criteria.items()]
    elif qtype == "choice":                           # a bare list of option ids
        opts = [(str(key), str(key)) for key in criteria]
    else:
        opts = [(str(i), text) for i, text in enumerate(criteria)]
    return [{"id": key, "description": f"{key}: {text}"} for key, text in opts]


def messages(item: dict) -> tuple[list[dict], list[str]]:
    opts = options(item["question"])
    if len(opts) > len(LETTERS):
        raise ValueError(f"{item['id']}: more than {len(LETTERS)} options")
    payload = {"evidence": item["state"], "criterion": item["question"]["instructions"],
               "options": [{"letter": LETTERS[i], "description": o["description"]} for i, o in enumerate(opts)]}
    return ([{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            [o["id"] for o in opts])


def letter_ids(tokenizer) -> list[int]:
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"the letter {letter!r} is not a single token for this tokenizer")
        ids.append(encoded[0])
    return ids


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "opendecisions.jsonl"))
    parser.add_argument("--temperatures", default="1,1,1", help="choice,yesno,score")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None, help="only the first N items (for a quick test)")
    args = parser.parse_args()
    temps = dict(zip(("choice", "noul", "score"), (float(x) for x in args.temperatures.split(","))))

    torch.backends.cuda.enable_cudnn_sdp(False)      # as in all our evaluations
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    letters = letter_ids(tokenizer)
    with open(args.data, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()][:args.limit]
    written = skipped = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for item in items:
            msgs, ids = messages(item)
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) > args.max_tokens:
                skipped += 1
                continue
            with torch.inference_mode():
                logits = model(input_ids=torch.tensor([tokens], device=model.device), use_cache=False,
                               logits_to_keep=1).logits[0, -1].float()
            qtype = item["question"]["type"]
            probs = torch.softmax(logits[letters[:len(ids)]] / temps[qtype], dim=-1).tolist()
            by_id = dict(zip(ids, probs))
            if qtype == "noul":
                by_id = {"yes": by_id["true"], "no": by_id["false"]}
            out.write(json.dumps({"id": item["id"], "probs": by_id}) + "\n")
            written += 1
    print(f"wrote {written} predictions to {args.out}; skipped {skipped} items above {args.max_tokens} tokens")


if __name__ == "__main__":
    main()
