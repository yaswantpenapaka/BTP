"""Turn a model Response into one JSONL row, and read rows back."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from eval.datasets import Item
from eval.models import Response

SCHEMA_VERSION = 1


def normalise(probs_raw: dict[str, float] | None, option_ids: list[str]):
    """Probabilities in the dataset's option order. Missing options get 0, then
    the vector is renormalised to sum 1. Returns (probs, notes)."""
    notes = []
    if not probs_raw:
        return None, ["no probabilities"]
    extra = [k for k in probs_raw if k not in option_ids]
    missing = [o for o in option_ids if o not in probs_raw]
    if extra:
        notes.append(f"unknown option keys {extra}")
    if missing:
        notes.append(f"missing options {missing} set to 0")
    vec = [max(0.0, float(probs_raw.get(o, 0.0))) for o in option_ids]
    total = sum(vec)
    if total <= 0:
        return None, notes + ["probabilities sum to 0"]
    if abs(total - 1.0) > 0.01:
        notes.append(f"raw sum {total:.4f} renormalised")
    return {o: v / total for o, v in zip(option_ids, vec)}, notes


def build_row(item: Item, plugin, resp: Response) -> dict:
    questions = {}
    errors = []
    refusal = False
    for q in item.questions:
        a = resp.answers.get(q.name)
        if a is None:
            qerr = resp.error or "no answer"
            questions[q.name] = {"choice": None, "probs": None, "probs_raw": None, "confidence": None,
                                 "api_confidence": None, "gold": q.gold, "correct": None, "error": qerr, "notes": []}
            errors.append(qerr)
            continue
        probs, notes = normalise(a.probs_raw, q.option_ids)
        choice = a.choice
        err = a.error
        if choice is not None and choice not in q.option_ids:
            err = err or f"choice {choice!r} not in options"
            choice = None
        if choice is None and probs and not a.refusal:
            notes.append("choice missing: argmax of probs used")
            choice = max(q.option_ids, key=lambda o: probs[o])
        if probs is None and not err:
            err = "no probabilities"
        conf = max(probs.values()) if probs else None
        correct = None
        if q.gold is not None and choice is not None:
            correct = int(choice == q.gold)
        refusal = refusal or a.refusal
        if err:
            errors.append(f"{q.name}: {err}")
        questions[q.name] = {"choice": choice, "probs": probs, "probs_raw": a.probs_raw, "confidence": conf,
                             "api_confidence": a.api_confidence, "gold": q.gold, "correct": correct,
                             "error": err, "notes": notes}
    qs = list(questions.values())
    answered = all(v["choice"] is not None and v["probs"] is not None for v in qs) and not resp.error
    if item.gradeable and answered:
        correct = int(all(v["correct"] == 1 for v in qs))
    else:
        correct = None
    single = len(qs) == 1
    row = {
        "schema": SCHEMA_VERSION,
        "item_id": item.item_id,
        "dataset": item.dataset,
        "model": plugin.name,
        "model_version": plugin.version,
        "model_reported": resp.model_reported,
        "chosen": qs[0]["choice"] if single else {k: v["choice"] for k, v in questions.items()},
        "probs": qs[0]["probs"] if single else {k: v["probs"] for k, v in questions.items()},
        "confidence": qs[0]["confidence"] if single else _mean([v["confidence"] for v in qs]),
        "gold": qs[0]["gold"] if single else {k: v["gold"] for k, v in questions.items()},
        "correct": correct,
        "answered": answered,
        "error": resp.error or "; ".join(errors),
        "refusal": refusal,
        "latency_ms": resp.latency_ms,
        "input_tokens": resp.input_tokens,
        "cost_usd": resp.cost_usd,
        "prompt_hash": item.prompt_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": questions,
    }
    if item.dataset == "fastdecisions":
        row["fd_pred"] = fd_predicted_labels(item, questions)
    return row


def fd_predicted_labels(item: Item, questions: dict) -> dict:
    pred: dict[str, list[str] | str | None] = {}
    for q in item.questions:
        c = questions[q.name]["choice"]
        if q.kind == "label_yesno":
            pred.setdefault(q.head, [])
            if c == "yes":
                pred[q.head].append(q.label)
        else:
            pred[q.head] = c
    return pred


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def read_rows(path: Path) -> dict[str, dict]:
    """Last row per item_id wins (a --retry-errors rerun appends a newer row)."""
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["item_id"]] = r
    return rows
