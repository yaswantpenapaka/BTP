"""CSV rows, probability JSONL, and the score block. No network calls."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

from clients.common import Parsed
from metrics import score_models
from tasks import MODEL_ORDER, Task

ITEM_COLUMNS = [
    "item_id",
    "model",
    "gold",
    "pred",
    "correct",
    "prob_pred",
    "prob_gold",
    "confidence",
    "latency_ms",
    "input_tokens",
    "cost_usd",
    "error",
]

# exact_agreement stays last. The two rows above it are how the score block
# reports missing probabilities and missing token counts.
SCORE_METRICS = [
    "n",
    "accuracy",
    "macro_f1",
    "ece_10bin",
    "brier",
    "refusal_rate",
    "mean_latency_ms",
    "p50_latency_ms",
    "total_input_tokens",
    "total_cost_usd",
    "excluded_no_probs",
    "input_tokens_missing",
    "exact_agreement",
]


def output_paths(root: Path, dataset: str) -> tuple[Path, Path]:
    folder = root / "outputs"
    return folder / f"{dataset}.csv", folder / f"{dataset}_probs.jsonl"


def validate_split(task: Task, feature_names: tuple[str, ...] | list[str], n_rows: int) -> None:
    names = tuple(feature_names)
    if names != task.feature_labels:
        raise SystemExit(
            f"{task.key}: dataset label names are {list(names)}, "
            f"expected {list(task.feature_labels)}. Refusing to guess a mapping."
        )
    if n_rows != task.expected_n:
        raise SystemExit(
            f"{task.key}: {task.split} split has {n_rows} rows, expected {task.expected_n}."
        )


def gold_from_index(task: Task, label: object) -> str:
    """Map a ClassLabel index through the checked feature order onto the prompt label.

    Rotten Tomatoes is the case this exists for: index 0 is neg and index 1 is
    pos on the Hub, and the frozen prompt says negative and positive.
    """
    if isinstance(label, bool):
        raise SystemExit(f"{task.key}: label {label!r} is not an integer class index.")
    try:
        index = label.__index__()
    except Exception as exc:
        raise SystemExit(f"{task.key}: label {label!r} is not an integer class index.") from exc
    if index < 0 or index >= len(task.prompt_labels):
        raise SystemExit(f"{task.key}: label index {index} is outside {list(task.prompt_labels)}.")
    return task.prompt_labels[index]


def decision_row(
    item_id: str,
    model: str,
    gold: str,
    parsed: Parsed,
    prompt_labels: tuple[str, ...],
) -> dict[str, str]:
    pred = parsed.pred
    error = parsed.error
    if pred and pred not in prompt_labels:
        error = f"choice not in options: {pred}"
        pred = ""
    probabilities = parsed.probabilities or {}
    correct = "1" if pred and pred == gold and not error else "0"
    return {
        "item_id": str(item_id),
        "model": model,
        "gold": gold,
        "pred": pred,
        "correct": correct,
        "prob_pred": _cell(_lookup(probabilities, pred)),
        "prob_gold": _cell(_lookup(probabilities, gold)),
        "confidence": _cell(parsed.confidence),
        "latency_ms": _cell(None if parsed.latency_ms is None else round(float(parsed.latency_ms), 1)),
        "input_tokens": "" if parsed.input_tokens is None else str(int(parsed.input_tokens)),
        "cost_usd": _cell(parsed.cost_usd),
        "error": error,
    }


def read_item_rows(path: Path) -> list[dict[str, str]]:
    """Item rows only. Score-block lines and blank lines are ignored."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [name.strip() for name in (reader.fieldnames or []) if name]
        if not fieldnames:
            return []
        if fieldnames != ITEM_COLUMNS:
            raise SystemExit(
                f"{path} header is {fieldnames}, expected {ITEM_COLUMNS}. Refusing to append to a different file."
            )
        rows = []
        for raw in reader:
            item_id = (raw.get("item_id") or "").strip()
            if not item_id.isdigit():
                continue
            rows.append({column: (raw.get(column) or "").strip() for column in ITEM_COLUMNS})
    return rows


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Last copy of an (item_id, model) pair wins. Earlier copies are dropped."""
    last_index: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        last_index[(row["item_id"], row["model"])] = index
    return [row for index, row in enumerate(rows) if last_index[(row["item_id"], row["model"])] == index]


def done_keys(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(row["item_id"], row["model"]) for row in rows}


def parse_item_row(raw: dict[str, str]) -> dict:
    return {
        "item_id": raw["item_id"],
        "model": raw["model"],
        "gold": raw["gold"],
        "pred": raw["pred"],
        "correct": int(raw["correct"] or 0),
        "prob_pred": _parse_float(raw["prob_pred"]),
        "prob_gold": _parse_float(raw["prob_gold"]),
        "confidence": _parse_float(raw["confidence"]),
        "latency_ms": _parse_float(raw["latency_ms"]),
        "input_tokens": _parse_int(raw["input_tokens"]),
        "cost_usd": _parse_float(raw["cost_usd"]),
        "error": raw["error"],
    }


def read_probabilities(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Last JSONL record for each (item_id, model) wins. Null distributions are dropped."""
    found: dict[tuple[str, str], dict[str, float]] = {}
    if not path.exists():
        return found
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed JSONL line {line_number} in {path}")
                continue
            key = (str(obj.get("item_id")), str(obj.get("model")))
            probabilities = obj.get("probabilities")
            if isinstance(probabilities, dict):
                found[key] = {str(label): float(value) for label, value in probabilities.items()}
            else:
                found.pop(key, None)
    return found


def score_lines(scores: dict, model_order: tuple[str, ...] = MODEL_ORDER) -> list[list[str]]:
    lines = [["metric", *model_order]]
    for name in SCORE_METRICS:
        if name == "exact_agreement":
            value = scores[model_order[0]].get("exact_agreement")
            row = ["exact_agreement", fmt_metric(value)]
            row.extend("" for _ in model_order[1:])
        else:
            row = [name, *[fmt_metric(scores[model].get(name)) for model in model_order]]
        lines.append(row)
    return lines


def rescore_rows(
    item_rows: list[dict[str, str]],
    probabilities: dict[tuple[str, str], dict[str, float]],
    labels: list[str],
    model_order: tuple[str, ...] = MODEL_ORDER,
) -> dict:
    typed = [parse_item_row(row) for row in dedupe_rows(item_rows)]
    return score_models(typed, probabilities, labels, model_order)


def write_table(path: Path, item_rows: list[dict[str, str]], score_rows: list[list[str]] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def action() -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ITEM_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in item_rows:
                writer.writerow({column: row.get(column, "") for column in ITEM_COLUMNS})
            if score_rows is not None:
                handle.write("\n")
                score_writer = csv.writer(handle, lineterminator="\n")
                score_writer.writerows(score_rows)
        os.replace(temporary, path)

    call_with_lock_retry(action, path)


def append_item_row(path: Path, row: dict[str, str]) -> None:
    def action() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = (not path.exists()) or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ITEM_COLUMNS, lineterminator="\n")
            if new_file:
                writer.writeheader()
            writer.writerow({column: row.get(column, "") for column in ITEM_COLUMNS})
            handle.flush()
            os.fsync(handle.fileno())

    call_with_lock_retry(action, path)


def append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"

    def action() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    call_with_lock_retry(action, path)


def call_with_lock_retry(
    action,
    path: Path,
    attempts: int = 15,
    delay_s: float = 2.0,
    sleep=time.sleep,
    echo=print,
):
    last: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except OSError as exc:
            if not _is_lock_error(exc):
                raise
            last = exc
            echo(
                f"Cannot write {path}. Close it if it is open in Excel or locked by OneDrive. "
                f"Retry {attempt}/{attempts}."
            )
            if attempt == attempts:
                break
            sleep(delay_s)
    raise SystemExit(
        f"Gave up writing {path}. Close the file and rerun. "
        "Rows already in the CSV are skipped, so completed calls are not sent again."
    ) from last


def token_note(scores: dict, model_order: tuple[str, ...] = MODEL_ORDER) -> str:
    missing = sum(int(scores[model]["input_tokens_missing"]) for model in model_order)
    excluded = ", ".join(f"{model}={scores[model]['excluded_no_probs']}" for model in model_order)
    return (
        f"Rows missing usage.input_tokens: {missing}. "
        "Those input_tokens and cost_usd cells are blank, and total_cost_usd does not include them. "
        f"Rows left out of ECE and Brier because the full probability distribution was missing: {excluded}."
    )


def _lookup(probabilities: dict[str, float], label: str) -> float | None:
    if not label or label not in probabilities:
        return None
    return float(probabilities[label])


def _cell(value: float | None) -> str:
    if value is None:
        return ""
    return format(float(value), ".12g")


def _parse_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_int(value: str) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def fmt_metric(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:
            return "nan"
        return format(value, ".12g")
    return str(value)


def _is_lock_error(exc: OSError) -> bool:
    if isinstance(exc, PermissionError):
        return True
    return getattr(exc, "winerror", None) in (5, 32, 33)
