"""Rebuild the score block from the CSV and the probability JSONL. No API calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from records import (
    dedupe_rows,
    output_paths,
    read_item_rows,
    read_probabilities,
    rescore_rows,
    score_lines,
    token_note,
    write_table,
)
from tasks import MODEL_ORDER, TASKS, get_task

ROOT = Path(__file__).resolve().parent


def rescore(dataset: str, root: Path = ROOT) -> dict:
    task = get_task(dataset)
    csv_path, jsonl_path = output_paths(root, task.key)
    if not csv_path.exists():
        raise SystemExit(f"No results file at {csv_path}. Run the dataset before scoring.")
    rows = dedupe_rows(read_item_rows(csv_path))
    if not rows:
        raise SystemExit(f"{csv_path} has no item rows to score.")
    probabilities = read_probabilities(jsonl_path)
    scores = rescore_rows(rows, probabilities, list(task.prompt_labels), MODEL_ORDER)
    lines = score_lines(scores, MODEL_ORDER)
    write_table(csv_path, rows, lines)
    for line in lines:
        print(",".join(line))
    print(token_note(scores, MODEL_ORDER))
    return scores


def main(argv: list[str] | None = None) -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required.")
    parser = argparse.ArgumentParser(description="Recompute scores from saved decisions. No API calls.")
    parser.add_argument("--dataset", required=True, choices=list(TASKS))
    args = parser.parse_args(argv)
    rescore(args.dataset)


if __name__ == "__main__":
    main()
