"""Map frozen BTP Items through the official semif-score CLI.

This does not score logits. It writes SemIf input rows, runs
`semif-score --mode direct`, and turns that output into eval `build_row`
records. Questions with fewer than 2 or more than 16 options are not sent.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BTP = Path("/userhome/btech/p.yaswant/BTP")
sys.path.insert(0, str(BTP / "Decision_Models"))

from eval.datasets import EXPECTED_N, load_items  # noqa: E402
from eval.models import Answer, Response  # noqa: E402
from eval.records import build_row, read_rows  # noqa: E402

MIN_OPTIONS = 2
MAX_OPTIONS = 16
DATASETS = ("jevbench", "opendecisions", "fastdecisions")


class _Plugin:
    name = "semif"

    def __init__(self, version: str):
        self.version = version


def semif_row_id(item_id: str, question_name: str) -> str:
    return f"{item_id}::{question_name}"


def split_semif_row_id(row_id: str) -> tuple[str, str]:
    item_id, name = row_id.rsplit("::", 1)
    return item_id, name


def option_error(count: int) -> str:
    return f"semif_unsupported_option_count:{count}"


def questions_for(item) -> tuple[list[dict], dict[str, str]]:
    """Return (semif rows, question_name -> error) for one item."""
    rows = []
    errors = {}
    for question in item.questions:
        count = len(question.options)
        if count < MIN_OPTIONS or count > MAX_OPTIONS:
            errors[question.name] = option_error(count)
            continue
        rows.append({
            "id": semif_row_id(item.item_id, question.name),
            "state": item.state,
            "question": question.instructions,
            "options": [{"id": oid, "description": desc} for oid, desc in question.options],
        })
    return rows, errors


def _probs_and_choice(question, result: dict) -> tuple[str | None, dict[str, float] | None, str]:
    option_ids = result.get("option_ids")
    probabilities = result.get("probabilities")
    if not isinstance(option_ids, list) or not isinstance(probabilities, list):
        return None, None, "semif_missing_probabilities"
    if len(option_ids) != len(probabilities):
        return None, None, f"semif_prob_length:{len(probabilities)}!={len(option_ids)}"
    raw = {}
    for oid, prob in zip(option_ids, probabilities):
        if not isinstance(oid, str):
            return None, None, "semif_option_id_not_string"
        try:
            raw[oid] = float(prob)
        except (TypeError, ValueError):
            return None, None, "semif_probability_not_float"
    missing = [oid for oid in question.option_ids if oid not in raw]
    if missing:
        return None, None, f"semif_missing_options:{missing}"
    # Dataset order. Python max keeps the first id when scores tie.
    choice = max(question.option_ids, key=lambda oid: raw[oid])
    return choice, {oid: raw[oid] for oid in question.option_ids}, ""


def response_for(item, scored: dict[str, dict], errors: dict[str, str], model_reported: str) -> Response:
    answers = {}
    tokens = 0
    seconds = 0.0
    for question in item.questions:
        if question.name in errors:
            answers[question.name] = Answer(error=errors[question.name])
            continue
        result = scored.get(question.name)
        if result is None:
            answers[question.name] = Answer(error="semif_missing_output")
            continue
        choice, raw, err = _probs_and_choice(question, result)
        answers[question.name] = Answer(choice=choice, probs_raw=raw, error=err)
        if isinstance(result.get("input_tokens"), int):
            tokens += result["input_tokens"]
        if isinstance(result.get("total_seconds"), (int, float)):
            seconds += float(result["total_seconds"])
    reported = model_reported
    for result in scored.values():
        model = result.get("model")
        if isinstance(model, dict) and model.get("source") and model.get("revision"):
            reported = f"{model['source']}@{model['revision']}"
            break
    return Response(
        answers=answers,
        latency_ms=seconds * 1000 if scored else None,
        input_tokens=tokens or None,
        model_reported=reported,
    )


def _load_scored(path: Path) -> dict[str, dict]:
    scored: dict[str, dict] = {}
    if not path.exists():
        return scored
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
        scored[row["id"]] = row
    return scored


def _failed_row_id(stderr: str, candidates: list[str]) -> str | None:
    hits = [row_id for row_id in candidates if f"Row {row_id}:" in stderr]
    if not hits:
        return None
    hits.sort(key=len, reverse=True)
    return hits[0]


def _run_semif(args, dataset: str, rows: list[dict], work: Path) -> tuple[Path, str]:
    """Score rows. On a mid-file failure, keep the flushed lines and return stderr."""
    stamp = time.strftime("%Y%m%dT%H%M%S")
    source = work / f"semif_inputs_{dataset}_{stamp}.jsonl"
    output = work / f"semif_raw_{dataset}_{stamp}.jsonl"
    with source.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    command = [
        args.semif_score,
        "--mode", "direct",
        "--model", args.model,
        "--revision", args.revision,
        "--input", str(source),
        "--output", str(output),
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    print(f"=== semif-score {dataset} rows={len(rows)} ===", flush=True)
    print(" ".join(command), flush=True)
    import threading
    proc = subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_parts: list[str] = []
    err_parts: list[str] = []

    def _pump(stream, sink, parts: list[str]) -> None:
        for line in stream:
            parts.append(line)
            sink.write(line)
            sink.flush()

    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, sys.stdout, out_parts)),
        threading.Thread(target=_pump, args=(proc.stderr, sys.stderr, err_parts)),
    ]
    for thread in threads:
        thread.start()
    return_code = proc.wait()
    for thread in threads:
        thread.join()
    if return_code != 0:
        print(f"semif-score exit {return_code} dataset={dataset}", file=sys.stderr)
        return output, "".join(err_parts)
    return output, ""


def score_dataset(args, dataset: str, plugin: _Plugin, dest: Path, work: Path) -> dict:
    items = load_items(dataset)
    if len(items) != EXPECTED_N[dataset]:
        raise SystemExit(f"{dataset}: expected {EXPECTED_N[dataset]} items, found {len(items)}")
    done = read_rows(dest)
    pending = [item for item in items if item.item_id not in done]
    print(f"{dataset}: items={len(items)} already={len(done)} pending={len(pending)}", flush=True)
    tally = {"unsupported_option_count": 0, "scored_questions": 0, "missing_or_failed": 0}
    by_count: dict[int, int] = {}

    inputs = []
    blocked: dict[str, dict[str, str]] = {}
    for item in pending:
        rows, errors = questions_for(item)
        blocked[item.item_id] = errors
        for err in errors.values():
            tally["unsupported_option_count"] += 1
            count = int(err.rsplit(":", 1)[1])
            by_count[count] = by_count.get(count, 0) + 1
        inputs.extend(rows)

    scored_by_id: dict[str, dict] = {}
    remaining = inputs
    model_reported = f"{args.model}@{args.revision}"
    while remaining:
        output, stderr = _run_semif(args, dataset, remaining, work)
        produced = _load_scored(output)
        scored_by_id.update(produced)
        if not stderr:
            missing = [row for row in remaining if row["id"] not in produced]
            if missing:
                raise SystemExit(f"{dataset}: semif-score returned no error but omitted {len(missing)} rows")
            break
        if all(row["id"] in produced for row in remaining):
            print(f"{dataset}: semif-score wrote every row but exited non-zero; keeping the rows", flush=True)
            break
        failed_id = _failed_row_id(stderr, [row["id"] for row in remaining])
        if failed_id is None:
            if not produced:
                raise SystemExit(f"{dataset}: semif-score failed before writing a row\n{stderr}")
            print(f"{dataset}: salvaged {len(produced)} rows after semif-score failure", flush=True)
            break
        item_id, question_name = split_semif_row_id(failed_id)
        message = next((line.strip() for line in stderr.splitlines() if f"Row {failed_id}:" in line), "semif-score failed")
        blocked.setdefault(item_id, {})[question_name] = f"semif_score_error:{message}"
        tally["missing_or_failed"] += 1
        remaining = [row for row in remaining if row["id"] != failed_id and row["id"] not in produced]
        print(f"{dataset}: recorded failure for {failed_id}; remaining={len(remaining)}", flush=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("a", encoding="utf-8") as handle:
        for item in pending:
            per_question = {}
            for question in item.questions:
                row_id = semif_row_id(item.item_id, question.name)
                if row_id in scored_by_id:
                    per_question[question.name] = scored_by_id[row_id]
            errors = blocked.get(item.item_id, {})
            # A killed scorer leaves some questions absent. Skip the item so
            # a resubmit retries it, unless every question is already resolved.
            unresolved = [
                question.name for question in item.questions
                if question.name not in errors and question.name not in per_question
            ]
            if unresolved:
                tally["missing_or_failed"] += len(unresolved)
                print(f"{dataset}: leave {item.item_id} for resubmit; missing {unresolved}", flush=True)
                continue
            row = build_row(item, plugin, response_for(item, per_question, errors, model_reported))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            tally["scored_questions"] += len(per_question)
    print(f"{dataset}: appended {written} eval rows -> {dest}", flush=True)
    tally["by_count"] = by_count
    tally["eval_rows_appended"] = written
    tally["eval_rows_total"] = len(read_rows(dest))
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--semif-commit", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--raw-dest", type=Path, required=True)
    parser.add_argument("--semif-score", default="semif-score")
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    short = args.revision[:12]
    plugin = _Plugin(f"semif-{args.semif_commit}+{short}")
    summary = {"semif_commit": args.semif_commit, "revision": args.revision, "model": args.model, "datasets": {}}
    exit_code = 0
    for dataset in args.datasets:
        dest = args.raw_dest / f"semif__{dataset}.jsonl"
        try:
            summary["datasets"][dataset] = score_dataset(args, dataset, plugin, dest, args.work)
        except SystemExit as exc:
            summary["datasets"][dataset] = {"error": str(exc)}
            print(exc, file=sys.stderr)
            exit_code = 1
            break
        got = summary["datasets"][dataset]["eval_rows_total"]
        if got != EXPECTED_N[dataset]:
            print(f"{dataset}: have {got} of {EXPECTED_N[dataset]}", file=sys.stderr)
            exit_code = 1
            break
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
