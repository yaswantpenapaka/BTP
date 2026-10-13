"""Run Jev and OpenAI Decisions on one frozen classification task.

No API call is made until the keys are present, the dataset labels match, and
either --limit is set or the operator types yes. Completed rows are skipped
on the next run.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from clients.common import Parsed, ResponseSchemaError, StopRun
from clients.jev import JevClient
from clients.openai_decisions import OpenAIDecisionsClient
from records import (
    append_item_row,
    append_jsonl,
    decision_row,
    dedupe_rows,
    done_keys,
    gold_from_index,
    output_paths,
    read_item_rows,
    read_probabilities,
    rescore_rows,
    score_lines,
    token_note,
    validate_split,
    write_table,
)
from safety import Budget, RedactFilter
from tasks import JEV_MODEL, MODEL_ORDER, OPENAI_MODEL, PRICE_PER_TOKEN, TASKS, Task, get_task

ROOT = Path(__file__).resolve().parent
ESTIMATED_INPUT_TOKENS = 1000
SMOKE_TEXT = "The film was dull and too long."

API_FACTORIES = {
    JEV_MODEL: ("TYPESAFE_API_KEY", JevClient),
    OPENAI_MODEL: ("OPENAI_API_KEY", OpenAIDecisionsClient),
}


@dataclass(frozen=True)
class Job:
    item_id: str
    model: str
    text: str
    gold: str


@dataclass
class ExecuteResult:
    completed: int = 0
    unsent: int = 0
    # Connection failures are not written. The next run sends them again.
    not_written: int = 0
    stop_reason: str | None = None


def models_for(flag: str) -> list[str]:
    if flag == "jev":
        return [JEV_MODEL]
    if flag == "openai":
        return [OPENAI_MODEL]
    if flag == "both":
        return list(MODEL_ORDER)
    raise SystemExit(f"Unknown model flag {flag!r}.")


def pending_jobs(
    items: list[tuple[str, str, str]],
    models: list[str],
    done: set[tuple[str, str]],
) -> list[Job]:
    jobs = []
    for item_id, text, gold in items:
        for model in models:
            if (item_id, model) in done:
                continue
            jobs.append(Job(item_id=item_id, model=model, text=text, gold=gold))
    return jobs


def estimate_usd(model: str, n_calls: int) -> float:
    return n_calls * ESTIMATED_INPUT_TOKENS * PRICE_PER_TOKEN[model]


def execute_jobs(
    jobs: list[Job],
    budget: Budget,
    concurrency: int,
    call_fn,
    on_result,
) -> ExecuteResult:
    """Send jobs until the queue is empty, the cap blocks them, or a call must stop the run.

    call_fn runs on worker threads. on_result runs on this thread, after the
    response is in hand, so the CSV write is the resume record.
    """
    pending: deque[Job] = deque(jobs)
    blocked: set[str] = set()
    parked: dict[str, deque[Job]] = defaultdict(deque)
    inflight = {}
    result = ExecuteResult()

    def account(job: Job, parsed: Parsed) -> None:
        if parsed.charge == "release":
            # The server did not accept the request. Leave the item out of the
            # CSV so the next run can send it without a second charge.
            budget.release(job.model)
            result.not_written += 1
            print(
                f"{job.model} item {job.item_id} was not saved ({parsed.error}). "
                "It will be sent again on the next run."
            )
            return
        budget.reconcile(job.model, parsed.cost_usd)
        on_result(job, parsed)
        result.completed += 1

    def take(fut) -> None:
        job = inflight.pop(fut)
        exc = fut.exception()
        if isinstance(exc, ResponseSchemaError):
            actual = None if exc.input_tokens is None else exc.input_tokens * PRICE_PER_TOKEN[job.model]
            parsed = Parsed(
                error=(
                    f"response schema mismatch: {exc} "
                    "Delete this CSV row after fixing the parser if this item should be sent again."
                ),
                input_tokens=exc.input_tokens,
                cost_usd=actual,
                latency_ms=getattr(exc, "latency_ms", None),
            )
            account(job, parsed)
            result.stop_reason = str(exc)
        elif isinstance(exc, StopRun):
            if exc.charge == "release":
                budget.release(job.model)
            else:
                budget.reconcile(job.model, None)
            result.stop_reason = str(exc)
        elif exc is not None:
            # The request may have been accepted. Keep the reservation and stop.
            budget.reconcile(job.model, None)
            result.stop_reason = f"{type(exc).__name__}: {exc}"
        else:
            account(job, fut.result())
        if job.model in blocked and job.model not in budget.stopped:
            blocked.discard(job.model)
            pending.extendleft(reversed(parked[job.model]))
            parked[job.model].clear()

    def drain() -> None:
        while inflight:
            finished, _ = wait(set(inflight), return_when=FIRST_COMPLETED)
            for fut in finished:
                take(fut)

    executor = ThreadPoolExecutor(max_workers=concurrency)
    # Each job is submitted at most once. This bound stops a scheduler bug
    # from sending the same item again inside one process.
    step_limit = len(jobs) * 5 + 10
    steps = 0
    try:
        try:
            while result.stop_reason is None:
                steps += 1
                if steps > step_limit:
                    result.stop_reason = (
                        "Stopped because the scheduler stopped making progress. "
                        "No further requests will be sent."
                    )
                    break
                while len(inflight) < concurrency:
                    job = _next_runnable(pending, blocked, parked)
                    if job is None:
                        break
                    if not budget.try_reserve(job.model):
                        blocked.add(job.model)
                        parked[job.model].append(job)
                        continue
                    inflight[executor.submit(call_fn, job)] = job
                if not inflight:
                    break
                finished, _ = wait(set(inflight), return_when=FIRST_COMPLETED)
                for fut in finished:
                    take(fut)
            if result.stop_reason is not None:
                drain()
        except KeyboardInterrupt:
            print("Stopping after the calls already in flight. No new requests will be sent.")
            drain()
            result.stop_reason = "Interrupted. Rows already written will be skipped on the next run."
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    result.unsent = len(pending) + sum(len(queue) for queue in parked.values())
    return result


def _next_runnable(
    pending: deque[Job],
    blocked: set[str],
    parked: dict[str, deque[Job]],
) -> Job | None:
    skipped: list[Job] = []
    chosen: Job | None = None
    while pending:
        job = pending.popleft()
        if job.model in blocked:
            skipped.append(job)
            continue
        chosen = job
        break
    for job in skipped:
        parked[job.model].append(job)
    return chosen


def require_key(name: str) -> str:
    value = os.environ.get(name, "")
    if not value.strip():
        raise SystemExit(
            f"Missing environment variable {name}. "
            "Set it in this PowerShell session only. Do not write it into a file."
        )
    return value


def build_clients(models: list[str], logs_dir: Path, logger: logging.Logger) -> dict:
    clients = {}
    for model in models:
        env_name, factory = API_FACTORIES[model]
        key = require_key(env_name)
        clients[model] = factory(
            key,
            logs_dir,
            log=lambda message: logger.info("%s", message),
            warn=lambda message: logger.warning("%s", message),
        )
    return clients


def load_items(task: Task, limit: int | None) -> list[tuple[str, str, str]]:
    cache = ROOT / "data" / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache))
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf-home"))
    from datasets import load_dataset

    if task.hf_config:
        dataset = load_dataset(task.hf_path, task.hf_config, split=task.split)
    else:
        dataset = load_dataset(task.hf_path, split=task.split)
    if task.text_column not in dataset.column_names:
        raise SystemExit(
            f"{task.key}: text column {task.text_column!r} is not in {list(dataset.column_names)}."
        )
    feature = dataset.features.get("label")
    names = getattr(feature, "names", None)
    if not names:
        raise SystemExit(f"{task.key}: the label feature has no ClassLabel names. Refusing to guess.")
    validate_split(task, tuple(names), len(dataset))
    count = len(dataset) if limit is None else min(limit, len(dataset))
    items = []
    for index in range(count):
        row = dataset[int(index)]
        text = row[task.text_column]
        if not isinstance(text, str):
            raise SystemExit(f"{task.key}: row {index} text is not a string.")
        items.append((str(index), text, gold_from_index(task, row["label"])))
    return items


def confirm_unlimited(task: Task, n_items: int, remaining: dict[str, int], cap: float) -> None:
    lines = [
        f"Dataset: {task.key}",
        f"Source: {task.hf_path} split={task.split}",
        f"Items in this run: {n_items}",
        "Calls still to make:",
    ]
    total = 0.0
    for model in remaining:
        cost = estimate_usd(model, remaining[model])
        total += cost
        lines.append(
            f"  {model}: {remaining[model]} calls, estimated ${cost:.6f} "
            f"at {ESTIMATED_INPUT_TOKENS} input tokens each (estimate, not a quote)"
        )
    lines.append(f"Estimated total: ${total:.6f}")
    lines.append(f"Cap per model for this process: ${cap:.2f}")
    lines.append("Rows already in the CSV are skipped.")
    print("\n".join(lines))
    if not sys.stdin.isatty():
        raise SystemExit(
            "Refusing to run without --limit because stdin is not a terminal. No requests were sent."
        )
    typed = input("Type yes to continue: ")
    if typed.strip() != "yes":
        raise SystemExit("Aborted. No requests were sent.")


def make_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"run_{stamp}.log"
    logger = logging.getLogger("jev_runs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RedactFilter())
    stream = logging.StreamHandler()
    stream.setLevel(logging.WARNING)
    stream.setFormatter(formatter)
    stream.addFilter(RedactFilter())
    logger.addHandler(file_handler)
    logger.addHandler(stream)
    logger.info("log file %s", path)
    return logger


def run_smoke(models: list[str], clients: dict, task: Task) -> None:
    print(f"Smoke sentence: {SMOKE_TEXT}")
    print("Expected choice: negative")
    failures = []
    for model in models:
        print(f"\n{model}")
        try:
            parsed = clients[model].decide(task, SMOKE_TEXT)
        except (ResponseSchemaError, StopRun) as exc:
            failures.append(str(exc))
            print(exc)
            continue
        _print_parsed(parsed)
        if parsed.pred != "negative":
            print("  warning: expected choice negative")
        if parsed.error:
            failures.append(f"{model}: {parsed.error}")
    if failures:
        raise SystemExit(1)


def _print_parsed(parsed: Parsed) -> None:
    print(f"  choice: {parsed.pred or '(none)'}")
    if parsed.probabilities:
        parts = " ".join(f"{label}={value:.6g}" for label, value in parsed.probabilities.items())
    else:
        parts = "(none)"
    print(f"  probabilities: {parts}")
    confidence = "(none)" if parsed.confidence is None else format(parsed.confidence, ".6g")
    tokens = "(not reported)" if parsed.input_tokens is None else str(parsed.input_tokens)
    cost = "(not reported)" if parsed.cost_usd is None else format(parsed.cost_usd, ".8g")
    print(f"  confidence: {confidence}")
    print(f"  input_tokens: {tokens}")
    print(f"  cost_usd: {cost}")
    if parsed.error:
        print(f"  error: {parsed.error}")


def run_dataset(task: Task, models: list[str], clients: dict, args, logger: logging.Logger) -> None:
    items = load_items(task, args.limit)
    csv_path, jsonl_path = output_paths(ROOT, task.key)
    existing = dedupe_rows(read_item_rows(csv_path)) if csv_path.exists() else []
    done = done_keys(existing)
    jobs = pending_jobs(items, models, done)
    remaining = {model: sum(1 for job in jobs if job.model == model) for model in models}
    print(f"{task.key}: {len(items)} items, {len(jobs)} calls still to make.")
    for model, count in remaining.items():
        print(f"  {model}: {count}")
    if not jobs:
        print("Nothing to send. Refreshing the score block from the rows already on disk.")
        _write_scores(task, csv_path, jsonl_path)
        return
    if args.limit is None:
        confirm_unlimited(task, len(items), remaining, args.max_cost_usd)
    budget = Budget(
        max_usd=args.max_cost_usd,
        reserve_usd={model: estimate_usd(model, 1) for model in models},
    )
    logger.info(
        "dataset %s items %s calls %s cap_usd %s",
        task.key,
        len(items),
        len(jobs),
        args.max_cost_usd,
    )

    def call_fn(job: Job) -> Parsed:
        return clients[job.model].decide(task, job.text)

    from tqdm import tqdm

    bar = tqdm(total=len(jobs), desc=task.key, unit="call")

    def on_result(job: Job, parsed: Parsed) -> None:
        row = decision_row(job.item_id, job.model, job.gold, parsed, task.prompt_labels)
        append_item_row(csv_path, row)
        append_jsonl(
            jsonl_path,
            {"item_id": job.item_id, "model": job.model, "probabilities": parsed.probabilities},
        )
        logger.info(
            "item %s model %s pred %s error %s tokens %s cost %s latency_ms %s",
            job.item_id,
            job.model,
            parsed.pred,
            parsed.error,
            parsed.input_tokens,
            parsed.cost_usd,
            parsed.latency_ms,
        )
        bar.update(1)

    try:
        result = execute_jobs(jobs, budget, args.concurrency, call_fn, on_result)
    finally:
        bar.close()
    _write_scores(task, csv_path, jsonl_path)
    if result.not_written:
        print(
            f"{result.not_written} calls were not accepted and were not written. "
            "Re-run the same command to send them."
        )
    if result.unsent and result.stop_reason is None:
        print(
            f"Stopped with {result.unsent} calls unsent because a model reached "
            f"the ${args.max_cost_usd:.2f} cap. Re-run the same command to resume."
        )
        for model, spent in budget.spent.items():
            print(f"  {model}: ${spent:.6f} counted in this process")
    if result.stop_reason:
        logger.error("%s", result.stop_reason)
        raise SystemExit(1)


def _write_scores(task: Task, csv_path: Path, jsonl_path: Path) -> None:
    rows = dedupe_rows(read_item_rows(csv_path))
    if not rows:
        print("No item rows to score.")
        return
    scores = rescore_rows(rows, read_probabilities(jsonl_path), list(task.prompt_labels), MODEL_ORDER)
    lines = score_lines(scores, MODEL_ORDER)
    write_table(csv_path, rows, lines)
    print(f"\nWrote {csv_path}")
    for line in lines:
        print(",".join(line))
    print(token_note(scores, MODEL_ORDER))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Jev and OpenAI Decisions on one classification dataset."
    )
    parser.add_argument("--smoke", action="store_true", help="One call per API on a fixed negative sentence.")
    parser.add_argument("--dataset", choices=list(TASKS))
    parser.add_argument("--model", choices=("jev", "openai", "both"), default="both")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N items.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-cost-usd", type=float, default=2.0, dest="max_cost_usd")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required.")
    args = parse_args(argv)
    if args.smoke and args.dataset:
        raise SystemExit("Use either --smoke or --dataset, not both. No requests were sent.")
    if not args.smoke and not args.dataset:
        raise SystemExit("Pass --smoke or --dataset. No requests were sent.")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be a positive integer. No requests were sent.")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be a positive integer. No requests were sent.")
    if args.max_cost_usd <= 0:
        raise SystemExit("--max-cost-usd must be positive. No requests were sent.")
    if args.smoke and args.limit is not None:
        raise SystemExit("--limit does not apply to --smoke. No requests were sent.")

    logs_dir = ROOT / "logs"
    logger = make_logger(logs_dir)
    models = models_for(args.model)
    # Check every selected key before the first paid call.
    clients = build_clients(models, logs_dir, logger)
    if args.smoke:
        run_smoke(models, clients, get_task("sst2"))
        return
    run_dataset(get_task(args.dataset), models, clients, args, logger)


if __name__ == "__main__":
    main()
