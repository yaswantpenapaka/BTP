"""Run decision models on the evaluation sets. One JSONL per model per dataset.

    python -m eval.run --models jev luna --datasets jevbench opendecisions fastdecisions
    python -m eval.run --models jev --datasets jevbench --limit 3      # smoke test

Resumable: item_ids already in the output file are skipped. --retry-errors
re-sends items whose last row has an error (the new row is appended; the
analysis uses the last row per item). Keys come only from environment
variables (TYPESAFE_API_KEY for Jev, OPENAI_API_KEY for Luna).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval.datasets import DATASET_ORDER, load_items
from eval.models import MODEL_ORDER, MODELS, StopModel
from eval.records import build_row, read_rows
from safety import redact

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def out_path(out_dir: Path, model: str, dataset: str) -> Path:
    return out_dir / "raw" / f"{model}__{dataset}.jsonl"


def run_one(plugin, dataset: str, items, out_dir: Path, retry_errors=False, workers=1, log=print) -> dict:
    path = out_path(out_dir, plugin.name, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = read_rows(path)
    todo = [i for i in items if i.item_id not in done or (retry_errors and done[i.item_id].get("error"))]
    log(f"[{plugin.name}/{dataset}] {len(items)} items, {len(items) - len(todo)} already done, {len(todo)} to send")
    lock = threading.Lock()
    stats = {"sent": 0, "errors": 0, "stopped": ""}
    stop = threading.Event()

    def work(item):
        if stop.is_set():
            return
        try:
            resp = plugin.decide(item)
        except StopModel as exc:
            stop.set()
            stats["stopped"] = redact(str(exc))
            return
        row = build_row(item, plugin, resp)
        with lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["sent"] += 1
            stats["errors"] += bool(row["error"])
            if stats["sent"] % 50 == 0:
                log(f"[{plugin.name}/{dataset}] {stats['sent']}/{len(todo)}")

    if workers <= 1:
        for item in todo:
            work(item)
            if stop.is_set():
                break
    else:
        with ThreadPoolExecutor(workers) as ex:
            for f in as_completed([ex.submit(work, i) for i in todo]):
                f.result()
    if stats["stopped"]:
        log(f"[{plugin.name}/{dataset}] STOPPED: {stats['stopped']}")
    log(f"[{plugin.name}/{dataset}] wrote {stats['sent']} rows ({stats['errors']} with errors) -> {path}")
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODEL_ORDER), choices=list(MODELS))
    p.add_argument("--datasets", nargs="+", default=list(DATASET_ORDER), choices=list(DATASET_ORDER))
    p.add_argument("--limit", type=int, default=None, help="first N items only (smoke test)")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="parallel requests per model (default 1)")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)

    plugins = {}
    for m in a.models:
        cls = MODELS[m]
        key = os.environ.get(cls.env_var, "")
        if not key.strip():
            print(f"Missing environment variable {cls.env_var} for {m}; skipping {m}.", file=sys.stderr)
            continue
        plugins[m] = cls(key, a.out / "logs")
    if not plugins:
        return 2
    rc = 0
    for d in a.datasets:
        items = load_items(d)
        if a.limit:
            items = items[: a.limit]
        for m, plugin in plugins.items():
            s = run_one(plugin, d, items, a.out, a.retry_errors, a.workers)
            rc = rc or (3 if s["stopped"] else 0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
