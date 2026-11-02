"""Laya (local, CPU) plug-in + training-data builder + runner.

    python -m eval.laya build-data                       # train/val JSONL for laya.finetune
    python -m eval.laya run --ckpt checkpoints/laya-base --tag base --datasets opendecisions [--limit 5]

Every eval Question is sent to Laya as a `choice` question whose criteria are the
dataset options in the SAME order the Jev/Luna adapters use, so item_id, option
order and prompt_hash all match the existing raw files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.datasets import ADAPTERS, DATASETS_DIR, Item, _read_jsonl, load_items
from eval.models import Answer, Response

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "checkpoints" / "laya_data"
SPLITS = {
    "jevbench": {s: DATASETS_DIR / "JevBench" / "org" / s / f"{s}.jsonl" for s in ("train", "validation")},
    "fastdecisions": {s: DATASETS_DIR / "FastDecisions" / s / f"{s}.jsonl" for s in ("train", "validation")},
}


def laya_questions(item: Item) -> dict:
    return {q.name: {"type": "choice", "instructions": q.instructions,
                     "criteria": {o: (d or o) for o, d in q.options}} for q in item.questions}


class LayaPlugin:
    name = "laya"
    env_var = ""

    def __init__(self, ckpt: str, tag: str):
        import laya
        self.version = f"laya-{tag}"
        _p = Path(ckpt)
        if _p.exists():
            ckpt = str(_p.resolve())
        self.agent = laya.load(ckpt, device="cpu")

    def decide(self, item: Item) -> Response:
        t0 = time.perf_counter()
        try:
            out = self.agent.predict(item.state, laya_questions(item))
        except Exception as exc:
            return Response(latency_ms=(time.perf_counter() - t0) * 1000, error=f"laya: {exc}")
        resp = Response(latency_ms=(time.perf_counter() - t0) * 1000, model_reported=self.version)
        answers = out.get("answers", {})
        resp.input_tokens = (out.get("usage") or {}).get("input_tokens")
        for q in item.questions:
            a = answers.get(q.name) or {}
            p = a.get("probabilities")
            p = {str(k): float(v) for k, v in p.items()} if isinstance(p, dict) else None
            c = a.get("choice")
            resp.answers[q.name] = Answer(choice=c if isinstance(c, str) else None, probs_raw=p,
                                          api_confidence=a.get("confidence"),
                                          error="" if p else "no probabilities")
        return resp


def build_data():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    for d, splits in SPLITS.items():
        for s, path in splits.items():
            n_rows = n_q = 0
            with open(TRAIN_DIR / f"{d}_{s}.jsonl", "w", encoding="utf-8") as fh:
                for r in _read_jsonl(path):
                    it = ADAPTERS[d](r)
                    qs = {q.name: lq for q, lq in zip(it.questions, laya_questions(it).values()) if q.gold is not None}
                    if not qs:
                        continue
                    gold = {q.name: {"probabilities": {o: float(o == q.gold) for o in q.option_ids}}
                            for q in it.questions if q.gold is not None}
                    fh.write(json.dumps({"item_id": it.item_id, "state": it.state, "questions": qs, "gold": gold},
                                        ensure_ascii=False) + "\n")
                    n_rows += 1; n_q += len(qs)
            print(f"{d}/{s}: {n_rows} rows, {n_q} questions")


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-data")
    r = sub.add_parser("run")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--datasets", nargs="+", required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "build-data":
        return build_data()
    from eval.run import OUT_DIR, run_one
    plugin = LayaPlugin(a.ckpt, a.tag)
    for d in a.datasets:
        items = load_items(d)
        if a.limit:
            items = items[: a.limit]
        run_one(plugin, d, items, a.out or OUT_DIR)


if __name__ == "__main__":
    main()


