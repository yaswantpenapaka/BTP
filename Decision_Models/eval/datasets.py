"""Shared dataset adapters. Every model sees exactly the Item these build.

An Item holds a state (the input object) and one or more Questions. Each
Question is sent to every model as a single-choice question over `options`,
in the dataset's original option order. Gold labels are option ids.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

BTP = Path(__file__).resolve().parents[2]
DATASETS_DIR = BTP / "Datasets"

PATHS = {
    "jevbench": DATASETS_DIR / "JevBench" / "org" / "test" / "test.jsonl",
    "opendecisions": DATASETS_DIR / "OpenDecisions" / "raw" / "opendecisions.jsonl",
    "fastdecisions": DATASETS_DIR / "FastDecisions" / "test" / "test.jsonl",
}
EXPECTED_N = {"jevbench": 335, "opendecisions": 2352, "fastdecisions": 255}
DATASET_ORDER = ("jevbench", "opendecisions", "fastdecisions")


@dataclass(frozen=True)
class Question:
    name: str
    instructions: str
    options: tuple[tuple[str, str], ...]  # (option id, description), original order
    gold: str | None = None  # option id, or None when no usable gold
    kind: str = "choice"  # source type: choice / noul / score / single_label / label_yesno
    head: str = ""  # Fast Decisions task name
    label: str = ""  # Fast Decisions label for a per-label yes/no question

    @property
    def option_ids(self) -> list[str]:
        return [o for o, _ in self.options]


@dataclass
class Item:
    item_id: str
    dataset: str
    state: dict
    questions: list[Question]
    meta: dict = field(default_factory=dict)

    @property
    def gradeable(self) -> bool:
        return all(q.gold is not None for q in self.questions)

    def shared_content(self) -> dict:
        return {
            "state": self.state,
            "questions": [
                {"name": q.name, "instructions": q.instructions, "options": [list(o) for o in q.options]}
                for q in self.questions
            ],
        }

    def prompt_hash(self) -> str:
        blob = json.dumps(self.shared_content(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------- JevBench

JEV_NOUL_OPTIONS = (
    ("true", "Yes, this is true according to the text."),
    ("false", "No, this is not true according to the text."),
)


def jevbench_item(row: dict) -> Item:
    kind = row["type"]
    gold = str(row["gold"])
    if kind == "choice":
        options = tuple((o["id"], o["description"]) for o in row["options"])
    elif kind == "noul":
        # Noul is a yes/no question. 'unknown' gold means the needed fact is absent
        # from the text: it is NOT an answer option, so those items have no usable gold.
        options = JEV_NOUL_OPTIONS
    elif kind == "score":
        options = tuple((str(i), level) for i, level in enumerate(row["levels"]))
    else:
        raise ValueError(f"unknown JevBench type {kind!r}")
    if gold == "unknown":  # any type: the needed fact is absent from the text; not an option
        gold = None
    if gold is not None and gold not in [o for o, _ in options]:
        raise ValueError(f"{row['item_id']}: gold {gold!r} not in options")
    q = Question("answer", row["instructions"], options, gold, kind)
    return Item(row["item_id"], "jevbench", {"text": row["state"]}, [q],
                {"type": kind, "domain": row["domain"], "gold_raw": str(row["gold"])})


# --------------------------------------------------------------- OpenDecisions

def opendecisions_item(row: dict) -> Item:
    question = row["question"]
    kind = question["type"]
    labels = [str(x) for x in row["labels"]]
    criteria = question["criteria"]
    if kind == "choice":
        options = tuple((lab, criteria[lab]) for lab in labels)
    elif kind == "noul":
        # labels are ["no", "yes"]; criteria describe true / false.
        desc = {"yes": criteria["true"], "no": criteria["false"]}
        options = tuple((lab, desc[lab]) for lab in labels)
    elif kind == "score":
        options = tuple((lab, criteria[int(lab)]) for lab in labels)
    else:
        raise ValueError(f"unknown OpenDecisions type {kind!r}")
    gold = row.get("expected")
    gold = None if gold is None else str(gold)
    if gold is not None and gold not in labels:
        raise ValueError(f"{row['id']}: expected {gold!r} not in labels")
    state = row["state"] if isinstance(row["state"], dict) else {"text": row["state"]}
    q = Question("answer", question["instructions"], options, gold, kind)
    return Item(row["id"], "opendecisions", state, [q],
                {"type": kind, "family": row.get("family"), "tier": row.get("tier"),
                 "language": row.get("language"), "target_probs": row.get("target_probs")})


# --------------------------------------------------------------- Fast Decisions

def fastdecisions_item(row: dict) -> Item:
    """Single-label head -> one choice question over its candidate labels (as-is).
    Multi-label head -> one yes/no question per candidate label (gold yes if the
    label is in true_label). Exact match = every question correct."""
    questions: list[Question] = []
    for h, head in enumerate(row["output"]["classifications"]):
        task, labels, truth = head["task"], list(head["labels"]), list(head["true_label"])
        if head.get("multi_label"):
            for j, lab in enumerate(labels):
                questions.append(Question(
                    f"q{h}_{j}",
                    f"Task '{task}' (multi-label). Does the label '{lab}' apply to `input`?",
                    (("yes", f"The label '{lab}' applies."), ("no", f"The label '{lab}' does not apply.")),
                    "yes" if lab in truth else "no", "label_yesno", task, lab))
        else:
            if len(truth) != 1:
                raise ValueError(f"{row['item_id']}: single-label head {task} has {truth}")
            questions.append(Question(
                f"q{h}", f"Task '{task}'. Which label best fits `input`?",
                tuple((lab, lab) for lab in labels), truth[0], "single_label", task))
    heads = [{"task": hd["task"], "labels": hd["labels"], "true_label": hd["true_label"],
              "multi_label": bool(hd.get("multi_label"))} for hd in row["output"]["classifications"]]
    return Item(row["item_id"], "fastdecisions", {"input": row["input"]}, questions,
                {"domain": row["domain"], "heads": heads})


ADAPTERS = {"jevbench": jevbench_item, "opendecisions": opendecisions_item, "fastdecisions": fastdecisions_item}


def load_items(dataset: str, path: Path | None = None, check_n: bool = True) -> list[Item]:
    rows = _read_jsonl(path or PATHS[dataset])
    items = [ADAPTERS[dataset](r) for r in rows]
    ids = [i.item_id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{dataset}: duplicate item_id")
    if check_n and path is None and len(items) != EXPECTED_N[dataset]:
        raise ValueError(f"{dataset}: expected {EXPECTED_N[dataset]} items, found {len(items)}")
    return items
