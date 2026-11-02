"""Model plug-ins. A plug-in turns a shared Item into a request, calls the
API, and returns one Answer per Question. Add a new model by writing a class
with name / version / env_var / decide(item) and registering it in MODELS.

The HTTP layer (retries with backoff on 429/5xx/connection errors, stop on
401/403/4xx, key only in the Authorization header) is reused from
Decision_Models/clients/transport.py.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from clients.common import read_input_tokens, save_raw
from clients.jev import JEV_URL
from clients.openai_decisions import OPENAI_URL
from clients.transport import post_json
from tasks import JEV_MODEL, OPENAI_MODEL, PRICE_PER_TOKEN

from eval.datasets import Item


@dataclass
class Answer:
    choice: str | None = None
    probs_raw: dict[str, float] | None = None
    api_confidence: float | None = None
    error: str = ""
    refusal: bool = False


@dataclass
class Response:
    answers: dict[str, Answer] = field(default_factory=dict)
    latency_ms: float | None = None
    input_tokens: int | None = None
    cost_usd: float | None = None
    model_reported: str | None = None
    error: str = ""  # request-level error (whole item failed)


class StopModel(RuntimeError):
    """API rejected the request/key: stop this model, write no row."""


def _num(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _map(raw) -> dict[str, float] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    out = {}
    for k, v in raw.items():
        n = _num(v)
        if n is None:
            return None
        out[str(k)] = n
    return out


def _list(raw) -> dict[str, float] | None:
    if not isinstance(raw, list) or not raw:
        return None
    out = {}
    for e in raw:
        if not isinstance(e, dict) or _num(e.get("probability")) is None:
            return None
        out[str(e.get("value"))] = _num(e.get("probability"))
    return out


class _HttpPlugin:
    name = ""
    version = ""
    env_var = ""
    url = ""

    def __init__(self, api_key: str, logs_dir: Path, log=print, session=None, sleep=None):
        if not str(api_key).strip():
            raise ValueError("empty API key")
        self._key = api_key
        self.logs_dir = logs_dir
        self._log = log
        self._session = session
        self._sleep = sleep or time.sleep

    def __repr__(self) -> str:  # never show the key
        return f"{type(self).__name__}(model={self.version})"

    def build_body(self, item: Item) -> dict:
        raise NotImplementedError

    def parse(self, item: Item, payload: dict) -> dict[str, Answer]:
        raise NotImplementedError

    def decide(self, item: Item) -> Response:
        out = post_json(self.url, self._key, self.build_body(item), api_name=self.name,
                        logs_dir=self.logs_dir, log=self._log, session=self._session, sleep=self._sleep)
        if out.kind == "stop":
            raise StopModel(out.message)
        if out.kind in ("error", "schema"):
            return Response(latency_ms=out.latency_ms, error=out.message or out.kind)
        payload = out.payload
        resp = Response(latency_ms=out.latency_ms)
        resp.input_tokens = read_input_tokens(payload)
        if resp.input_tokens is not None:
            resp.cost_usd = resp.input_tokens * PRICE_PER_TOKEN[self.version]
        m = payload.get("model")
        resp.model_reported = m if isinstance(m, str) else None
        try:
            resp.answers = self.parse(item, payload)
        except Exception as exc:  # unexpected shape: keep raw on disk, mark item
            path = save_raw(self.logs_dir, self.name, payload)
            resp.error = f"schema: {exc} (raw saved to {path.name})"
        return resp


class JevPlugin(_HttpPlugin):
    name = "jev"
    version = JEV_MODEL
    env_var = "TYPESAFE_API_KEY"
    url = JEV_URL

    def build_body(self, item: Item) -> dict:
        return {
            "model": JEV_MODEL,
            "state": item.state,
            "questions": {
                q.name: {"type": "choice", "instructions": q.instructions, "criteria": dict(q.options)}
                for q in item.questions
            },
        }

    def parse(self, item: Item, payload: dict) -> dict[str, Answer]:
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("missing answers object")
        out = {}
        for q in item.questions:
            a = answers.get(q.name)
            if not isinstance(a, dict):
                out[q.name] = Answer(error="missing answer")
                continue
            choice = a.get("choice")
            out[q.name] = Answer(
                choice=choice if isinstance(choice, str) and choice else None,
                probs_raw=_map(a.get("probabilities")),
                api_confidence=_num(a.get("confidence")),
                error="" if isinstance(choice, str) and choice else "missing choice",
            )
        return out


class LunaPlugin(_HttpPlugin):
    name = "luna"
    version = OPENAI_MODEL
    env_var = "OPENAI_API_KEY"
    url = OPENAI_URL

    def build_body(self, item: Item) -> dict:
        return {
            "model": OPENAI_MODEL,
            "input": json.dumps(item.state, ensure_ascii=False),
            "questions": [
                {"type": "choice", "name": q.name, "instructions": q.instructions,
                 "choices": [{"value": o, "description": d} for o, d in q.options]}
                for q in item.questions
            ],
        }

    def parse(self, item: Item, payload: dict) -> dict[str, Answer]:
        answers = payload.get("answers")
        if not isinstance(answers, list):
            raise ValueError("missing answers list")
        by_name = {a.get("name"): a for a in answers if isinstance(a, dict)}
        if len(item.questions) == 1 and item.questions[0].name not in by_name and len(answers) == 1:
            by_name = {item.questions[0].name: answers[0]}
        out = {}
        for q in item.questions:
            a = by_name.get(q.name)
            if not isinstance(a, dict):
                out[q.name] = Answer(error="missing answer")
                continue
            if a.get("type") == "refusal":
                out[q.name] = Answer(error="refusal", refusal=True)
                continue
            choice = a.get("choice")
            out[q.name] = Answer(
                choice=choice if isinstance(choice, str) and choice else None,
                probs_raw=_list(a.get("probabilities")),
                api_confidence=_num(a.get("confidence")),
                error="" if isinstance(choice, str) and choice else "missing choice",
            )
        return out


MODELS = {"jev": JevPlugin, "luna": LunaPlugin}
MODEL_ORDER = ("jev", "luna", "laya")  # laya runs locally via python -m eval.laya

