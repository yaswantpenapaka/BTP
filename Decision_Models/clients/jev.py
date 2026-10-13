"""TypeSafe Jev client. Request words come from tasks.py."""

from __future__ import annotations

import time
from pathlib import Path

from clients.common import (
    Parsed,
    expect_confidence,
    expect_prob_map,
    fail_schema,
    parsed_from_http,
    read_input_tokens,
    warn_if_model_differs,
)
from clients.transport import post_json
from tasks import JEV_MODEL, Task

JEV_URL = "https://api.typesafe.ai/v1/systemone"


def build_body(task: Task, text: str) -> dict:
    return {
        "model": JEV_MODEL,
        "state": task.state(text),
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": task.instruction,
                "criteria": task.criteria(),
            }
        },
    }


def parse_response(payload: object, logs_dir: Path) -> Parsed:
    """Read answers.answer.choice, the probability map, and confidence.

    A missing choice on an otherwise readable choice object is a row error.
    A missing answers object, probability map, or confidence is a schema error:
    the raw body is saved and the run stops so the parser can be fixed.
    """
    tokens = read_input_tokens(payload) if isinstance(payload, dict) else None

    def fail(message: str):
        fail_schema(logs_dir, "jev", payload, message, tokens)

    if not isinstance(payload, dict):
        fail("Jev response is not a JSON object.")
    answers = payload.get("answers")
    if not isinstance(answers, dict) or "answer" not in answers:
        fail("Jev response is missing answers.answer.")
    answer = answers.get("answer")
    if not isinstance(answer, dict):
        fail("Jev answers.answer is not an object.")

    choice = answer.get("choice") if "choice" in answer else None
    if choice is None or choice == "":
        return Parsed(
            pred="",
            probabilities=_optional_map(answer, fail),
            confidence=_optional_confidence(answer, fail),
            input_tokens=tokens,
            error="missing choice",
        )
    if not isinstance(choice, str):
        fail("Jev answers.answer.choice is not a string.")
    if "probabilities" not in answer:
        fail("Jev response is missing answers.answer.probabilities.")
    try:
        probabilities = expect_prob_map(answer.get("probabilities"), "answers.answer.probabilities")
    except ValueError as exc:
        fail(str(exc))
    if "confidence" not in answer:
        fail("Jev response is missing answers.answer.confidence.")
    try:
        confidence = expect_confidence(answer.get("confidence"), "answers.answer.confidence")
    except ValueError as exc:
        fail(str(exc))
    return Parsed(
        pred=choice,
        probabilities=probabilities,
        confidence=confidence,
        input_tokens=tokens,
    )


def _optional_map(answer: dict, fail) -> dict[str, float] | None:
    if "probabilities" not in answer or answer.get("probabilities") is None:
        return None
    try:
        return expect_prob_map(answer.get("probabilities"), "answers.answer.probabilities")
    except ValueError as exc:
        fail(str(exc))


def _optional_confidence(answer: dict, fail) -> float | None:
    if "confidence" not in answer or answer.get("confidence") is None:
        return None
    try:
        return expect_confidence(answer.get("confidence"), "answers.answer.confidence")
    except ValueError as exc:
        fail(str(exc))


class JevClient:
    model = JEV_MODEL
    api_name = "jev"

    def __init__(self, api_key: str, logs_dir: Path, log=None, warn=None, session=None, sleep=None):
        if not str(api_key).strip():
            raise ValueError("empty API key")
        self._api_key = api_key
        self.logs_dir = logs_dir
        self._log = log or (lambda _message: None)
        self._warn = warn or self._log
        self._session = session
        self._sleep = sleep or time.sleep

    def __repr__(self) -> str:
        return "JevClient(model=jev-1.13.0)"

    def decide(self, task: Task, text: str) -> Parsed:
        outcome = post_json(
            JEV_URL,
            self._api_key,
            build_body(task, text),
            api_name=self.api_name,
            logs_dir=self.logs_dir,
            log=self._log,
            session=self._session,
            sleep=self._sleep,
        )
        warn_if_model_differs(outcome, self.model, self.api_name, self._warn)
        try:
            parsed = parsed_from_http(outcome, self.model, self.logs_dir, parse_response)
        except Exception as exc:
            if outcome.latency_ms is not None and hasattr(exc, "latency_ms"):
                exc.latency_ms = outcome.latency_ms
            raise
        if parsed.error == "" and parsed.input_tokens is None:
            self._log(f"{self.api_name} response had no usage.input_tokens. cost_usd left blank.")
        return parsed
