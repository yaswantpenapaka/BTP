"""OpenAI Decisions client. Request words come from the same task as Jev."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from clients.common import (
    Parsed,
    expect_confidence,
    fail_schema,
    parsed_from_http,
    read_input_tokens,
    warn_if_model_differs,
)
from clients.transport import post_json
from tasks import OPENAI_MODEL, Task

OPENAI_URL = "https://api.openai.com/v1/decisions"


def build_body(task: Task, text: str) -> dict:
    state = task.state(text)
    return {
        "model": OPENAI_MODEL,
        "input": json.dumps(state, ensure_ascii=False),
        "questions": [
            {
                "type": "choice",
                "name": "answer",
                "instructions": task.instruction,
                "choices": [
                    {"value": label, "description": task.descriptions[label]}
                    for label in task.prompt_labels
                ],
            }
        ],
    }


def parse_response(payload: object, logs_dir: Path) -> Parsed:
    """Read answers[name=answer].choice, the probability list, and confidence.

    type "refusal", or a choice object with an empty choice, is a row error.
    A missing answers list, an unexpected answer type, or a choice that has
    no probability list is a schema error: the raw body is saved and the run stops.
    """
    tokens = read_input_tokens(payload) if isinstance(payload, dict) else None

    def fail(message: str):
        fail_schema(logs_dir, "openai", payload, message, tokens)

    if not isinstance(payload, dict):
        fail("OpenAI Decisions response is not a JSON object.")
    answers = payload.get("answers")
    if not isinstance(answers, list) or not answers:
        fail("OpenAI Decisions response is missing a non-empty answers list.")
    answer = _select_answer(answers)
    if not isinstance(answer, dict):
        fail("OpenAI Decisions response has no answer named 'answer'.")

    kind = answer.get("type")
    if kind == "refusal":
        return Parsed(pred="", input_tokens=tokens, error="refusal")
    if kind not in (None, "choice"):
        fail(f"OpenAI Decisions answer type {kind!r} is not choice or refusal.")

    choice = answer.get("choice") if "choice" in answer else None
    if choice is None or choice == "":
        return Parsed(
            pred="",
            probabilities=_optional_list(answer, fail),
            confidence=_optional_confidence(answer, fail),
            input_tokens=tokens,
            error="missing choice",
        )
    if not isinstance(choice, str):
        fail("OpenAI Decisions answers choice is not a string.")
    if "probabilities" not in answer:
        fail("OpenAI Decisions response is missing answers[].probabilities.")
    try:
        probabilities = expect_prob_list(answer.get("probabilities"))
    except ValueError as exc:
        fail(str(exc))
    if "confidence" not in answer:
        fail("OpenAI Decisions response is missing answers[].confidence.")
    try:
        confidence = expect_confidence(answer.get("confidence"), "answers[].confidence")
    except ValueError as exc:
        fail(str(exc))
    return Parsed(
        pred=choice,
        probabilities=probabilities,
        confidence=confidence,
        input_tokens=tokens,
    )


def _select_answer(answers: list):
    named = [item for item in answers if isinstance(item, dict) and item.get("name") == "answer"]
    if named:
        return named[0]
    if len(answers) == 1 and isinstance(answers[0], dict) and "name" not in answers[0]:
        return answers[0]
    return None


def expect_prob_list(raw: object) -> dict[str, float]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("answers[].probabilities must be a non-empty list of {value, probability}")
    out: dict[str, float] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("answers[].probabilities entries must be objects")
        value = item.get("value")
        probability = item.get("probability")
        if not isinstance(value, str) or isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(
                "answers[].probabilities entries need a string value and a numeric probability"
            )
        number = float(probability)
        if not math.isfinite(number):
            raise ValueError("answers[].probabilities entries must be finite numbers")
        out[value] = number
    return out


def _optional_list(answer: dict, fail) -> dict[str, float] | None:
    if "probabilities" not in answer or answer.get("probabilities") is None:
        return None
    try:
        return expect_prob_list(answer.get("probabilities"))
    except ValueError as exc:
        fail(str(exc))


def _optional_confidence(answer: dict, fail) -> float | None:
    if "confidence" not in answer or answer.get("confidence") is None:
        return None
    try:
        return expect_confidence(answer.get("confidence"), "answers[].confidence")
    except ValueError as exc:
        fail(str(exc))


class OpenAIDecisionsClient:
    model = OPENAI_MODEL
    api_name = "openai"

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
        return "OpenAIDecisionsClient(model=gpt-6-luna)"

    def decide(self, task: Task, text: str) -> Parsed:
        outcome = post_json(
            OPENAI_URL,
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
