"""Shared decision result and the loud schema-mismatch path."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NoReturn

from safety import redact
from tasks import PRICE_PER_TOKEN


@dataclass
class Parsed:
    pred: str = ""
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    input_tokens: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None
    error: str = ""
    # "actual": the server accepted the request. cost_usd None keeps the reservation.
    # "release": the request was not accepted. Drop the reservation.
    charge: str = "actual"


class ResponseSchemaError(RuntimeError):
    """The body is not the documented decision object. The raw JSON is on disk."""

    def __init__(self, message: str, saved_path: Path, input_tokens: int | None = None):
        super().__init__(f"{message} Raw response saved to {saved_path}")
        self.saved_path = saved_path
        self.input_tokens = input_tokens
        self.latency_ms: float | None = None


class StopRun(RuntimeError):
    """The API rejected the request. The item is left unwritten so a rerun can retry it."""

    def __init__(self, message: str, charge: str = "release"):
        super().__init__(message)
        self.charge = charge


@dataclass
class HttpOutcome:
    kind: str  # ok, error, stop, schema
    payload: dict | None = None
    message: str = ""
    latency_ms: float | None = None
    charge: str = "release"
    saved_path: Path | None = None


def save_raw(logs_dir: Path, api: str, payload: object) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = logs_dir / f"schema_{api}_{stamp}.json"
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(redact(text), encoding="utf-8")
    return path


def fail_schema(logs_dir: Path, api: str, payload: object, message: str, input_tokens: int | None) -> NoReturn:
    path = save_raw(logs_dir, api, payload)
    raise ResponseSchemaError(message, path, input_tokens=input_tokens)


def read_input_tokens(payload: dict) -> int | None:
    """usage.input_tokens when that field is present. Any other shape stays blank."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    raw = usage.get("input_tokens")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return int(raw)


def expect_prob_map(raw: object, path_label: str) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path_label} must be a non-empty map of option to probability")
    out: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path_label} has a non-numeric entry for {key!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path_label} has a non-finite probability for {key!r}")
        out[key] = number
    return out


def expect_confidence(raw: object, path_label: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{path_label} must be a number")
    number = float(raw)
    if not math.isfinite(number):
        raise ValueError(f"{path_label} must be a finite number")
    return number


def warn_if_model_differs(outcome: HttpOutcome, expected: str, api_name: str, warn) -> None:
    payload = outcome.payload
    if not isinstance(payload, dict):
        return
    reported = payload.get("model")
    if isinstance(reported, str) and reported != expected:
        warn(f"{api_name} responded as model {reported}; this run records {expected}")


def parsed_from_http(outcome: HttpOutcome, model: str, logs_dir: Path, parse_response) -> Parsed:
    if outcome.kind == "schema":
        raise ResponseSchemaError(
            outcome.message,
            outcome.saved_path or logs_dir,
            input_tokens=None,
        )
    if outcome.kind == "stop":
        raise StopRun(outcome.message, charge=outcome.charge)
    if outcome.kind == "error":
        return Parsed(
            error=outcome.message,
            latency_ms=outcome.latency_ms,
            charge="release" if outcome.charge == "release" else "actual",
        )
    parsed = parse_response(outcome.payload, logs_dir)
    parsed.latency_ms = outcome.latency_ms
    if parsed.input_tokens is not None:
        parsed.cost_usd = parsed.input_tokens * PRICE_PER_TOKEN[model]
    return parsed
