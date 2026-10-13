"""HTTP retry tests use a fake session. They do not call either API."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from clients.transport import post_json

SECRET = "unit-test-key-do-not-leak"


class FakeResponse:
    def __init__(self, status, payload=None, headers=None, text=None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload
        self.text = text if text is not None else (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(headers)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        raise AssertionError("an injected session must not be closed")


class TransportTests(unittest.TestCase):
    def test_retries_503_then_returns_body(self):
        payload = {"answers": {"answer": {"choice": "negative"}}}
        session = FakeSession(
            [
                FakeResponse(503, text=f"nope Bearer {SECRET}"),
                FakeResponse(200, payload=payload),
            ]
        )
        logs = []
        with tempfile.TemporaryDirectory() as directory:
            outcome = post_json(
                "https://api.typesafe.ai/v1/systemone",
                SECRET,
                {"model": "jev-1.13.0"},
                api_name="jev",
                logs_dir=Path(directory),
                log=logs.append,
                session=session,
                sleep=lambda _seconds: None,
            )
        self.assertEqual(outcome.kind, "ok")
        self.assertEqual(outcome.payload, payload)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[0]["Authorization"], f"Bearer {SECRET}")
        self.assertNotIn(SECRET, outcome.message)
        self.assertNotIn(SECRET, "\n".join(logs))

    def test_exhausted_500_keeps_the_charge_and_stops_retrying(self):
        session = FakeSession([FakeResponse(500, text="down") for _ in range(6)])
        sleeps = []
        with tempfile.TemporaryDirectory() as directory:
            outcome = post_json(
                "https://api.openai.com/v1/decisions",
                SECRET,
                {"model": "gpt-6-luna"},
                api_name="openai",
                logs_dir=Path(directory),
                log=lambda _message: None,
                session=session,
                sleep=sleeps.append,
            )
        self.assertEqual(outcome.kind, "error")
        self.assertEqual(outcome.charge, "actual")
        self.assertEqual(outcome.message, "HTTP 500 after 6 attempts")
        self.assertEqual(len(session.calls), 6)
        self.assertEqual(len(sleeps), 5)
        self.assertNotIn(SECRET, outcome.message)

    def test_401_stops_without_logging_the_key(self):
        session = FakeSession([FakeResponse(401, text=f"bad Bearer {SECRET}")])
        with tempfile.TemporaryDirectory() as directory:
            outcome = post_json(
                "https://api.openai.com/v1/decisions",
                SECRET,
                {},
                api_name="openai",
                logs_dir=Path(directory),
                log=lambda _message: None,
                session=session,
                sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError("401 must not sleep")),
            )
        self.assertEqual(outcome.kind, "stop")
        self.assertEqual(outcome.charge, "release")
        self.assertIn("HTTP 401", outcome.message)
        self.assertNotIn(SECRET, outcome.message)
        self.assertEqual(len(session.calls), 1)

    def test_400_saves_a_redacted_body_and_does_not_retry(self):
        session = FakeSession([FakeResponse(400, text=f"bad request Bearer {SECRET}")])
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory)
            outcome = post_json(
                "https://api.typesafe.ai/v1/systemone",
                SECRET,
                {},
                api_name="jev",
                logs_dir=logs,
                log=lambda _message: None,
                session=session,
                sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError("400 must not sleep")),
            )
            saved = list(logs.glob("schema_jev_*.json"))
            self.assertEqual(len(saved), 1)
            body = saved[0].read_text(encoding="utf-8")
        self.assertEqual(outcome.kind, "stop")
        self.assertNotIn(SECRET, body)
        self.assertNotIn(SECRET, outcome.message)
        self.assertIn("Bearer [redacted]", body)
        self.assertEqual(len(session.calls), 1)

    def test_read_timeout_is_not_retried(self):
        session = FakeSession([requests.ReadTimeout("timed out")])
        with tempfile.TemporaryDirectory() as directory:
            outcome = post_json(
                "https://api.openai.com/v1/decisions",
                SECRET,
                {},
                api_name="openai",
                logs_dir=Path(directory),
                log=lambda _message: None,
                session=session,
                sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError("timeout must not sleep")),
            )
        self.assertEqual(outcome.kind, "error")
        self.assertEqual(outcome.charge, "actual")
        self.assertEqual(outcome.message, "read timeout; not retried")
        self.assertEqual(len(session.calls), 1)
        self.assertNotIn(SECRET, outcome.message)


if __name__ == "__main__":
    unittest.main()
