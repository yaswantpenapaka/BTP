"""HTTP POST with retries. The API key is placed on the request and nowhere else."""

from __future__ import annotations

import time
from pathlib import Path

import requests

from clients.common import HttpOutcome, save_raw
from safety import redact, retry_delay_seconds

# One initial attempt plus five retries.
MAX_RETRIES = 5
CONNECT_TIMEOUT_S = 10
READ_TIMEOUT_S = 60


def post_json(
    url: str,
    api_key: str,
    body: dict,
    *,
    api_name: str,
    logs_dir: Path,
    log,
    session=None,
    sleep=time.sleep,
) -> HttpOutcome:
    owns_session = session is None
    session = session or requests.Session()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = (CONNECT_TIMEOUT_S, READ_TIMEOUT_S)
    try:
        for attempt in range(MAX_RETRIES + 1):
            started = time.perf_counter()
            try:
                response = session.post(url, headers=headers, json=body, timeout=timeout)
            except requests.ConnectionError:
                latency_ms = (time.perf_counter() - started) * 1000
                if attempt < MAX_RETRIES:
                    delay = retry_delay_seconds(None, attempt + 1)
                    log(f"{api_name} connection error, retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s")
                    sleep(delay)
                    continue
                return HttpOutcome(
                    kind="error",
                    message=f"connection error after {MAX_RETRIES + 1} attempts",
                    latency_ms=latency_ms,
                    charge="release",
                )
            except requests.Timeout:
                # A read timeout may already have been billed. Do not send it again.
                latency_ms = (time.perf_counter() - started) * 1000
                log(f"{api_name} read timeout after {latency_ms:.0f}ms. Not retried.")
                return HttpOutcome(
                    kind="error",
                    message="read timeout; not retried",
                    latency_ms=latency_ms,
                    charge="actual",
                )
            except requests.RequestException as exc:
                latency_ms = (time.perf_counter() - started) * 1000
                log(f"{api_name} {type(exc).__name__}. Not retried.")
                return HttpOutcome(
                    kind="error",
                    message=f"{type(exc).__name__}; not retried",
                    latency_ms=latency_ms,
                    charge="actual",
                )

            latency_ms = (time.perf_counter() - started) * 1000
            status = response.status_code
            if status == 200:
                return _ok_body(response, api_name, logs_dir, latency_ms)
            if status in (401, 403):
                return HttpOutcome(
                    kind="stop",
                    message=(
                        f"{api_name} rejected the key (HTTP {status}). "
                        "The key was not logged. Check the environment variable for this API."
                    ),
                    latency_ms=latency_ms,
                    charge="release",
                )
            if status == 429 or status >= 500:
                if attempt < MAX_RETRIES:
                    delay = retry_delay_seconds(response.headers, attempt + 1)
                    log(f"{api_name} HTTP {status}, retry {attempt + 1}/{MAX_RETRIES} in {delay:.1f}s")
                    sleep(delay)
                    continue
                log(f"{api_name} HTTP {status} after {MAX_RETRIES + 1} attempts. {_snippet(response)}")
                return HttpOutcome(
                    kind="error",
                    message=f"HTTP {status} after {MAX_RETRIES + 1} attempts",
                    latency_ms=latency_ms,
                    charge="actual",
                )
            saved = save_raw(logs_dir, api_name, _body_text(response))
            return HttpOutcome(
                kind="stop",
                message=(
                    f"{api_name} rejected the request (HTTP {status}). "
                    f"No row was written for this item. Body saved to {saved}"
                ),
                latency_ms=latency_ms,
                charge="release",
                saved_path=saved,
            )
        return HttpOutcome(kind="error", message="request failed", charge="actual")
    finally:
        if owns_session:
            session.close()


def _ok_body(response, api_name: str, logs_dir: Path, latency_ms: float) -> HttpOutcome:
    try:
        payload = response.json()
    except ValueError:
        saved = save_raw(logs_dir, api_name, _body_text(response))
        return HttpOutcome(
            kind="schema",
            message=f"{api_name} returned HTTP 200 with a body that is not JSON.",
            latency_ms=latency_ms,
            charge="actual",
            saved_path=saved,
        )
    if not isinstance(payload, dict):
        saved = save_raw(logs_dir, api_name, payload)
        return HttpOutcome(
            kind="schema",
            message=f"{api_name} returned JSON that is not an object.",
            latency_ms=latency_ms,
            charge="actual",
            saved_path=saved,
        )
    return HttpOutcome(kind="ok", payload=payload, latency_ms=latency_ms, charge="actual")


def _body_text(response) -> str:
    try:
        return redact(response.text[:8000])
    except Exception:
        return ""


def _snippet(response) -> str:
    return _body_text(response)[:300]
