"""Budget guard, retry delay, and log redaction. No network calls in this module."""

from __future__ import annotations

import logging
import re
import threading
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone


_BEARER = re.compile(r"Bearer\s+[^\s\"']+", re.IGNORECASE)


def redact(text: str) -> str:
    return _BEARER.sub("Bearer [redacted]", text)


class RedactFilter(logging.Filter):
    """Strips bearer tokens from log lines before they are written."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact(item) if isinstance(item, str) else item for item in record.args)
        return True


class Budget:
    """Stops a model once this run's reserved-or-actual spend reaches the cap.

    Each call reserves `reserve_usd` before it is sent. A success replaces
    that reservation with the real token cost. A transport failure releases
    the reservation. A success with no token count keeps the reservation, so
    a missing usage field cannot hide spend.
    """

    def __init__(self, max_usd: float, reserve_usd: dict[str, float]):
        if max_usd <= 0:
            raise ValueError("max_usd must be positive")
        self.max_usd = float(max_usd)
        self.reserve_usd = dict(reserve_usd)
        self.spent = {model: 0.0 for model in reserve_usd}
        self.stopped: set[str] = set()
        self._lock = threading.Lock()

    def try_reserve(self, model: str) -> bool:
        with self._lock:
            reserve = self.reserve_usd[model]
            if self.spent[model] + reserve > self.max_usd + 1e-12:
                self.stopped.add(model)
                return False
            self.spent[model] += reserve
            self._sync_stopped(model)
            return True

    def reconcile(self, model: str, actual_usd: float | None) -> None:
        """Replace this call's reservation with the real cost.

        actual_usd None keeps the reservation. That is the safe choice when
        the server answered but did not report a token count.
        """
        with self._lock:
            if actual_usd is not None:
                reserve = self.reserve_usd[model]
                self.spent[model] += float(actual_usd) - reserve
                if self.spent[model] < 0:
                    self.spent[model] = 0.0
            self._sync_stopped(model)

    def release(self, model: str) -> None:
        """Drop this call's reservation. Use only when the request was not accepted."""
        with self._lock:
            self.spent[model] -= self.reserve_usd[model]
            if self.spent[model] < 0:
                self.spent[model] = 0.0
            self._sync_stopped(model)

    def _sync_stopped(self, model: str) -> None:
        # Cleared when an in-flight call reconciles below the cap, so the
        # remaining rows can still run inside the same limit.
        if self.spent[model] >= self.max_usd - 1e-12:
            self.stopped.add(model)
        else:
            self.stopped.discard(model)


def retry_delay_seconds(headers: dict | None, attempt: int) -> float:
    """attempt is 1-based retry count. Uses Retry-After, else 1, 2, 4, 8, 16 capped at 60."""
    delay = min(60.0, float(2 ** (attempt - 1)))
    if not headers:
        return delay
    raw = None
    for key, value in headers.items():
        if key.lower() == "retry-after":
            raw = value
            break
    if raw is None:
        return delay
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    try:
        when = parsedate_to_datetime(str(raw))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return delay
