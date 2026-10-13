"""Budget, retry delay, and log redaction. No network calls."""

from __future__ import annotations

import logging
import sys
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety import Budget, RedactFilter, redact, retry_delay_seconds


class SafetyTests(unittest.TestCase):
    def test_redact_bearer(self):
        text = "Authorization: Bearer sk-test-SHOULD-NOT-LEAK"
        cleaned = redact(text)
        self.assertIn("Bearer [redacted]", cleaned)
        self.assertNotIn("SHOULD-NOT-LEAK", cleaned)

    def test_log_filter_redacts_message_and_args(self):
        record = logging.LogRecord(
            name="jev_runs",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="header %s",
            args=("Bearer sk-test-SHOULD-NOT-LEAK",),
            exc_info=None,
        )
        self.assertTrue(RedactFilter().filter(record))
        rendered = record.msg % record.args
        self.assertNotIn("SHOULD-NOT-LEAK", rendered)

    def test_retry_delay_exponential_and_retry_after(self):
        self.assertEqual(retry_delay_seconds(None, 1), 1.0)
        self.assertEqual(retry_delay_seconds({}, 2), 2.0)
        self.assertEqual(retry_delay_seconds(None, 7), 60.0)
        self.assertEqual(retry_delay_seconds({"Retry-After": "3"}, 1), 3.0)
        when = datetime.now(timezone.utc) + timedelta(seconds=5)
        delay = retry_delay_seconds({"retry-after": format_datetime(when)}, 1)
        self.assertGreater(delay, 0)
        self.assertLess(delay, 10)

    def test_budget_frees_room_after_reconcile(self):
        budget = Budget(1.0, {"jev-1.13.0": 0.6})
        self.assertTrue(budget.try_reserve("jev-1.13.0"))
        self.assertFalse(budget.try_reserve("jev-1.13.0"))
        self.assertIn("jev-1.13.0", budget.stopped)
        budget.reconcile("jev-1.13.0", 0.1)
        self.assertNotIn("jev-1.13.0", budget.stopped)
        self.assertTrue(budget.try_reserve("jev-1.13.0"))

    def test_release_allows_another_reservation(self):
        budget = Budget(0.5, {"jev-1.13.0": 0.4})
        self.assertTrue(budget.try_reserve("jev-1.13.0"))
        budget.release("jev-1.13.0")
        self.assertTrue(budget.try_reserve("jev-1.13.0"))

    def test_missing_token_count_keeps_the_reservation(self):
        budget = Budget(1.0, {"jev-1.13.0": 0.4})
        budget.try_reserve("jev-1.13.0")
        budget.reconcile("jev-1.13.0", None)
        self.assertAlmostEqual(budget.spent["jev-1.13.0"], 0.4)


if __name__ == "__main__":
    unittest.main()
