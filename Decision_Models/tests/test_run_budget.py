"""Scheduler tests use a fake decision function. They do not call either API."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clients.common import Parsed, ResponseSchemaError, StopRun
from run import Job, execute_jobs, main, pending_jobs
from safety import Budget
from tasks import JEV_MODEL, OPENAI_MODEL, PRICE_PER_TOKEN


def _ok(cost: float, charge: str = "actual") -> Parsed:
    return Parsed(
        pred="negative",
        probabilities={"negative": 0.8, "positive": 0.2},
        confidence=0.7,
        input_tokens=10,
        cost_usd=cost,
        latency_ms=5,
        charge=charge,
    )


class BudgetRunTests(unittest.TestCase):
    def test_cap_stops_before_the_next_reservation(self):
        jobs = [Job(str(i), JEV_MODEL, "text", "negative") for i in range(3)]
        budget = Budget(1.0, {JEV_MODEL: 0.4})
        calls = []

        def call_fn(job: Job) -> Parsed:
            calls.append(job.item_id)
            return _ok(0.4)

        written = []
        result = execute_jobs(jobs, budget, 1, call_fn, lambda job, _parsed: written.append(job.item_id))
        self.assertEqual(calls, ["0", "1"])
        self.assertEqual(written, ["0", "1"])
        self.assertEqual(result.unsent, 1)
        self.assertIsNone(result.stop_reason)

    def test_smaller_actual_cost_unparks_later_jobs(self):
        jobs = [Job(str(i), JEV_MODEL, "text", "negative") for i in range(3)]
        budget = Budget(1.0, {JEV_MODEL: 0.6})
        calls = []

        def call_fn(job: Job) -> Parsed:
            calls.append(job.item_id)
            return _ok(0.1)

        result = execute_jobs(jobs, budget, 2, call_fn, lambda _job, _parsed: None)
        self.assertEqual(calls, ["0", "1", "2"])
        self.assertEqual(result.unsent, 0)

    def test_released_reservation_does_not_block_the_next_call(self):
        jobs = [Job(str(i), JEV_MODEL, "text", "negative") for i in range(2)]
        budget = Budget(0.5, {JEV_MODEL: 0.4})
        calls = []

        def call_fn(job: Job) -> Parsed:
            calls.append(job.item_id)
            if job.item_id == "0":
                return Parsed(error="connection error after 6 attempts", charge="release")
            return _ok(0.4)

        written = []
        result = execute_jobs(jobs, budget, 1, call_fn, lambda job, _parsed: written.append(job.item_id))
        self.assertEqual(calls, ["0", "1"])
        self.assertEqual(written, ["1"])
        self.assertEqual(result.not_written, 1)

    def test_rejected_request_does_not_mark_the_item_done(self):
        jobs = [Job(str(i), JEV_MODEL, "text", "negative") for i in range(3)]
        budget = Budget(5.0, {JEV_MODEL: 0.1})
        calls = []

        def call_fn(job: Job) -> Parsed:
            calls.append(job.item_id)
            if job.item_id == "1":
                raise StopRun("jev rejected the request (HTTP 400).", charge="release")
            return _ok(0.1)

        written = []
        result = execute_jobs(jobs, budget, 1, call_fn, lambda job, _parsed: written.append(job.item_id))
        self.assertEqual(calls, ["0", "1"])
        self.assertEqual(written, ["0"])
        self.assertEqual(result.unsent, 1)
        self.assertIn("HTTP 400", result.stop_reason or "")

    def test_schema_error_is_written_once_and_stops_the_queue(self):
        jobs = [Job(str(i), JEV_MODEL, "text", "negative") for i in range(2)]
        budget = Budget(5.0, {JEV_MODEL: 0.1})

        def call_fn(job: Job) -> Parsed:
            if job.item_id == "0":
                raise ResponseSchemaError("missing answers.answer.", Path("logs/schema_jev_test.json"), input_tokens=12)
            return _ok(0.1)

        written = []
        result = execute_jobs(jobs, budget, 1, call_fn, lambda job, parsed: written.append(parsed))
        self.assertEqual(len(written), 1)
        self.assertIn("schema mismatch", written[0].error)
        self.assertAlmostEqual(written[0].cost_usd, 12 * PRICE_PER_TOKEN[JEV_MODEL])
        self.assertEqual(result.unsent, 1)

    def test_one_model_hitting_the_cap_does_not_block_the_other(self):
        jobs = [
            Job("0", JEV_MODEL, "text", "negative"),
            Job("0", OPENAI_MODEL, "text", "negative"),
        ]
        budget = Budget(1.0, {JEV_MODEL: 2.0, OPENAI_MODEL: 0.1})
        calls = []

        def call_fn(job: Job) -> Parsed:
            calls.append(job.model)
            return _ok(0.1)

        result = execute_jobs(jobs, budget, 2, call_fn, lambda _job, _parsed: None)
        self.assertEqual(calls, [OPENAI_MODEL])
        self.assertEqual(result.unsent, 1)

    def test_pending_jobs_skip_pairs_already_in_the_csv(self):
        items = [("0", "a", "negative"), ("1", "b", "positive")]
        done = {("0", JEV_MODEL)}
        jobs = pending_jobs(items, [JEV_MODEL, OPENAI_MODEL], done)
        self.assertEqual(
            [(job.item_id, job.model) for job in jobs],
            [("0", OPENAI_MODEL), ("1", JEV_MODEL), ("1", OPENAI_MODEL)],
        )

    def test_smoke_and_dataset_together_send_nothing(self):
        with self.assertRaises(SystemExit) as caught:
            main(["--smoke", "--dataset", "sst2"])
        self.assertIn("No requests were sent", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
