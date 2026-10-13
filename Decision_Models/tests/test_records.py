"""CSV resume and score-block tests. No network calls."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clients.common import Parsed
from records import (
    append_item_row,
    call_with_lock_retry,
    decision_row,
    done_keys,
    gold_from_index,
    read_item_rows,
    rescore_rows,
    score_lines,
    validate_split,
    write_table,
)
from tasks import JEV_MODEL, OPENAI_MODEL, get_task


class RecordTests(unittest.TestCase):
    def test_rotten_tomatoes_index_map_stops_on_a_different_name_list(self):
        task = get_task("rotten_tomatoes")
        validate_split(task, ("neg", "pos"), 1066)
        self.assertEqual(gold_from_index(task, 0), "negative")
        self.assertEqual(gold_from_index(task, 1), "positive")
        with self.assertRaises(SystemExit):
            validate_split(task, ("negative", "positive"), 1066)
        with self.assertRaises(SystemExit):
            validate_split(task, ("neg", "pos"), 1067)
        with self.assertRaises(SystemExit):
            gold_from_index(task, True)

    def test_expected_counts_are_checked(self):
        validate_split(get_task("sst2"), ("negative", "positive"), 872)
        validate_split(get_task("emotion"), ("sadness", "joy", "love", "anger", "fear", "surprise"), 2000)
        validate_split(get_task("financial_phrasebank"), ("negative", "neutral", "positive"), 970)
        with self.assertRaises(SystemExit):
            validate_split(get_task("sst2"), ("negative", "positive"), 871)

    def test_score_rows_are_not_treated_as_finished_items(self):
        task = get_task("sst2")
        row = decision_row(
            "0",
            JEV_MODEL,
            "negative",
            Parsed(pred="negative", probabilities={"negative": 0.9, "positive": 0.1}, confidence=0.8, input_tokens=20, cost_usd=0.000001),
            task.prompt_labels,
        )
        self.assertEqual(row["correct"], "1")
        self.assertEqual(row["prob_pred"], "0.9")
        self.assertEqual(row["prob_gold"], "0.9")
        refusal = decision_row(
            "1",
            OPENAI_MODEL,
            "negative",
            Parsed(pred="", error="refusal", input_tokens=5, cost_usd=0.0000005),
            task.prompt_labels,
        )
        self.assertEqual(refusal["pred"], "")
        self.assertEqual(refusal["correct"], "0")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sst2.csv"
            write_table(
                path,
                [row, refusal],
                [
                    ["metric", JEV_MODEL, OPENAI_MODEL],
                    ["n", "1", "1"],
                    ["exact_agreement", "0.5", ""],
                ],
            )
            loaded = read_item_rows(path)
            self.assertEqual(done_keys(loaded), {("0", JEV_MODEL), ("1", OPENAI_MODEL)})
            self.assertNotIn(("n", JEV_MODEL), done_keys(loaded))
            append_item_row(
                path,
                decision_row(
                    "2",
                    JEV_MODEL,
                    "positive",
                    Parsed(pred="positive", probabilities={"negative": 0.2, "positive": 0.8}, confidence=0.7),
                    task.prompt_labels,
                ),
            )
            self.assertIn(("2", JEV_MODEL), done_keys(read_item_rows(path)))

    def test_lock_retry_waits_and_then_writes(self):
        attempts = {"n": 0}
        sleeps = []

        def action():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise PermissionError("locked")
            return "ok"

        messages = []
        result = call_with_lock_retry(
            action,
            Path("outputs/sst2.csv"),
            attempts=3,
            delay_s=2,
            sleep=sleeps.append,
            echo=messages.append,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [2, 2])
        self.assertEqual(len(messages), 2)
        self.assertIn("Excel", messages[0])

    def test_rescore_does_not_need_an_api(self):
        task = get_task("sst2")
        rows = [
            decision_row(
                "0",
                JEV_MODEL,
                "positive",
                Parsed(
                    pred="positive",
                    probabilities={"negative": 0.2, "positive": 0.8},
                    confidence=0.7,
                    input_tokens=10,
                    cost_usd=0.001,
                    latency_ms=10,
                ),
                task.prompt_labels,
            )
        ]
        probs = {("0", JEV_MODEL): {"negative": 0.2, "positive": 0.8}}
        scores = rescore_rows(rows, probs, list(task.prompt_labels))
        lines = score_lines(scores)
        self.assertEqual(lines[1][0], "n")
        self.assertEqual(lines[1][1], "1")
        self.assertEqual(lines[-1][0], "exact_agreement")


if __name__ == "__main__":
    unittest.main()
