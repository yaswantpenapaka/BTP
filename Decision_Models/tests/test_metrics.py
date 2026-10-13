"""Hand-checked accuracy, macro-F1, ECE, and Brier. No network calls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import accuracy, brier_score, expected_calibration_error, macro_f1, score_models
from records import fmt_metric, score_lines
from tasks import JEV_MODEL, MODEL_ORDER, OPENAI_MODEL

LABELS = ["negative", "positive"]


class MetricTests(unittest.TestCase):
    def test_hand_checked_example(self):
        # Item 0: gold positive, pred positive, P(pred)=0.8. Squared error 0.08.
        # Item 1: gold positive, pred negative, P(pred)=0.6. Squared error 0.72.
        # Item 2: refusal, no distribution.
        # Accuracy = 1/3.
        # macro-F1: positive F1 = 2/3, negative F1 = 0, macro = 1/3.
        # ECE bins [0.6, 0.7) and [0.8, 0.9): 0.5*|0-0.6| + 0.5*|1-0.8| = 0.4.
        # Brier over the two items with a distribution: (0.08 + 0.72) / 2 = 0.4.
        rows = [
            _row("0", JEV_MODEL, "positive", "positive", 0.8, 10, 100, 0.001),
            _row("1", JEV_MODEL, "positive", "negative", 0.6, 30, 100, 0.001),
            _row("2", JEV_MODEL, "negative", "", None, 20, None, None, error="refusal"),
            _row("0", OPENAI_MODEL, "positive", "positive", 0.7, 12, 50, 0.002),
            _row("1", OPENAI_MODEL, "positive", "positive", 0.55, 14, 50, 0.002),
            _row("2", OPENAI_MODEL, "negative", "negative", 0.9, 16, 50, 0.002),
        ]
        probs = {
            ("0", JEV_MODEL): {"negative": 0.2, "positive": 0.8},
            ("1", JEV_MODEL): {"negative": 0.6, "positive": 0.4},
            ("0", OPENAI_MODEL): {"negative": 0.3, "positive": 0.7},
            ("1", OPENAI_MODEL): {"negative": 0.45, "positive": 0.55},
            ("2", OPENAI_MODEL): {"negative": 0.9, "positive": 0.1},
        }
        scores = score_models(rows, probs, LABELS, MODEL_ORDER)
        jev = scores[JEV_MODEL]
        self.assertEqual(jev["n"], 3)
        self.assertAlmostEqual(jev["accuracy"], 1 / 3)
        self.assertAlmostEqual(jev["macro_f1"], 1 / 3)
        self.assertAlmostEqual(jev["ece_10bin"], 0.4)
        self.assertAlmostEqual(jev["brier"], 0.4)
        self.assertAlmostEqual(jev["refusal_rate"], 1 / 3)
        self.assertAlmostEqual(jev["mean_latency_ms"], 20)
        self.assertAlmostEqual(jev["p50_latency_ms"], 20)
        self.assertEqual(jev["total_input_tokens"], 200)
        self.assertEqual(jev["excluded_no_probs"], 1)
        self.assertEqual(jev["input_tokens_missing"], 1)
        self.assertAlmostEqual(jev["exact_agreement"], 1 / 3)
        self.assertAlmostEqual(accuracy(["positive", "positive", "negative"], ["positive", "negative", ""]), 1 / 3)
        self.assertAlmostEqual(
            macro_f1(["positive", "positive", "negative"], ["positive", "negative", ""], LABELS),
            1 / 3,
        )

    def test_ece_bin_edges(self):
        # 0.09 and 0.11 are in different bins. Merging them would score 0.4, not 0.51.
        ece = expected_calibration_error([0.09, 0.11], [1, 0])
        self.assertAlmostEqual(ece, 0.51)
        # The last bin includes 1.0. Dropping it would return nan.
        self.assertAlmostEqual(expected_calibration_error([1.0], [0]), 1.0)
        self.assertAlmostEqual(expected_calibration_error([1.0], [1]), 0.0)

    def test_brier_extremes(self):
        perfect = brier_score([{"negative": 0.0, "positive": 1.0}], ["positive"], LABELS)
        uniform = brier_score([{"negative": 0.5, "positive": 0.5}], ["positive"], LABELS)
        self.assertAlmostEqual(perfect, 0.0)
        self.assertAlmostEqual(uniform, 0.5)

    def test_distribution_without_prediction_counts_for_brier_only(self):
        rows = [_row("0", JEV_MODEL, "positive", "", None, 5, 10, 0.0, error="missing choice")]
        probs = {("0", JEV_MODEL): {"negative": 0.25, "positive": 0.75}}
        scores = score_models(rows, probs, LABELS, (JEV_MODEL,))
        # (0.25-0)^2 + (0.75-1)^2 = 0.125
        self.assertAlmostEqual(scores[JEV_MODEL]["brier"], 0.125)
        self.assertTrue(scores[JEV_MODEL]["ece_10bin"] != scores[JEV_MODEL]["ece_10bin"])
        self.assertEqual(scores[JEV_MODEL]["excluded_no_probs"], 0)
        self.assertEqual(scores[JEV_MODEL]["accuracy"], 0.0)

    def test_exact_agreement_column_is_only_the_first_model(self):
        rows = [
            _row("0", JEV_MODEL, "positive", "positive", 0.8, 10, 100, 0.001),
            _row("0", OPENAI_MODEL, "positive", "negative", 0.6, 10, 100, 0.001),
        ]
        probs = {
            ("0", JEV_MODEL): {"negative": 0.2, "positive": 0.8},
            ("0", OPENAI_MODEL): {"negative": 0.6, "positive": 0.4},
        }
        scores = score_models(rows, probs, LABELS, MODEL_ORDER)
        lines = score_lines(scores, MODEL_ORDER)
        self.assertEqual(lines[0], ["metric", JEV_MODEL, OPENAI_MODEL])
        self.assertEqual(lines[-1][0], "exact_agreement")
        self.assertEqual(lines[-1][1], fmt_metric(scores[JEV_MODEL]["exact_agreement"]))
        self.assertEqual(lines[-1][2], "")
        names = [line[0] for line in lines[1:]]
        self.assertEqual(names[-3:], ["excluded_no_probs", "input_tokens_missing", "exact_agreement"])


def _row(item_id, model, gold, pred, prob_pred, latency, tokens, cost, error=""):
    return {
        "item_id": item_id,
        "model": model,
        "gold": gold,
        "pred": pred,
        "correct": 1 if pred and pred == gold else 0,
        "prob_pred": prob_pred,
        "prob_gold": prob_pred if pred == gold else None,
        "confidence": prob_pred,
        "latency_ms": latency,
        "input_tokens": tokens,
        "cost_usd": cost,
        "error": error,
    }


if __name__ == "__main__":
    unittest.main()
