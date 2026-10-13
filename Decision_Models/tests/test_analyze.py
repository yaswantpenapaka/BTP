"""Hand-checked paired tests. No network calls and no writes to outputs/."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze import (
    DataError,
    Paired,
    analyze_paired,
    ci95,
    ece_batch,
    holm_bonferroni,
    load_paired,
    mcnemar_p,
    outcome_counts,
    paired_se,
    read_item_rows_until_blank,
    tost,
    unpaired_se,
)
from metrics import brier_score, expected_calibration_error
from tasks import JEV_MODEL, OPENAI_MODEL

LABELS = ("negative", "positive")


class McNemarTests(unittest.TestCase):
    def test_hand_worked_two_by_two(self):
        # Bin(6, 0.5). P(0) = P(6) = 1/64, and those are the only outcomes
        # whose probability is at most P(0). Two-sided p = 2/64.
        self.assertAlmostEqual(_exact_two_sided(0, 6), 2 / 64)
        self.assertAlmostEqual(mcnemar_p(0, 6), 2 / 64)
        # P(1) = P(5) = 6/64. Outcomes at least that rare: 0, 1, 5, 6. Total 14/64.
        self.assertAlmostEqual(_exact_two_sided(1, 5), 14 / 64)
        self.assertAlmostEqual(mcnemar_p(1, 5), 14 / 64)
        self.assertAlmostEqual(mcnemar_p(5, 5), 1.0)
        self.assertAlmostEqual(mcnemar_p(0, 0), 1.0)


class TostTests(unittest.TestCase):
    def test_constructed_inside_and_outside_the_margin(self):
        inside = tost(0.0, 0.01, 0.02)
        self.assertEqual((0.0 + 0.02) / 0.01, 2.0)
        self.assertAlmostEqual(inside["p_lower"], norm.sf(2.0))
        self.assertAlmostEqual(inside["p_upper"], norm.cdf(-2.0))
        self.assertAlmostEqual(inside["p_tost"], norm.sf(2.0))
        self.assertGreater(inside["ci90_low"], -0.02)
        self.assertLess(inside["ci90_high"], 0.02)
        self.assertTrue(inside["equivalent"])

        outside = tost(0.015, 0.01, 0.02)
        self.assertAlmostEqual(outside["p_upper"], norm.cdf((0.015 - 0.02) / 0.01))
        self.assertGreater(outside["ci90_high"], 0.02)
        self.assertFalse(outside["equivalent"])
        self.assertAlmostEqual(outside["p_tost"], outside["p_upper"])

        point = tost(0.0, 0.0, 0.02)
        self.assertTrue(point["equivalent"])
        self.assertEqual(point["p_tost"], 0.0)
        boundary = tost(0.02, 0.0, 0.02)
        self.assertFalse(boundary["equivalent"])
        self.assertAlmostEqual(boundary["p_tost"], 0.5)


class MetricHandTests(unittest.TestCase):
    def test_ece_and_brier_on_a_tiny_set(self):
        # Jev confidences 1.0 correct, 0.7 correct, 0.9 wrong, 0.55 wrong.
        # Bin [0.5, 0.6): 0.25 * |0 - 0.55| = 0.1375
        # Bin [0.7, 0.8): 0.25 * |1 - 0.7| = 0.075
        # Bin [0.9, 1]: two items, accuracy 0.5, confidence 0.95, weight 0.5, contribution 0.225
        # ECE = 0.4375
        conf = np.array([1.0, 0.7, 0.9, 0.55])
        hit = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(expected_calibration_error(conf.tolist(), hit.tolist()), 0.4375)
        identity = np.arange(4, dtype=int).reshape(1, 4)
        self.assertAlmostEqual(float(ece_batch(conf, hit.astype(float), identity)[0]), 0.4375)
        self.assertAlmostEqual(expected_calibration_error([0.09, 0.11], [1, 0]), 0.51)
        self.assertAlmostEqual(float(ece_batch(np.array([0.09, 0.11]), np.array([1.0, 0.0]), np.array([[0, 1]]))[0]), 0.51)
        self.assertAlmostEqual(expected_calibration_error([1.0], [0]), 1.0)
        self.assertAlmostEqual(expected_calibration_error([0.8, 0.6], [1, 0]), 0.4)

        jev_dists = [
            {"negative": 0.0, "positive": 1.0},
            {"negative": 0.3, "positive": 0.7},
            {"negative": 0.1, "positive": 0.9},
            {"negative": 0.55, "positive": 0.45},
        ]
        golds = ["positive", "positive", "negative", "positive"]
        # Squared errors: 0, 0.18, 1.62, 0.605. Mean = 0.60125.
        self.assertAlmostEqual(brier_score(jev_dists, golds, list(LABELS)), 0.60125)

        result = analyze_paired(_tiny_paired(), margin=0.02, n_resamples=30, seed=0)
        self.assertEqual(result["n"], 4)
        self.assertEqual(
            result["both_right"] + result["only_jev"] + result["only_luna"] + result["both_wrong"],
            result["n"],
        )
        self.assertAlmostEqual(result["ece_jev"], 0.4375)
        self.assertAlmostEqual(result["brier_jev"], 0.60125)
        self.assertAlmostEqual(result["acc_jev"], 0.5)
        self.assertAlmostEqual(result["delta"], 0.0)
        self.assertAlmostEqual(result["paired_se"], math.sqrt(2 / 3) / 2)
        low, high = ci95(0.0, math.sqrt(2 / 3) / 2)
        self.assertAlmostEqual(result["ci95_low"], low)
        self.assertAlmostEqual(result["ci95_high"], high)
        self.assertAlmostEqual(low, -1.96 * math.sqrt(2 / 3) / 2)
        expected_logloss = (
            0.0 + -math.log(0.7) + -math.log(0.1) + -math.log(0.45)
        ) / 4
        self.assertAlmostEqual(result["logloss_jev"], expected_logloss)
        self.assertEqual(result["shared_patterns"], [("positive", "negative", 1)])

    def test_counts_add_to_n_with_a_different_wrong_label(self):
        frame = pd.DataFrame(
            {
                "correct_jev": [1, 1, 0, 0, 0],
                "correct_luna": [1, 0, 1, 0, 0],
                "pred_jev": ["joy", "joy", "sadness", "sadness", "anger"],
                "pred_luna": ["joy", "sadness", "joy", "sadness", "sadness"],
            }
        )
        counts = outcome_counts(frame)
        self.assertEqual(counts["n"], 5)
        self.assertEqual(counts["both_right"], 1)
        self.assertEqual(counts["only_jev"], 1)
        self.assertEqual(counts["only_luna"], 1)
        self.assertEqual(counts["both_wrong"], 2)
        self.assertEqual(
            counts["both_right"] + counts["only_jev"] + counts["only_luna"] + counts["both_wrong"],
            counts["n"],
        )
        self.assertEqual(counts["agreement"], 2)
        self.assertEqual(counts["shared_mistakes"], 1)
        self.assertEqual(counts["both_wrong_different"], 1)

    def test_missing_probability_is_reported_and_the_item_stays_in_n(self):
        paired = _tiny_paired()
        paired.dist_jev[3] = None
        paired.missing.append("item 3 jev-1.13.0: full probability distribution is missing")
        result = analyze_paired(paired, margin=0.02, n_resamples=10, seed=0)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["n_missing_prob"], 1)
        self.assertTrue(math.isnan(result["brier_jev"]))
        self.assertTrue(math.isnan(result["ece_jev"]))
        self.assertEqual(result["both_right"] + result["only_jev"] + result["only_luna"] + result["both_wrong"], 4)


class IntervalTests(unittest.TestCase):
    def test_paired_and_unpaired_standard_errors(self):
        differences = np.array([1.0, -1.0, 0.0, 0.0])
        self.assertAlmostEqual(paired_se(differences), math.sqrt(2 / 3) / 2)
        self.assertAlmostEqual(unpaired_se(0.5, 0.25, 4), math.sqrt(0.5 * 0.5 / 4 + 0.25 * 0.75 / 4))
        low, high = ci95(0.1, 0.01)
        self.assertAlmostEqual(low, 0.0804)
        self.assertAlmostEqual(high, 0.1196)

    def test_holm_keeps_order_and_monotonicity(self):
        # Sorted p: 0.01, 0.03, 0.04, 0.20 with multipliers 4, 3, 2, 1.
        # Raw adjusted values 0.04, 0.09, 0.08, 0.20 become 0.04, 0.09, 0.09, 0.20.
        adjusted = holm_bonferroni([0.01, 0.04, 0.03, 0.20])
        self.assertAlmostEqual(adjusted[0], 0.04)
        self.assertAlmostEqual(adjusted[1], 0.09)
        self.assertAlmostEqual(adjusted[2], 0.09)
        self.assertAlmostEqual(adjusted[3], 0.20)

    def test_ece_batch_matches_the_reference_on_bin_edges(self):
        rng = np.random.default_rng(0)
        conf = np.concatenate([rng.random(50), np.array([0.0, 0.1, 0.3, 0.9, 1.0])])
        hit = np.concatenate([rng.integers(0, 2, size=50), np.array([1, 0, 1, 0, 0])]).astype(float)
        identity = np.arange(conf.size, dtype=int).reshape(1, -1)
        batched = float(ece_batch(conf, hit, identity)[0])
        reference = expected_calibration_error(conf.tolist(), hit.astype(int).tolist())
        self.assertAlmostEqual(batched, reference, places=12)


class ReaderTests(unittest.TestCase):
    def test_reader_stops_at_the_score_block(self):
        text = (
            "item_id,model,gold,pred,correct,prob_pred,prob_gold,confidence,latency_ms,input_tokens,cost_usd,error\n"
            f"0,{JEV_MODEL},positive,positive,1,1,1,1,10,100,0.001,\n"
            f"0,{OPENAI_MODEL},positive,positive,1,1,1,1,10,100,0.001,\n"
            "\n"
            f"metric,{JEV_MODEL},{OPENAI_MODEL}\n"
            f"999,{JEV_MODEL},positive,negative,0,0.2,0.2,0.2,10,100,0.001,\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.csv"
            path.write_text(text, encoding="utf-8", newline="\n")
            rows = read_item_rows_until_blank(path)
        self.assertEqual([row["item_id"] for row in rows], ["0", "0"])

    def test_gold_mismatch_lists_the_item_and_does_not_require_probabilities(self):
        text = (
            "item_id,model,gold,pred,correct,prob_pred,prob_gold,confidence,latency_ms,input_tokens,cost_usd,error\n"
            f"5,{JEV_MODEL},positive,positive,1,1,1,1,10,100,0.001,\n"
            f"5,{OPENAI_MODEL},negative,negative,1,1,1,1,10,100,0.001,\n"
            "\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            path.write_text(text, encoding="utf-8", newline="\n")
            with self.assertRaises(DataError) as caught:
                load_paired(path, path.with_name("missing.jsonl"), "sst2")
        self.assertIn("item 5 gold mismatch", str(caught.exception))
        self.assertIn("positive", str(caught.exception))
        self.assertIn("negative", str(caught.exception))


def _exact_two_sided(only_jev: int, only_luna: int) -> float:
    n = only_jev + only_luna
    target = comb(n, only_jev)
    total = sum(comb(n, index) for index in range(n + 1) if comb(n, index) <= target)
    return total / 2**n


def _tiny_paired() -> Paired:
    frame = pd.DataFrame(
        [
            {
                "item_id": "0",
                "gold": "positive",
                "pred_jev": "positive",
                "pred_luna": "positive",
                "correct_jev": 1,
                "correct_luna": 1,
                "prob_pred_jev": 1.0,
                "prob_pred_luna": 0.8,
            },
            {
                "item_id": "1",
                "gold": "positive",
                "pred_jev": "positive",
                "pred_luna": "negative",
                "correct_jev": 1,
                "correct_luna": 0,
                "prob_pred_jev": 0.7,
                "prob_pred_luna": 0.6,
            },
            {
                "item_id": "2",
                "gold": "negative",
                "pred_jev": "positive",
                "pred_luna": "negative",
                "correct_jev": 0,
                "correct_luna": 1,
                "prob_pred_jev": 0.9,
                "prob_pred_luna": 0.8,
            },
            {
                "item_id": "3",
                "gold": "positive",
                "pred_jev": "negative",
                "pred_luna": "negative",
                "correct_jev": 0,
                "correct_luna": 0,
                "prob_pred_jev": 0.55,
                "prob_pred_luna": 0.7,
            },
        ]
    )
    dist_jev = [
        {"negative": 0.0, "positive": 1.0},
        {"negative": 0.3, "positive": 0.7},
        {"negative": 0.1, "positive": 0.9},
        {"negative": 0.55, "positive": 0.45},
    ]
    dist_luna = [
        {"negative": 0.2, "positive": 0.8},
        {"negative": 0.6, "positive": 0.4},
        {"negative": 0.8, "positive": 0.2},
        {"negative": 0.7, "positive": 0.3},
    ]
    return Paired("tiny", frame, dist_jev, dist_luna, LABELS)


if __name__ == "__main__":
    unittest.main()
