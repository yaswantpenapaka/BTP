import math
import unittest

from eval import metrics as M


class MetricTests(unittest.TestCase):
    def test_tvd(self):
        self.assertAlmostEqual(M.tvd([0.5, 0.5], [1.0, 0.0]), 0.5)
        self.assertAlmostEqual(M.tvd([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]), 0.0)

    def test_jsd(self):
        self.assertAlmostEqual(M.jsd([1, 0], [0, 1]), 1.0)
        self.assertAlmostEqual(M.jsd([0.5, 0.5], [1, 0]), 0.311278124459, places=9)
        self.assertAlmostEqual(M.jsd([0.5, 0.5], [1, 0]), M.jsd([1, 0], [0.5, 0.5]))

    def test_kl_directed_and_smoothed(self):
        self.assertAlmostEqual(M.kl([0.5, 0.5], [0.25, 0.75]), 0.20751874963, places=5)
        self.assertNotAlmostEqual(M.kl([0.5, 0.5], [0.25, 0.75]), M.kl([0.25, 0.75], [0.5, 0.5]), places=3)
        self.assertTrue(math.isfinite(M.kl([1, 0], [0, 1])))  # epsilon keeps it finite
        self.assertAlmostEqual(M.kl([0.3, 0.7], [0.3, 0.7]), 0.0)

    def test_agreement_kappa(self):
        a, b = [1, 1, 0, 0], [1, 0, 0, 0]
        self.assertAlmostEqual(M.agreement(a, b), 0.75)
        self.assertAlmostEqual(M.cohen_kappa(a, b), 0.5)
        self.assertAlmostEqual(M.cohen_kappa(["x", "y"], ["x", "y"]), 1.0)
        self.assertIsNone(M.cohen_kappa([], []))

    def test_correlations(self):
        self.assertAlmostEqual(M.pearson([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(M.pearson([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(M.pearson([1, 1, 1], [1, 2, 3]))
        self.assertAlmostEqual(M.spearman([1, 2, 2, 3], [1, 2, 3, 4]), 4.5 / math.sqrt(22.5))
        self.assertAlmostEqual(M.spearman([1, 10, 100], [1, 2, 3]), 1.0)

    def test_mcnemar_exact(self):
        r = M.mcnemar([1, 1, 1, 0, 1], [0, 0, 0, 0, 1])
        self.assertEqual((r["b"], r["c"], r["test"]), (3, 0, "exact"))
        self.assertAlmostEqual(r["p"], 0.25)

    def test_mcnemar_chi2(self):
        ci = [1] * 20 + [0] * 10
        cj = [0] * 20 + [1] * 10
        r = M.mcnemar(ci, cj)
        self.assertEqual((r["b"], r["c"], r["test"]), (20, 10, "chi2_cc"))
        self.assertAlmostEqual(r["stat"], 2.7)
        self.assertAlmostEqual(r["p"], 0.100348, places=5)

    def test_mcnemar_no_discordant(self):
        self.assertEqual(M.mcnemar([1, 0], [1, 0])["p"], 1.0)


if __name__ == "__main__":
    unittest.main()
