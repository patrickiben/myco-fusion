import unittest

import numpy as np

from myco_fusion.intervals import chi2_quantile, median_interval, share_interval


class MedianIntervalTests(unittest.TestCase):
    def test_too_few_units_give_no_interval(self):
        self.assertIsNone(median_interval([1.0]))

    def test_levels_match_the_binomial_formula(self):
        for n, expected in ((2, .5), (3, .75), (4, .875), (5, .9375)):
            result = median_interval(np.arange(n, dtype=float))
            self.assertAlmostEqual(result["level"], expected)
            self.assertFalse(result["reaches_target"])
            self.assertEqual((result["low"], result["high"]), (0, n - 1))

    def test_order_statistics_reach_the_target_from_six_units(self):
        result = median_interval(np.arange(6, dtype=float))
        self.assertAlmostEqual(result["level"], 1 - 2 / 64)
        self.assertTrue(result["reaches_target"])
        # Known textbook value: n=20 uses the 6th and 15th order statistics.
        result = median_interval(np.arange(20, dtype=float))
        self.assertEqual((result["low"], result["high"]), (5, 14))
        self.assertAlmostEqual(result["level"], 0.9586, places=4)

    def test_coverage_holds_on_skewed_data(self):
        rng = np.random.default_rng(3)
        hits = 0
        for _ in range(2000):
            result = median_interval(np.exp(rng.normal(0, 1, 10)))
            hits += result["low"] <= 1 <= result["high"]
        self.assertGreater(hits / 2000, 0.95)

    def test_nonfinite_input_is_rejected(self):
        with self.assertRaises(ValueError):
            median_interval([1, float("nan")])


class HelperTests(unittest.TestCase):
    def test_chi_square_matches_table_values(self):
        for p, df, expected in ((.975, 1, 5.0239), (.025, 2, .0506), (.975, 2, 7.3778), (.975, 60, 83.2977), (.025, 60, 40.4817)):
            self.assertAlmostEqual(chi2_quantile(p, df), expected, places=3)

    def test_wilson_interval(self):
        result = share_interval(10, 20)
        self.assertAlmostEqual(result["low"], .2993, places=3)
        self.assertAlmostEqual(result["high"], .7007, places=3)
        self.assertEqual(share_interval(0, 5)["low"], 0)


if __name__ == "__main__":
    unittest.main()
