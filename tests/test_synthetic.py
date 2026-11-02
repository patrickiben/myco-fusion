import json
import unittest

import numpy as np

from myco_fusion.synthetic import run_synthetic


class SyntheticTests(unittest.TestCase):
    def test_rank_distinguishes_brightness_from_spectral_diversity(self):
        report = run_synthetic(noise=0)
        scenarios = {row["id"]: row for row in report["scenarios"]}
        self.assertEqual(scenarios["matched"]["rank"], 3)
        self.assertEqual(scenarios["intensity_only"]["rank"], 3)
        self.assertIsNone(scenarios["intensity_only"]["condition_number"])
        self.assertEqual(scenarios["spectrally_distinct"]["rank"], 4)
        # Exact recovery is justified only for the noiseless, known, full-rank model.
        self.assertLess(scenarios["spectrally_distinct"]["median_mae"], 1e-12)
        self.assertGreater(scenarios["intensity_only"]["median_mae"], 1e-5)

    def test_metamer_is_physical_and_rgb_indistinguishable(self):
        metamer = run_synthetic()["metamer"]
        self.assertLess(metamer["rgb_difference"], 1e-12)
        self.assertGreater(metamer["spectral_rmse"], 0.05)
        for key in ("first", "second"):
            values = np.asarray(metamer[key])
            self.assertTrue(np.all((values >= 0) & (values <= 1)))

    def test_report_is_deterministic_and_strict_json(self):
        first = run_synthetic(seed=13)
        self.assertEqual(first, run_synthetic(seed=13))
        json.dumps(first, allow_nan=False)
        self.assertEqual(first["kind"], "synthetic")
        self.assertEqual(len(first["bands_nm"]), 27)
        for scenario in first["scenarios"]:
            self.assertEqual(len(scenario["reference"]), 27)
            self.assertEqual(len(scenario["estimate"]), 27)
            self.assertTrue(all(0 <= x <= 1 for x in scenario["reference"]))

    def test_invalid_noise_is_rejected(self):
        for noise in (-.1, float("nan"), .2, True, False, "0.003"):
            with self.assertRaises(ValueError):
                run_synthetic(noise=noise)

    def test_seed_is_an_explicit_bounded_integer(self):
        for seed in (-1, 2**32, True, False, 7.0, "7", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                run_synthetic(seed=seed)
        for seed in (0, 2**32 - 1):
            self.assertEqual(run_synthetic(seed=seed)["seed"], seed)


if __name__ == "__main__":
    unittest.main()
