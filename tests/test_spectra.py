import unittest

import numpy as np

from myco_fusion.spectra import apply_tm, fit_tm, spectral_angle


class SpectraTests(unittest.TestCase):
    def test_affine_map_recovers_independent_targets(self):
        rng = np.random.default_rng(21)
        rgb = rng.uniform(0.1, 0.8, (40, 3))
        truth = np.vstack((rng.uniform(-0.02, 0.02, (3, 27)), np.full((1, 27), 0.5)))
        reference = np.column_stack((rgb, np.ones(len(rgb)))) @ truth
        fitted = fit_tm(rgb, reference, alpha=0)
        unseen = rng.uniform(0.1, 0.8, (8, 3))
        expected = np.column_stack((unseen, np.ones(len(unseen)))) @ truth
        np.testing.assert_allclose(apply_tm(unseen, fitted), expected, atol=1e-12)

    def test_predictions_are_not_clipped(self):
        weights = np.zeros((4, 27))
        weights[-1] = 1.25
        np.testing.assert_allclose(apply_tm([0.1, 0.2, 0.3], weights), 1.25)

    def test_metric_known_angles_and_scale(self):
        self.assertAlmostEqual(spectral_angle([1, 0], [0, 1]), 90)
        self.assertAlmostEqual(spectral_angle([1, 1], [10, 10]), 0, places=5)
        self.assertAlmostEqual(spectral_angle([1e300, 0], [0, 1e300]), 90)

    def test_undefined_and_malformed_inputs_rejected(self):
        for a, b in [([0, 0], [1, 1]), ([float("nan"), 1], [1, 1]), ([1, 1], [1, 1, 1])]:
            with self.assertRaises(ValueError):
                spectral_angle(a, b)
        with self.assertRaises(ValueError):
            fit_tm(np.ones((10, 3)), np.ones((10, 26)))
        with self.assertRaises(ValueError):
            fit_tm(np.ones((10, 3)), np.ones((10, 27)), alpha=-1)
        with self.assertRaises(ValueError):
            apply_tm([1, 2], np.zeros((4, 27)))


if __name__ == "__main__":
    unittest.main()
