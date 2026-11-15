import unittest

import numpy as np

from study.admission_trial import (ESTIMATORS, METRIC_INDEX, SCREEN_COLORS, ForwardModel, GroupBootstrap, Inputs, SmoothRidge,
                                   check_replicate, draw_errors, holm, photos, screen_choice, simplex_weights, stack, verdicts)
from study.pipeline import SIM_NM, Noise, Ridge


def toy_inputs(n_groups=24, per_group=3, seed=0):
    """Smooth synthetic spectra in [0, 1] and random camera curves; never the real data."""
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    centers = rng.uniform(420, 680, (n, 2))
    widths = rng.uniform(30, 90, (n, 2))
    heights = rng.uniform(0.1, 0.5, (n, 2))
    bumps = heights[:, :, None] * np.exp(-0.5 * ((SIM_NM[None, None, :] - centers[:, :, None]) / widths[:, :, None]) ** 2)
    sim = np.clip(0.05 + bumps.sum(axis=1), 0, 1)
    camera = rng.uniform(0.1, 1.0, (3, len(SIM_NM)))
    camera /= camera.sum(axis=1, keepdims=True)
    flash = np.ones(len(SIM_NM))
    screens = {color: np.exp(-0.5 * ((SIM_NM - center) / 20) ** 2) * 3 for color, (center, _) in SCREEN_COLORS.items()}
    return Inputs([f"s{i}" for i in range(n)], sim, sim[:, METRIC_INDEX], np.repeat(np.arange(n_groups), per_group),
                  camera, flash, camera[::-1].copy(), screens, Noise(0.005, 0.02))


class SimplexWeightsTests(unittest.TestCase):
    def test_recovers_a_convex_combination(self):
        rng = np.random.default_rng(1)
        a, b, c = (rng.normal(size=(40, 27)) for _ in range(3))
        np.testing.assert_allclose(simplex_weights([a, b, c], 0.3 * a + 0.7 * b), [0.3, 0.7, 0.0], atol=1e-8)

    def test_matches_a_fine_grid_and_stays_on_the_simplex(self):
        rng = np.random.default_rng(2)
        target = rng.normal(size=(30, 27))
        preds = [target + rng.normal(scale=s, size=target.shape) + shift for s, shift in ((0.5, 0.2), (0.8, -0.1), (1.0, 0.0))]
        weights = simplex_weights(preds, target)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertTrue((weights >= 0).all())

        def loss(v):
            return np.sum((sum(vi * p for vi, p in zip(v, preds)) - target) ** 2)

        grid = [(i / 100, j / 100, 1 - (i + j) / 100) for i in range(101) for j in range(101 - i)]
        self.assertLessEqual(loss(weights), min(loss(g) for g in grid) + 1e-9)

    def test_identical_predictions_still_give_valid_weights(self):
        rng = np.random.default_rng(3)
        target = rng.normal(size=(20, 27))
        weights = simplex_weights([target, target, target + 1], target)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertAlmostEqual(weights[2], 0.0)


class StatisticsTests(unittest.TestCase):
    def test_holm_matches_a_worked_example(self):
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])

    def test_bootstrap_contrast(self):
        boot = GroupBootstrap(np.repeat(np.arange(20), 3), np.random.default_rng(0), resamples=500)
        zero = boot.contrast(np.zeros(60))
        self.assertEqual((zero["low"], zero["high"], zero["p_one_sided"]), (0.0, 0.0, 1.0))
        positive = boot.contrast(np.full(60, 0.01))
        self.assertAlmostEqual(positive["p_one_sided"], 1 / 501)
        self.assertGreater(positive["low"], 0)
        mixed = np.random.default_rng(1).normal(size=60)
        self.assertAlmostEqual(boot.contrast(mixed)["mean"], mixed.mean())


class EstimatorTests(unittest.TestCase):
    def test_forward_model_recovers_spectra_from_a_three_direction_description(self):
        rng = np.random.default_rng(4)
        basis = np.linalg.qr(rng.normal(size=(len(SIM_NM), 3)))[0]
        spectra = 0.4 + rng.normal(scale=0.1, size=(200, 3)) @ basis.T
        operator = rng.uniform(0.0, 0.05, size=(3, len(SIM_NM)))
        model = ForwardModel(3, operator, Noise(1e-7, 0.0)).fit(spectra)
        np.testing.assert_allclose(model.predict(spectra @ operator.T), spectra[:, METRIC_INDEX], atol=1e-5)

    def test_forward_model_falls_back_to_the_mean_under_heavy_noise(self):
        rng = np.random.default_rng(5)
        spectra = rng.uniform(0.1, 0.6, (100, len(SIM_NM)))
        operator = rng.uniform(0.0, 0.05, size=(3, len(SIM_NM)))
        model = ForwardModel(5, operator, Noise(1e3, 0.0)).fit(spectra)
        estimate = model.predict(spectra[:4] @ operator.T)
        np.testing.assert_allclose(estimate, np.tile(spectra.mean(axis=0)[METRIC_INDEX], (4, 1)), atol=1e-6)

    def test_smoothing_ridge_reduces_to_affine_ridge_and_smooths_when_strong(self):
        rng = np.random.default_rng(6)
        x = rng.uniform(0, 1, (60, 3))
        y = np.clip(x @ rng.uniform(0, 0.3, (3, 27)) + rng.normal(0, 0.05, (60, 27)), 0, 1)
        np.testing.assert_allclose(SmoothRidge(1e-4, 1e-12).fit(x, y).predict(x), Ridge(1e-4).fit(x, y).predict(x), atol=1e-6)
        rough = np.abs(np.diff(SmoothRidge(1e-4, 1e-12).fit(x, y).predict(x), 2, axis=1)).mean()
        smooth = np.abs(np.diff(SmoothRidge(1e-4, 100.0).fit(x, y).predict(x), 2, axis=1)).mean()
        self.assertLess(smooth, rough / 5)


class RuleTests(unittest.TestCase):
    def test_rule_admits_a_planted_component_and_rejects_a_useless_one(self):
        rng = np.random.default_rng(7)
        y_dev, y_held = rng.uniform(0.1, 0.6, (120, 27)), rng.uniform(0.1, 0.6, (90, 27))
        affine = (y_dev + rng.normal(0, 0.05, y_dev.shape), y_held + rng.normal(0, 0.05, y_held.shape))
        preds = {
            "affine": affine,
            "other": (y_dev + rng.normal(0, 0.06, y_dev.shape), y_held + rng.normal(0, 0.06, y_held.shape)),
            "planted": (affine[0] + 0.5 * (y_dev - affine[0]), affine[1] + 0.5 * (y_held - affine[1])),
            "useless": (affine[0][rng.permutation(120)], affine[1][rng.permutation(90)]),
        }
        tested = [("planted",), ("useless",)]
        maes, weights, _ = draw_errors(preds, y_dev, y_held, [("other",), *tested], tested, rng)
        boot = GroupBootstrap(np.repeat(np.arange(30), 3), np.random.default_rng(8), resamples=500)
        table, family = verdicts(maes, tested, [("other",)], boot)
        self.assertTrue(table["planted"]["passes"])
        self.assertFalse(table["useless"]["condition_2"])
        self.assertFalse(table["useless"]["passes"])
        self.assertIn("other", family)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertFalse(verdicts(maes, tested, [("other",)], boot, frozen_ok=False)[0]["planted"]["passes"])

    def test_a_zero_weight_component_has_exactly_zero_unique_gain(self):
        rng = np.random.default_rng(10)
        y_dev, y_held = rng.uniform(0.1, 0.6, (80, 27)), rng.uniform(0.1, 0.6, (60, 27))
        affine = (y_dev + rng.normal(0, 0.05, y_dev.shape), y_held + rng.normal(0, 0.05, y_held.shape))
        preds = {
            "affine": affine,
            "other": (y_dev + rng.normal(0, 0.05, y_dev.shape), y_held + rng.normal(0, 0.05, y_held.shape)),
            "far_off": (affine[0].copy(), affine[1].copy()),  # an exact copy adds nothing, so its weight is zero
        }
        maes, weights, _ = draw_errors(preds, y_dev, y_held, [("far_off",), ("other",)], [("far_off",)], rng)
        self.assertEqual(weights["far_off"], 0.0)
        np.testing.assert_array_equal(maes["without:far_off"], maes["combination"])
        boot = GroupBootstrap(np.repeat(np.arange(20), 3), np.random.default_rng(11), resamples=500)
        table, _ = verdicts(maes, [("far_off",)], [("other",)], boot)
        self.assertEqual(table["far_off"]["unique_gain"]["p_one_sided"], 1.0)
        self.assertFalse(table["far_off"]["condition_2"])

    def test_rounding_level_differences_count_as_zero(self):
        boot = GroupBootstrap(np.repeat(np.arange(20), 3), np.random.default_rng(12), resamples=500)
        tiny = boot.contrast(np.full(60, 3e-15))
        self.assertEqual((tiny["mean"], tiny["low"], tiny["high"], tiny["p_one_sided"]), (0.0, 0.0, 0.0, 1.0))
        self.assertGreater(boot.contrast(np.full(60, 1e-6))["low"], 0)

    def test_stack_and_check_replicate_run_end_to_end_on_toy_data(self):
        inputs = toy_inputs()
        x = photos(inputs, inputs.flash, np.random.default_rng(9))
        dev, held = np.arange(0, 36), np.arange(36, 72)
        oof, out = stack(ESTIMATORS, inputs, x, dev, held, np.repeat(np.arange(4), 9), ("test",))
        for kind in ESTIMATORS:
            self.assertEqual(oof[kind].shape, (36, 27))
            self.assertEqual(out[kind].shape, (36, 27))
            self.assertTrue(np.isfinite(out[kind]).all())
        record = check_replicate(0, inputs)
        self.assertEqual(set(record["configs"]), {"useless", "planted_0.05", "planted_0.1", "planted_0.2", "planted_0.5", "duplicates"})
        self.assertIn("duplicate_1+duplicate_2", record["configs"]["duplicates"]["tested"])
        self.assertEqual(set(record["configs"]["useless"]["family"]), {"polynomial", "kernel", "smoothing", "forward_model"})

    def test_trial_draws_and_analysis_run_on_toy_data(self):
        from unittest.mock import patch

        from study.admission_trial import CANDIDATES, analyze_trial, trial_draws

        inputs = toy_inputs()
        dev, held = np.arange(0, 36), np.arange(36, 72)
        with patch("study.admission_trial.DRAWS", 2):
            for shot in (False, True):
                maes, shares, weights, best = trial_draws(inputs, dev, held, np.repeat(np.arange(4), 9), shot)
                self.assertEqual(len(weights), 2)
                self.assertTrue(set(best) <= set(ESTIMATORS))
                analysis = analyze_trial(maes, shares, inputs.groups[held], "green", True, False, "toy")
                self.assertEqual(set(analysis["components"]), set(CANDIDATES))
                self.assertIn("vs_blue", analysis["screen_choice"])
                self.assertIn("combination", analysis["systems"])

    def test_screen_choice_never_predicts_worse_than_flash_alone(self):
        choice, predicted = screen_choice(toy_inputs(), np.arange(48))
        self.assertIn(choice, SCREEN_COLORS)
        for color in SCREEN_COLORS:
            self.assertLessEqual(predicted[color], predicted["flash alone"] + 1e-12)


if __name__ == "__main__":
    unittest.main()
