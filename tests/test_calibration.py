import copy
import unittest
import numpy as np
from myco_fusion.calibration import calibration_example, canonical_digest, fit_calibration, predict_calibration


class CalibrationTests(unittest.TestCase):
    def test_heldout_predictions_beat_training_mean(self):
        result = fit_calibration(calibration_example())
        self.assertEqual(result["counts"]["test"]["specimens"], 36)
        self.assertLess(result["test_mean_specimen_mae"], result["baseline_mean_specimen_mae"] / 2)
        self.assertLess(result["model_vs_median_baseline"]["median_difference"], 0)
        self.assertLess(result["model_vs_median_baseline"]["interval"]["high"], 0)
        self.assertEqual(result["data_kind"], "synthetic")
        model = dict(result["model"]); digest = model.pop("model_sha256")
        self.assertEqual(digest, canonical_digest(model))

    def test_test_references_do_not_change_fit_or_tuning(self):
        data = calibration_example()
        before = fit_calibration(data)
        for row in data["observations"]:
            if row["split"] == "test": row["reference"] = [.7] * 27
        after = fit_calibration(data)
        self.assertEqual(before["model"], after["model"])
        self.assertEqual(before["tuning"], after["tuning"])
        self.assertGreater(after["test_mean_specimen_mae"], before["test_mean_specimen_mae"])

    def test_specimen_and_session_leakage_blocked(self):
        for key, message in (("specimen_id", "Specimen leakage"), ("session_id", "Session leakage")):
            data = calibration_example(); data["observations"][-1][key] = data["observations"][0][key]
            with self.assertRaisesRegex(ValueError, message): fit_calibration(data)

    def test_duplicate_and_nonlinear_inputs_blocked(self):
        data = calibration_example(); data["observations"].append(copy.deepcopy(data["observations"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"): fit_calibration(data)
        data = calibration_example(); data["rgb_encoding"] = "srgb"
        with self.assertRaisesRegex(ValueError, "rgb_encoding"): fit_calibration(data)

    def test_rank_deficiency_and_nonfinite_rejected(self):
        data = calibration_example()
        for row in data["observations"]:
            if row["split"] == "train": row["rgb"] = [.5] * 3
        with self.assertRaisesRegex(ValueError, "rank deficient"): fit_calibration(data)
        data = calibration_example(); data["observations"][0]["rgb"][0] = float('nan')
        with self.assertRaisesRegex(ValueError, "finite"): fit_calibration(data)

    def test_no_validation_uses_fixed_alpha(self):
        data = calibration_example(); data["observations"] = [r for r in data["observations"] if r["split"] != "validation"]
        result = fit_calibration(data)
        self.assertEqual(result["model"]["alpha"], 1e-4)
        self.assertEqual(result["tuning"], [])

    def test_repeats_do_not_reweight_the_fit(self):
        data = calibration_example(); first = fit_calibration(data)
        specimen = data["observations"][0]["specimen_id"]
        for i, row in enumerate(list(data["observations"])):
            if row["specimen_id"] == specimen:
                extra = copy.deepcopy(row); extra["observation_id"] += "-extra"
                data["observations"].append(extra)
        repeated = fit_calibration(data)
        np.testing.assert_allclose(first["model"]["weights"], repeated["model"]["weights"], atol=1e-12)

    def test_session_uncertainty_not_invented_for_single_session(self):
        data = calibration_example(); data["uncertainty"] = "session_cluster"
        for row in data["observations"]:
            if row["split"] == "test": row["session_id"] = "one-test-session"
        report = fit_calibration(data)
        for regime in report["evaluation"]["regimes"]:
            self.assertIsNone(regime["mae_interval"])
            self.assertEqual(regime["n_sessions"], 1)
            self.assertTrue(regime["interval_note"])

    def test_example_is_measured_reflectance_with_labeled_simulated_rgb(self):
        data = calibration_example()
        self.assertEqual(data["data_kind"], "synthetic")
        self.assertIn("simulated camera", data["device"])
        report = fit_calibration(data)
        self.assertEqual(report["evaluation"]["uncertainty_method"], "session_cluster")
        for regime in report["evaluation"]["regimes"]:
            self.assertGreaterEqual(regime["interval_level"], .95)
            self.assertGreater(regime["median_mae"], .005)  # Unlike an exact-model demo, real spectra leave error.

    def test_exported_model_predicts_without_refitting_and_keeps_provenance(self):
        data = calibration_example(); report = fit_calibration(data)
        row = next(r for r in data["observations"] if r["split"] == "test")
        capture = {k: data[k] for k in ("device", "pipeline_id", "rgb_encoding", "data_kind")}
        capture["observations"] = [{"observation_id": row["observation_id"], "rgb": row["rgb"]}]
        predicted = predict_calibration(report["model"], capture)
        self.assertEqual(predicted["evidence_type"], "inferred")
        self.assertEqual(predicted["model_sha256"], report["model"]["model_sha256"])
        expected = np.append(row["rgb"], 1) @ np.asarray(report["model"]["weights"])
        np.testing.assert_allclose(predicted["predictions"][0]["estimated_reflectance"], expected)
        changed = copy.deepcopy(report["model"]); changed["weights"][0][0] += .01
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"): predict_calibration(changed, capture)
        capture["pipeline_id"] = "different-decoding"
        with self.assertRaisesRegex(ValueError, "pipeline_id"): predict_calibration(report["model"], capture)

    def test_model_cannot_silently_cross_synthetic_physical_boundary(self):
        data = calibration_example(); model = fit_calibration(data)["model"]
        capture = {k: data[k] for k in ("device", "pipeline_id", "rgb_encoding")}
        capture["observations"] = [{"observation_id": "new", "rgb": [.5, .5, .5]}]
        with self.assertRaisesRegex(ValueError, "Synthetic"): predict_calibration(model, capture)
        data["data_kind"] = "measured"
        with self.assertRaisesRegex(ValueError, "synthetic_linear"): fit_calibration(data)
