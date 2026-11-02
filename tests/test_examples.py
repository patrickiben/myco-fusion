import unittest

from myco_fusion.examples import validation_csv
from myco_fusion.validation import evaluate_csv


class ExampleTests(unittest.TestCase):
    def test_validation_example_is_labeled_and_supports_a_95_percent_interval(self):
        report = evaluate_csv(validation_csv())
        self.assertEqual(report["data_kind"], "synthetic")
        self.assertEqual(report["uncertainty_method"], "session_cluster")
        self.assertEqual(report["source_labels"], ["simulated_camera_on_measured_reflectance"])
        for regime in report["regimes"]:
            self.assertEqual(regime["n_sessions"], 6)
            self.assertGreaterEqual(regime["interval_level"], .95)
        self.assertEqual(report["paired_r1_r3"]["n_specimens"], 48)


if __name__ == "__main__":
    unittest.main()
