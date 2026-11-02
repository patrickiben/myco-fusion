import csv
import hashlib
import io
import unittest

from myco_fusion.validation import evaluate_csv


def make_csv(observations):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["specimen_id", "regime", "repeat", "wavelength_nm", "reference", "estimate"])
    for specimen, regime, repeat, reference, estimate in observations:
        for index, wavelength in enumerate(range(420, 681, 10)):
            writer.writerow([specimen, regime, repeat, wavelength,
                             reference[index] if isinstance(reference, list) else reference,
                             estimate[index] if isinstance(estimate, list) else estimate])
    return output.getvalue()


class ValidationTests(unittest.TestCase):
    def test_repeats_do_not_become_independent_specimens(self):
        # A contributes one mean error (0), irrespective of its ten repeats.
        observations = [("A", "R1", i, 0.5, 0.5) for i in range(10)]
        observations.append(("B", "R1", 1, 0.5, 0.7))
        text = make_csv(observations)
        report = evaluate_csv(text)
        result = report["regimes"][0]
        self.assertEqual(result["n_specimens"], 2)
        self.assertEqual(result["n_observations"], 11)
        self.assertAlmostEqual(result["median_mae"], 0.1)
        self.assertFalse(result["median_within_targets"])
        self.assertEqual(result["n_within_targets"], 1)
        self.assertEqual(report["input_sha256"], hashlib.sha256(text.encode()).hexdigest())

    def test_repeats_are_averaged_before_median(self):
        report = evaluate_csv(make_csv([
            ("A", "R1", 1, 0.5, 0.5), ("A", "R1", 2, 0.5, 0.9),
            ("B", "R1", 1, 0.5, 0.6),
        ]))
        self.assertAlmostEqual(report["regimes"][0]["median_mae"], 0.15)

    def test_repeatable_report_and_regime_separation(self):
        text = make_csv([("A", "R1", 1, .5, .5), ("A", "R3", 1, .5, .9)])
        first = evaluate_csv(text)
        self.assertEqual(first, evaluate_csv(text))
        self.assertEqual([x["median_within_targets"] for x in first["regimes"]], [True, False])

    def test_estimates_remain_unbounded(self):
        report = evaluate_csv(make_csv([("A", "R1", 1, .5, 1.5)]))
        self.assertEqual(report["rows"][0]["estimate"][0], 1.5)
        self.assertAlmostEqual(report["regimes"][0]["median_mae"], 1.0)
        self.assertTrue(any("outside [0, 1]" in note for note in report["notes"]))

    def test_synthetic_origin_label_is_preserved_as_unverified(self):
        lines = make_csv([("A", "R1", 1, .5, .6)]).splitlines()
        text = "\n".join([lines[0] + ",data_origin"] + [line + ",synthetic_demo" for line in lines[1:]])
        report = evaluate_csv(text)
        self.assertEqual(report["source_labels"], ["synthetic_demo"])
        self.assertTrue(any("synthetic_demo" in note for note in report["notes"]))

    def test_synthetic_kind_requires_every_row_to_declare_it(self):
        lines = make_csv([("SYNTHETIC_A", "R1", 1, .5, .6)]).splitlines()
        text = "\n".join([lines[0] + ",data_kind"] + [line + ",synthetic" for line in lines[1:]])
        self.assertEqual(evaluate_csv(text)["data_kind"], "synthetic")
        mixed = text.rsplit(",synthetic", 1)[0] + ","
        self.assertEqual(evaluate_csv(mixed)["data_kind"], "user_supplied_unverified")
        self.assertEqual(evaluate_csv("\n".join(lines))["data_kind"], "user_supplied_unverified")

    def test_median_inside_targets_does_not_hide_failing_specimens(self):
        observations = [(f"good-{i}", "R1", 1, .5, .51) for i in range(6)] + [(f"bad-{i}", "R1", 1, .5, .8) for i in range(5)]
        regime = evaluate_csv(make_csv(observations))["regimes"][0]
        self.assertTrue(regime["median_within_targets"])
        self.assertEqual(regime["n_within_targets"], 6)
        self.assertAlmostEqual(regime["share_within_targets"], 6 / 11)
        self.assertLess(regime["share_interval"]["low"], .5)

    def test_interval_states_its_true_level_with_few_specimens(self):
        three = evaluate_csv(make_csv([(name, "R1", 1, .5, .5 + d) for name, d in (("A", .01), ("B", .02), ("C", .04))]))["regimes"][0]
        self.assertAlmostEqual(three["interval_level"], .75)
        self.assertIn("75.0%", three["interval_note"])
        self.assertAlmostEqual(three["mae_interval"]["low"], .01)
        self.assertAlmostEqual(three["mae_interval"]["high"], .04)
        six = evaluate_csv(make_csv([(f"S{i}", "R1", 1, .5, .5 + .01 * (i + 1)) for i in range(6)]))["regimes"][0]
        self.assertAlmostEqual(six["interval_level"], 1 - 2 / 64)
        self.assertIsNone(six["interval_note"])

    def test_duplicate_missing_nonfinite_zero_and_reference_range(self):
        valid = make_csv([("A", "R1", 1, .5, .6)])
        malformed = [valid + valid.splitlines()[1] + "\n",
                     "\n".join(valid.splitlines()[:-1]),
                     valid.replace("0.6", "nan", 1),
                     make_csv([("A", "R1", 1, 0, .5)]),
                     make_csv([("A", "R1", 1, .5, 0)]),
                     make_csv([("A", "R1", 1, 1.1, .5)])]
        for text in malformed:
            with self.subTest(text=text[:80]):
                with self.assertRaises(ValueError):
                    evaluate_csv(text)


if __name__ == "__main__":
    unittest.main()
