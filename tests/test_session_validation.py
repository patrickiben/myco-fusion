import csv
import io
import unittest

from myco_fusion.validation import evaluate_csv


def session_csv(observations, fields=("session_id", "spot_id")):
    """Observations are specimen, regime, repeat, session, spot, estimate."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["specimen_id", "regime", "repeat", "wavelength_nm", "reference", "estimate", *fields])
    for specimen, regime, repeat, session, spot, estimate in observations:
        for wavelength in range(420, 681, 10):
            metadata = {"session_id": session, "spot_id": spot}
            writer.writerow([specimen, regime, repeat, wavelength, 0.5, estimate,
                             *[metadata[field] for field in fields]])
    return output.getvalue()


class SessionValidationTests(unittest.TestCase):
    def test_repeat_numbers_can_recur_in_sessions_and_spots(self):
        text = session_csv([
            ("A", "R1", 1, "day1", "cap", .5),
            ("A", "R1", 1, "day1", "stipe", .6),
            ("A", "R1", 1, "day2", "cap", .7),
            ("B", "R1", 1, "day2", "cap", .9),
        ])
        report = evaluate_csv(text)
        regime = report["regimes"][0]
        self.assertEqual(regime["n_specimens"], 2)
        self.assertEqual(regime["n_observations"], 4)
        self.assertEqual(regime["n_sessions"], 2)
        self.assertEqual(regime["interval_units"], 2)
        self.assertAlmostEqual(regime["median_mae"], .25)
        self.assertAlmostEqual(regime["worst_specimen_mae"], .4)
        self.assertAlmostEqual(regime["p90_specimen_mae"], .37)
        self.assertAlmostEqual(report["specimen_summaries"][0]["mean_mae"], .1)
        self.assertEqual(report["specimen_summaries"][0]["session_ids"], ["day1", "day2"])
        self.assertTrue(any("multiple sessions" in note for note in report["interval_notes"]))
        sessions = {r["session_id"]: r for r in report["session_summaries"]}
        self.assertAlmostEqual(sessions["day1"]["median_mae"], .05)
        self.assertAlmostEqual(sessions["day2"]["median_mae"], .3)
        self.assertEqual(sessions["day1"]["n_specimens"], 1)
        self.assertEqual(sessions["day1"]["n_observations"], 2)

    def test_session_interval_reflects_shared_session_shifts(self):
        observations = []
        for session, error in [("day1", 0), ("day2", .1), ("day3", .2)]:
            for specimen in range(20):
                # Two repeats each do not turn three sessions into 120 units.
                for repeat in (1, 2):
                    observations.append((f"{session}-{specimen}", "R1", repeat, session, "cap", .5 + error))
        text = session_csv(observations)
        default = evaluate_csv(text, uncertainty="specimen")
        clustered = evaluate_csv(text, uncertainty="session_cluster")
        self.assertEqual(clustered, evaluate_csv(text, uncertainty="session_cluster"))
        ordinary, result = default["regimes"][0], clustered["regimes"][0]
        self.assertAlmostEqual(result["median_mae"], ordinary["median_mae"])
        self.assertAlmostEqual(result["median_mae"], .1)
        self.assertAlmostEqual(result["mae_interval"]["low"], 0)
        self.assertAlmostEqual(result["mae_interval"]["high"], .2)
        self.assertAlmostEqual(result["interval_level"], .75)  # Three sessions cannot support 95%.
        self.assertLess(ordinary["mae_interval"]["high"] - ordinary["mae_interval"]["low"], .25)
        self.assertEqual(result["interval_units"], 3)
        self.assertEqual(result["interval_unit"], "session")
        self.assertEqual(clustered["counts"], {
            "n_observations": 120, "n_specimens": 60, "n_sessions": 3, "n_specimen_regime_units": 60,
        })

    def test_unequal_cluster_sizes_keep_specimen_weighted_point_estimand(self):
        text = session_csv([
            ("A", "R1", 1, "small", "cap", .5),
            ("B", "R1", 1, "large", "cap", .7),
            ("C", "R1", 1, "large", "cap", .7),
            ("D", "R1", 1, "large", "cap", .7),
        ])
        result = evaluate_csv(text, uncertainty="session_cluster")["regimes"][0]
        self.assertAlmostEqual(result["median_mae"], .2)  # Not median of two session medians (.1).
        self.assertAlmostEqual(result["mae_interval"]["low"], 0)
        self.assertAlmostEqual(result["mae_interval"]["high"], .2)
        self.assertAlmostEqual(result["interval_level"], .5)

    def test_single_session_interval_is_unavailable(self):
        text = session_csv([
            ("A", "R1", 1, "only-day", "cap", .5),
            ("B", "R1", 1, "only-day", "cap", .9),
        ])
        report = evaluate_csv(text, uncertainty="session_cluster")
        result = report["regimes"][0]
        self.assertIsNone(result["sam_interval"])
        self.assertIsNone(result["mae_interval"])
        self.assertIn("two observed sessions", result["interval_note"])
        self.assertEqual(result["n_specimens"], 2)
        self.assertEqual(result["interval_units"], 1)
        self.assertAlmostEqual(result["median_mae"], .2)

    def test_crossed_specimens_are_rejected_across_regimes(self):
        text = session_csv([
            ("A", "R1", 1, "day1", "cap", .5),
            ("A", "R3", 1, "day2", "cap", .6),
        ])
        with self.assertRaisesRegex(ValueError, "Crossed specimens"):
            evaluate_csv(text, uncertainty="session_cluster")
        report = evaluate_csv(text)
        self.assertEqual(report["counts"]["n_specimens"], 1)
        self.assertEqual(report["counts"]["n_specimen_regime_units"], 2)
        self.assertEqual(report["paired_r1_r3"]["n_specimens"], 1)

    def test_missing_session_column_rejected_only_for_cluster_method(self):
        text = session_csv([("A", "R1", 1, "ignored", "ignored", .6)], fields=())
        result = evaluate_csv(text)
        self.assertEqual(result["uncertainty_method"], "specimen")
        self.assertEqual(result["uncertainty_requested"], "auto")
        self.assertIsNone(result["regimes"][0]["n_sessions"])
        self.assertEqual(result["session_summaries"], [])
        self.assertTrue(any("No session identifiers" in n for n in result["interval_notes"]))
        with self.assertRaisesRegex(ValueError, "complete session_id column"):
            evaluate_csv(text, uncertainty="session_cluster")

    def test_optional_metadata_must_be_complete_and_consistent_by_band(self):
        text = session_csv([("A", "R1", 1, "day1", "cap", .6)])
        table = list(csv.reader(io.StringIO(text)))
        for field in ("session_id", "spot_id"):
            index = table[0].index(field)
            for replacement in ("", "other-id"):
                with self.subTest(field=field, replacement=replacement):
                    changed = [row.copy() for row in table]
                    changed[1][index] = replacement
                    out = io.StringIO()
                    csv.writer(out).writerows(changed)
                    with self.assertRaisesRegex(ValueError, "complete|consistent"):
                        evaluate_csv(out.getvalue())

    def test_spot_only_column_and_session_only_column_work(self):
        observations = [("A", "R1", 1, "day1", "cap", .6),
                        ("A", "R1", 1, "day2", "stipe", .8)]
        for fields in (("spot_id",), ("session_id",)):
            with self.subTest(fields=fields):
                result = evaluate_csv(session_csv(observations, fields=fields))
                self.assertEqual(result["regimes"][0]["n_observations"], 2)
                self.assertEqual(result["regimes"][0]["n_specimens"], 1)
                self.assertAlmostEqual(result["regimes"][0]["median_mae"], .2)

    def test_paired_delta_uses_matched_specimens_not_regime_medians(self):
        text = session_csv([
            ("A", "R1", 1, "day1", "cap", .5),
            ("A", "R3", 1, "day1", "cap", .7),
            ("B", "R1", 1, "day2", "cap", .9),
            ("C", "R3", 1, "day3", "cap", .5),
        ])
        report = evaluate_csv(text, uncertainty="session_cluster")
        paired = report["paired_r1_r3"]
        self.assertEqual(paired["n_specimens"], 1)
        self.assertEqual(paired["r1_only_specimens"], ["B"])
        self.assertEqual(paired["r3_only_specimens"], ["C"])
        self.assertAlmostEqual(paired["median_delta_mae"], .2)
        # The unpaired median difference would instead be -.1.
        difference = report["regimes"][1]["median_mae"] - report["regimes"][0]["median_mae"]
        self.assertAlmostEqual(difference, -.1)

    def test_invalid_uncertainty(self):
        text = session_csv([("A", "R1", 1, "day1", "cap", .6)])
        for value in (None, 1, [], "cluster", ""):
            with self.subTest(uncertainty=value), self.assertRaises(ValueError):
                evaluate_csv(text, uncertainty=value)

    def test_auto_falls_back_to_specimens_when_sessions_are_crossed(self):
        text = session_csv([
            ("A", "R1", 1, "day1", "cap", .5), ("A", "R1", 1, "day2", "cap", .6),
            ("B", "R1", 1, "day1", "cap", .55), ("B", "R1", 1, "day2", "cap", .65),
        ])
        report = evaluate_csv(text)
        self.assertEqual(report["uncertainty_method"], "specimen")
        self.assertTrue(any("multiple sessions" in note for note in report["interval_notes"]))


if __name__ == "__main__":
    unittest.main()
