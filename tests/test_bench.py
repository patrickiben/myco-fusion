import json
import unittest

import numpy as np

from myco_fusion.bench import group_split, run_bench
from myco_fusion.data import natural_objects


class BenchTests(unittest.TestCase):
    def test_bundled_spectra_are_measured_and_bounded(self):
        data = natural_objects()
        self.assertEqual(data["spectra"].shape, (532, 27))
        self.assertTrue(np.all((data["spectra"] >= 0) & (data["spectra"] <= 1)))
        self.assertEqual(len(set(data["domain"])), 10)

    def test_species_never_split_across_folds(self):
        data = natural_objects()
        folds = group_split(data["group"], 5, 7)
        for group in set(data["group"]):
            self.assertEqual(len({int(folds[i]) for i, g in enumerate(data["group"]) if g == group}), 1)

    def test_brightness_adds_channels_but_no_directions(self):
        report = run_bench()
        scenarios = {row["id"]: row for row in report["scenarios"]}
        self.assertEqual((scenarios["brightness"]["channels"], scenarios["brightness"]["independent"]), (9, 3))
        self.assertEqual(scenarios["screen"]["channels"], 12)
        self.assertGreater(scenarios["screen"]["independent"], 3)
        self.assertGreater(scenarios["changed_light"]["median_mae"], scenarios["flash"]["median_mae"])

    def test_measured_errors_exceed_the_idealized_best_case(self):
        measured = {row["id"]: row for row in run_bench(noise=0.003)["scenarios"]}
        idealized = {row["id"]: row for row in run_bench(noise=0.003, source="idealized")["scenarios"]}
        self.assertGreater(measured["flash"]["median_sam_deg"], idealized["matched"]["median_sam_deg"])

    def test_report_is_deterministic_and_strict_json(self):
        first = run_bench(seed=13)
        self.assertEqual(first, run_bench(seed=13))
        json.dumps(first, allow_nan=False)

    def test_bad_arguments_are_rejected(self):
        for kwargs in ({"seed": -1}, {"seed": True}, {"noise": -0.1}, {"noise": float("nan")}, {"source": "other"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_bench(**kwargs)


if __name__ == "__main__":
    unittest.main()
