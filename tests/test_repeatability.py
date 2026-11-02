import csv
import io
import unittest

import numpy as np

from myco_fusion.examples import repeatability_csv
from myco_fusion.repeatability import analyze_channel, analyze_repeatability, independent_measurements, variance_components


def balanced(rng, objects=8, sessions=4, repeats=3, sd=(0.3, 0.05, 0.03, 0.02)):
    obj = rng.normal(0, sd[0], objects)
    ses = rng.normal(0, sd[1], sessions)
    cross = rng.normal(0, sd[2], (objects, sessions))
    y, labels_o, labels_s = [], [], []
    for i in range(objects):
        for j in range(sessions):
            for _ in range(repeats):
                y.append(1 + obj[i] + ses[j] + cross[i, j] + rng.normal(0, sd[3]))
                labels_o.append(f"o{i}"); labels_s.append(f"s{j}")
    return np.array(y), labels_o, labels_s


def anova_estimates(y, objects, sessions, repeats):
    """Textbook mean-square estimators for a balanced crossed design."""
    table = y.reshape(objects, sessions, repeats)
    grand = table.mean()
    cell = table.mean(axis=2)
    row = table.mean(axis=(1, 2))
    column = table.mean(axis=(0, 2))
    ms_object = sessions * repeats * np.sum((row - grand) ** 2) / (objects - 1)
    ms_session = objects * repeats * np.sum((column - grand) ** 2) / (sessions - 1)
    ms_cross = repeats * np.sum((cell - row[:, None] - column[None, :] + grand) ** 2) / ((objects - 1) * (sessions - 1))
    ms_error = np.sum((table - cell[:, :, None]) ** 2) / (objects * sessions * (repeats - 1))
    return [(ms_object - ms_cross) / (sessions * repeats), (ms_session - ms_cross) / (objects * repeats), (ms_cross - ms_error) / repeats, ms_error]


class VarianceTests(unittest.TestCase):
    def test_reml_equals_anova_on_balanced_data(self):
        y, o, s = balanced(np.random.default_rng(5))
        theta, _, df, estimable = variance_components(y, o, s)
        self.assertEqual(estimable, [True, True, True, True])
        np.testing.assert_allclose(theta, anova_estimates(y, 8, 4, 3), rtol=1e-5, atol=1e-8)
        self.assertEqual(df, 8 * 4 * 2)

    def test_recovers_truth_on_average(self):
        rng = np.random.default_rng(6)
        estimates = []
        for _ in range(60):
            y, o, s = balanced(rng, objects=10, sessions=8)
            estimates.append(variance_components(y, o, s)[0])
        mean = np.mean(estimates, axis=0)
        np.testing.assert_allclose(mean[1:], [0.05 ** 2, 0.03 ** 2, 0.02 ** 2], rtol=0.25)

    def test_missing_photos_do_not_break_the_fit(self):
        y, o, s = balanced(np.random.default_rng(7))
        keep = np.random.default_rng(8).random(len(y)) > 0.15
        theta, _, _, _ = variance_components(y[keep], [o[i] for i in np.flatnonzero(keep)], [s[i] for i in np.flatnonzero(keep)])
        self.assertTrue(np.all(theta >= 0))
        self.assertAlmostEqual(float(np.sqrt(theta[3])), 0.02, delta=0.005)

    def test_no_session_variance_is_held_at_zero(self):
        rng = np.random.default_rng(9)
        y, o, s = balanced(rng, sd=(0.3, 0.0, 0.0, 0.02))
        self.assertLess(np.sqrt(variance_components(y, o, s)[0][1]), 0.01)

    def test_boundary_solution_is_a_true_optimum(self):
        # At a zero component the likelihood must not want it back, and free ones must be at their peak.
        rng = np.random.default_rng(12)
        y, o, s = balanced(rng, objects=6, sessions=3, repeats=2, sd=(0.3, 0.0, 0.0, 0.05))
        theta, _, _, _ = variance_components(y, o, s)
        n = len(y)
        cells = [f"{a}|{b}" for a, b in zip(o, s)]
        kernels = []
        for labels in (o, s, cells):
            names = sorted(set(labels))
            z = np.zeros((n, len(names)))
            z[np.arange(n), [names.index(v) for v in labels]] = 1
            kernels.append(z @ z.T)
        kernels.append(np.eye(n))
        x = np.ones((n, 1))

        def reml(values):
            v = sum(t * k for t, k in zip(values, kernels))
            vi = np.linalg.inv(v)
            info = x.T @ vi @ x
            p = vi - vi @ x @ np.linalg.inv(info) @ x.T @ vi
            return -0.5 * (np.linalg.slogdet(v)[1] + np.linalg.slogdet(info)[1] + y @ p @ y)

        best = reml(theta + np.array([0, 0, 0, 1e-12]))
        probe = np.random.default_rng(13)
        for _ in range(300):
            trial = np.maximum(theta * np.exp(probe.normal(0, 0.15, 4)) + probe.uniform(0, 2e-5, 4) * (theta == 0), 0)
            self.assertLessEqual(reml(trial + np.array([0, 0, 0, 1e-12])), best + 1e-6)

    def test_incomplete_designs_are_flagged_not_fatal(self):
        rng = np.random.default_rng(14)
        y, o, s = balanced(rng, objects=20, sessions=1, repeats=3)
        result = analyze_channel(y, o, s)
        self.assertIsNone(result["sd_session"])
        self.assertIsNone(result["reproducibility_coefficient"])
        self.assertAlmostEqual(result["sd_residual"], 0.02, delta=0.006)
        y, o, s = balanced(rng, objects=8, sessions=4, repeats=1)
        result = analyze_channel(y, o, s)
        self.assertIsNone(result["sd_object_by_session"])
        self.assertTrue(result["residual_includes_object_by_session"])
        with self.assertRaisesRegex(ValueError, "cannot separate"):
            analyze_channel(*balanced(rng, objects=5, sessions=1, repeats=1))

    def test_tiny_and_constant_channels_stay_well_behaved(self):
        import warnings
        rng = np.random.default_rng(16)
        y, o, s = balanced(rng, objects=10, sessions=3, repeats=3, sd=(0.002, 0.0001, 0.0001, 0.0001))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = analyze_channel(y * 0.01, o, s)
            self.assertGreater(result["sd_residual"], 0)
            constant = analyze_channel(np.full(len(y), 0.5), o, s)
            self.assertEqual(constant["sd_residual"], 0)

    def test_channel_summary_uses_the_right_formulas(self):
        y, o, s = balanced(np.random.default_rng(10))
        result = analyze_channel(y, o, s)
        self.assertAlmostEqual(result["repeatability_coefficient"], 1.96 * np.sqrt(2) * result["sd_residual"], places=2)
        between = result["sd_session"] ** 2 + result["sd_object_by_session"] ** 2 + result["sd_residual"] ** 2
        self.assertAlmostEqual(result["reproducibility_coefficient"], 1.96 * np.sqrt(2 * between), places=2)
        self.assertGreater(result["icc_agreement"], 0.9)
        self.assertLess(result["sd_residual_interval"]["low"], 0.02)
        self.assertGreater(result["sd_residual_interval"]["high"], 0.02)


def simulated_rows(rng, responses_by_light, repeats=3, sessions=3, additive=0.002, gain=0.01):
    rows = []
    for session in range(sessions):
        for light, table in responses_by_light.items():
            for i, clean in enumerate(table):
                for repeat in range(repeats):
                    values = clean * (1 + rng.normal(0, gain)) + rng.normal(0, additive, 3)
                    rows.append({"object": f"o{i}", "session": f"s{session}", "light": light, "repeat": str(repeat), "values": list(values)})
    return rows


class IndependentMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)
        self.latent = self.rng.uniform(0.1, 0.9, (25, 3))

    def test_brightness_copies_add_nothing(self):
        rows = simulated_rows(self.rng, {"half": 0.5 * self.latent, "full": self.latent, "bright": 1.5 * self.latent})
        result = independent_measurements(rows, ["half", "full", "bright"])
        self.assertEqual(result["channels"], 9)
        self.assertEqual(result["independent"], 3)

    def test_truly_new_information_is_counted(self):
        extra = self.rng.uniform(0.1, 0.9, (25, 3))
        rows = simulated_rows(self.rng, {"first": self.latent, "second": extra})
        self.assertEqual(independent_measurements(rows, ["first", "second"])["independent"], 6)

    def test_errors_shared_within_a_session_are_not_counted_as_directions(self):
        rng = np.random.default_rng(15)
        false_alarms = 0
        for _ in range(12):
            rows = []
            for session in range(3):
                for light, scale in (("a", 1.0), ("b", 0.7), ("c", 1.3)):
                    for i, clean in enumerate(self.latent):
                        drift = 1 + rng.normal(0, 0.03)
                        for repeat in range(3):
                            rows.append({"object": f"o{i}", "session": f"s{session}", "light": light, "repeat": str(repeat),
                                         "values": list(scale * clean * drift + rng.normal(0, 0.002, 3))})
            false_alarms += independent_measurements(rows, ["a", "b", "c"], draws=120)["independent"] > 3
        self.assertLessEqual(false_alarms, 2)

    def test_cannot_exceed_objects_minus_one(self):
        latent = self.rng.uniform(0.1, 0.9, (4, 3))
        rows = simulated_rows(self.rng, {"a": latent, "b": self.rng.uniform(0.1, 0.9, (4, 3))})
        self.assertLessEqual(independent_measurements(rows, ["a", "b"])["independent"], 3)


class ReportTests(unittest.TestCase):
    def test_example_report(self):
        report = analyze_repeatability(repeatability_csv())
        self.assertEqual(report["data_kind"], "synthetic")
        self.assertEqual(report["counts"], {"n_photos": 360, "n_objects": 10, "n_sessions": 3, "n_lights": 4,
                                            "n_missing_combinations": 0, "photos_per_combination_median": 3.0, "crossed": True})
        flash = next(row for row in report["lights"] if row["light"] == "flash")
        self.assertAlmostEqual(flash["channels"]["g"]["sd_residual"], 0.003, delta=0.001)
        by_lights = {tuple(row["lights"]): row["independent"] for row in report["independent_measurements"]}
        self.assertEqual(by_lights[("flash",)], 3)
        self.assertTrue(any("degrees of freedom" in note for note in report["notes"]))

    def test_bad_inputs_are_rejected_with_reasons(self):
        good = "object_id,session_id,light,repeat,r,g,b\na,s1,flash,1,.1,.2,.3\n"
        cases = {"": "empty", "a,b\n1,2\n": "columns", good + "a,s1,flash,1,.1,.2,.3\n": "Duplicate",
                 good + "b,s1,flash,1,x,.2,.3\n": "Non-numeric", good + "b,s1,flash,1,inf,.2,.3\n": "finite",
                 good + "b,,flash,1,.1,.2,.3\n": "nonempty"}
        for text, message in cases.items():
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                analyze_repeatability(text)

    def test_tabs_spaces_and_mixed_case_are_accepted(self):
        rows = list(csv.reader(io.StringIO(repeatability_csv())))
        out = io.StringIO()
        writer = csv.writer(out, delimiter="\t")
        writer.writerow([name.upper() + " " if name == "light" else " " + name for name in rows[0]])
        writer.writerows(rows[1:])
        self.assertEqual(analyze_repeatability(out.getvalue())["counts"]["n_photos"], 360)
        spaced = "object_id, session_id, light, repeat, r, g, b\n" + "".join(f"o{i}, s1, Flash, {r}, .{i + 1}, .2, .3\n" for i in range(3) for r in (1, 2))
        report = analyze_repeatability(spaced)
        self.assertEqual([row["light"] for row in report["lights"]], ["flash"])
        self.assertTrue(any("one session" in note for note in report["notes"]))

    def test_one_session_report_explains_what_is_missing(self):
        text = "object_id,session_id,light,repeat,r,g,b\n" + "".join(f"o{i},s1,flash,{r},.{i + 1}{r},.2,.3\n" for i in range(5) for r in (1, 2, 3))
        report = analyze_repeatability(text)
        self.assertFalse(report["counts"]["crossed"])
        channel = report["lights"][0]["channels"]["g"]
        self.assertIsNone(channel["reproducibility_coefficient"])

    def test_negative_linear_values_are_allowed(self):
        text = "object_id,session_id,light,repeat,r,g,b\n" + "".join(f"o{i},s1,flash,{r},-0.001,.2,.3\n" for i in range(3) for r in (1, 2))
        self.assertEqual(analyze_repeatability(text)["counts"]["n_photos"], 6)

    def test_user_data_is_not_labeled_synthetic(self):
        rows = list(csv.reader(io.StringIO(repeatability_csv())))
        header = rows[0][:-1]
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(row[:-1] for row in rows[1:])
        self.assertEqual(analyze_repeatability(out.getvalue())["data_kind"], "user_supplied_unverified")


if __name__ == "__main__":
    unittest.main()
