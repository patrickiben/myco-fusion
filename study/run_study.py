"""Run questions Q1 to Q5 of study/PLAN.md and write results/study.json."""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from myco_fusion.intervals import chi2_quantile

from .pipeline import (ALPHAS, METRIC_NM, NOISE, SIM_NM, Evaluation, camera_curves, capture, capture_arm, cv_predict,
                       errors, fit_selected, gaussian_light, group_folds, light_curve, match_exposure, operator_rank,
                       paired_reduction, resample, share_bootstrap)
from .sources import load_colour_tables, load_mushrooms, load_natural_objects, load_phone_curves

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261101
DRAWS = 20


def rng_for(*names):
    return np.random.default_rng([SEED, *[zlib.crc32(str(name).encode()) for name in names]])


def prepare(wavelengths, spectra):
    metric = resample(wavelengths, spectra, METRIC_NM)
    clipped = int(np.sum(metric > 1) + np.sum(metric < 0))
    bounded = np.clip(spectra, 0, 1)
    return resample(wavelengths, bounded, SIM_NM), resample(wavelengths, bounded, METRIC_NM), clipped


class Dataset:
    def __init__(self, name, wavelengths, spectra, groups):
        self.name = name
        self.sim, self.metric, self.clipped = prepare(wavelengths, spectra)
        self.groups = np.asarray(groups)
        self.folds = group_folds(self.groups, 5, rng_for("folds", name))

    def subset(self, name, mask):
        other = object.__new__(Dataset)
        other.name, other.sim, other.metric, other.clipped = name, self.sim[mask], self.metric[mask], 0
        other.groups = self.groups[mask]
        other.folds = group_folds(other.groups, 5, rng_for("folds", name))
        return other


def run_cv(dataset, make_x, label, kind="ridge", draws=DRAWS):
    """Cross-validate one arm over noise draws."""
    sams, maes = [], []
    for draw in range(draws):
        rng = rng_for(dataset.name, label, kind, draw)
        estimate = cv_predict(make_x(dataset.sim, rng), dataset.metric, dataset.folds, dataset.groups, rng, kind)
        sam, mae = errors(dataset.metric, estimate)
        sams.append(sam); maes.append(mae)
    return Evaluation(sams, maes)


def run_chart(chart, dataset, make_x, label, draws=DRAWS):
    """Fit on the chart only and test on the whole dataset, over noise draws."""
    sams, maes = [], []
    for draw in range(draws):
        rng = rng_for(dataset.name, label, "chart", draw)
        model = fit_selected(make_x(chart.sim, rng), chart.metric, chart.groups, rng)
        sam, mae = errors(dataset.metric, model.predict(make_x(dataset.sim, rng)))
        sams.append(sam); maes.append(mae)
    return Evaluation(sams, maes)


def pca_floor(dataset, k_values=range(1, 9)):
    """Median MAE if the best linear k-number summary of each held-out spectrum were known exactly."""
    out = {}
    for k in k_values:
        errors_k = np.empty(len(dataset.metric))
        for fold in np.unique(dataset.folds):
            train, test = dataset.folds != fold, dataset.folds == fold
            mean = dataset.metric[train].mean(axis=0)
            _, _, vt = np.linalg.svd(dataset.metric[train] - mean, full_matrices=False)
            basis = vt[:k]
            residual = (dataset.metric[test] - mean) - (dataset.metric[test] - mean) @ basis.T @ basis
            errors_k[test] = np.mean(np.abs(residual), axis=1)
        out[k] = float(np.median(errors_k))
    return out


def constant_baseline(dataset):
    estimate = np.empty_like(dataset.metric)
    for fold in np.unique(dataset.folds):
        train, test = dataset.folds != fold, dataset.folds == fold
        estimate[test] = np.median(dataset.metric[train], axis=0)
    sam, mae = errors(dataset.metric, estimate)
    return Evaluation([sam], [mae]).summary()


def with_share_interval(evaluation, groups, *label):
    row = evaluation.summary()
    row["share_interval"] = share_bootstrap(evaluation.within, groups, rng_for("share", *label))
    return row


def question_1(natural, mushrooms, mixed, camera, flash, noise, sigma_camera, repeats_27):
    out, kept = {}, {}
    for dataset in (natural, mushrooms, mixed):
        def one_photo(spectra, rng):
            return capture(spectra, flash, camera, noise, rng)
        affine = run_cv(dataset, one_photo, "one-photo")
        polynomial = run_cv(dataset, one_photo, "one-photo", "polynomial")
        kernel = run_cv(dataset, one_photo, "one-photo", "kernel")
        best_name, best = min((("polynomial", polynomial), ("kernel", kernel)), key=lambda item: np.median(item[1].mae))
        gain = paired_reduction(affine.mae, best.mae, dataset.groups, rng_for("boot-q1", dataset.name), statistic="ratio")
        kept[dataset.name] = {"affine": affine, "kernel": kernel}
        out[dataset.name] = {
            "n": len(dataset.metric),
            "affine": with_share_interval(affine, dataset.groups, dataset.name, "affine"),
            "polynomial": polynomial.summary(), "kernel": kernel.summary(),
            "best_nonlinear": best_name, "best_nonlinear_summary": best.summary(), "nonlinear_gain": gain,
            "limit_holds": bool(gain["percent"] <= 10), "limit_resolved": bool(not gain["low"] <= 10 <= gain["high"]),
            "constant_baseline": constant_baseline(dataset), "pca_floor": pca_floor(dataset),
            "sigma_camera": run_cv(dataset, lambda s, r: capture(s, flash, sigma_camera, noise, r), "one-photo-sigma").summary(),
        }
    # Repeat-measurement error: each repeat against the mean of the others, per sample.
    sam_list, mae_list = [], []
    for repeats in repeats_27:
        if len(repeats) < 2:
            continue
        sams, maes = [], []
        for i in range(len(repeats)):
            others = np.delete(repeats, i, axis=0).mean(axis=0)
            sam, mae = errors(np.clip(others, 0, 1)[None, :], np.clip(repeats[i], 0, 1)[None, :])
            sams.append(sam[0]); maes.append(mae[0])
        sam_list.append(np.mean(sams)); mae_list.append(np.mean(maes))
    out["repeat_measurement"] = {"n_samples": len(mae_list), "sam_median": float(np.median(sam_list)), "mae_median": float(np.median(mae_list))}
    out["natural_by_domain"] = {domain: kept["natural"]["affine"].summary(natural.domain == domain) for domain in sorted(set(natural.domain.tolist()))}
    return out, kept


def question_2(natural, mushrooms, chart, camera, flash, noise):
    def one_photo(spectra, rng):
        return capture(spectra, flash, camera, noise, rng)
    out = {"domains": {}}
    for domain in sorted(set(natural.domain.tolist())):
        subset = natural.subset(f"natural:{domain}", natural.domain == domain)
        chart_cal, same = run_chart(chart, subset, one_photo, "transfer").summary(), run_cv(subset, one_photo, "same-domain").summary()
        ratio = chart_cal["mae_median"] / same["mae_median"]
        out["domains"][domain] = {"n": len(subset.metric), "chart": chart_cal, "same_domain": same, "ratio": ratio, "transfer_adequate": bool(ratio <= 1.2)}
    chart_all = run_chart(chart, natural, one_photo, "transfer")
    pooled = run_cv(natural, one_photo, "one-photo").summary()
    ratio = chart_all.summary()["mae_median"] / pooled["mae_median"]
    out["natural_all"] = {"chart": with_share_interval(chart_all, natural.groups, "natural", "chart"), "pooled_cv": pooled, "ratio": ratio, "transfer_adequate": bool(ratio <= 1.2)}
    chart_m = run_chart(chart, mushrooms, one_photo, "transfer").summary()
    within = run_cv(mushrooms, one_photo, "one-photo").summary()
    sams, maes = [], []
    for draw in range(DRAWS):
        rng = rng_for("natural-to-mushroom", draw)
        model = fit_selected(one_photo(natural.sim, rng), natural.metric, natural.groups, rng)
        sam, mae = errors(mushrooms.metric, model.predict(one_photo(mushrooms.sim, rng)))
        sams.append(sam); maes.append(mae)
    ratio = chart_m["mae_median"] / within["mae_median"]
    out["mushrooms"] = {"n": len(mushrooms.metric), "chart": chart_m, "natural_objects": Evaluation(sams, maes).summary(), "within_set": within,
                        "constant_within_set": constant_baseline(mushrooms), "ratio": ratio, "transfer_adequate": bool(ratio <= 1.2)}
    return out


def question_3(natural, mushrooms, chart, camera, flash, noise, lights):
    white = chart.names.index("white 9.5 (.05 D)") if "white 9.5 (.05 D)" in chart.names else 18
    out = {}
    for dataset in (natural, mushrooms):
        same_light = run_chart(chart, dataset, lambda s, r: capture(s, flash, camera, noise, r), "transfer").summary()
        rows = {}
        for name, light in lights.items():
            sams = {key: [] for key in ("none", "white_patch", "chart_under_test_light")}
            maes = {key: [] for key in sams}
            for draw in range(DRAWS):
                rng = rng_for(dataset.name, "shift", name, draw)
                chart_flash = capture(chart.sim, flash, camera, noise, rng)
                chart_test = capture(chart.sim, light, camera, noise, rng)
                test = capture(dataset.sim, light, camera, noise, rng)
                base = fit_selected(chart_flash, chart.metric, chart.groups, rng)
                retrained = fit_selected(chart_test, chart.metric, chart.groups, rng)
                gains = chart_flash[white] / chart_test[white]
                estimates = {"none": base.predict(test), "white_patch": base.predict(test * gains), "chart_under_test_light": retrained.predict(test)}
                for key, estimate in estimates.items():
                    sam, mae = errors(dataset.metric, estimate)
                    sams[key].append(sam); maes[key].append(mae)
            evaluations = {key: Evaluation(sams[key], maes[key]) for key in sams}
            rows[name] = {key: evaluation.summary() for key, evaluation in evaluations.items()}
            for key in ("white_patch", "chart_under_test_light"):
                reduction = paired_reduction(evaluations["none"].mae, evaluations[key].mae, dataset.groups, rng_for("boot", dataset.name, name, key))
                reduction["helps"] = bool(reduction["low"] > 0 and reduction["percent"] >= 20)
                rows[name][f"{key}_reduction"] = reduction
        out[dataset.name] = {"same_light": same_light, "lights": rows}
    return out


def arms(flash, ambient, screen, factor):
    primaries = [(curve, factor) for curve in screen]
    return {
        "flash": [(flash, 1.0)],
        "flash_three_brightness": [(0.5 * flash, 1.0), (flash, 1.0), (1.5 * flash, 1.0)],
        "flash_plus_ambient": [(flash, 1.0), (ambient, 1.0)],
        "screen_rgb": primaries,
        "flash_plus_screen_rgb": [(flash, 1.0), *primaries],
    }


def question_4(natural, mushrooms, chart, cameras, flashes, ambient, screens, noises, q1_kept):
    camera, flash, noise = cameras["Nikon D5100"], flashes["LED-B4"], noises["moderate"]
    out = {"primary": {}, "chart_calibrated": {}, "sensitivity": {}}

    def exposed(primaries, camera_c=camera, flash_c=flash):
        return [match_exposure(curve, flash_c, camera_c) for curve in primaries]

    table = arms(flash, match_exposure(ambient, flash, camera), exposed(screens["medium"]), 3.0)
    kept = {}
    for dataset in (natural, mushrooms):
        rows, evaluations = {}, {}
        for name, lights in table.items():
            label = "one-photo" if name == "flash" else f"arm-{name}"
            evaluations[name] = run_cv(dataset, lambda s, r, ls=lights: capture_arm(s, ls, camera, noise, r), label)
            rows[name] = evaluations[name].summary()
            rows[name]["channels"] = 3 * len(lights)
            rows[name]["independent_directions"] = operator_rank(lights, camera)
        for name in table:
            if name != "flash":
                reduction = paired_reduction(evaluations["flash"].mae, evaluations[name].mae, dataset.groups, rng_for("boot", dataset.name, name))
                reduction["helps"] = bool(reduction["low"] > 0 and reduction["percent"] >= 20)
                rows[name]["reduction_vs_flash"] = reduction
        out["primary"][dataset.name] = rows
        kept[dataset.name] = evaluations
        rows, evaluations = {}, {}
        for name, lights in table.items():
            label = "transfer" if name == "flash" else f"arm-{name}"
            evaluations[name] = run_chart(chart, dataset, lambda s, r, ls=lights: capture_arm(s, ls, camera, noise, r), label)
            rows[name] = evaluations[name].summary()
        for name in table:
            if name != "flash":
                reduction = paired_reduction(evaluations["flash"].mae, evaluations[name].mae, dataset.groups, rng_for("boot-chart", dataset.name, name))
                reduction["helps"] = bool(reduction["low"] > 0 and reduction["percent"] >= 20)
                rows[name]["reduction_vs_flash"] = reduction
        out["chart_calibrated"][dataset.name] = rows

    # Exploratory: a nonlinear fit on one photo against a linear fit on four.
    out["nonlinear_one_photo_vs_linear_four_photos"] = paired_reduction(
        q1_kept["natural"]["kernel"].mae, kept["natural"]["flash_plus_screen_rgb"].mae, natural.groups, rng_for("boot-nonlinear-vs-arm"))

    def contrast(label, camera_c, flash_c, noise_c, screen_c, factor_c):
        lights_flash = [(flash_c, 1.0)]
        lights_all = [(flash_c, 1.0), *[(curve, factor_c) for curve in screen_c]]
        base = run_cv(natural, lambda s, r: capture_arm(s, lights_flash, camera_c, noise_c, r), f"sens-flash-{label}")
        arm = run_cv(natural, lambda s, r: capture_arm(s, lights_all, camera_c, noise_c, r), f"sens-all-{label}")
        reduction = paired_reduction(base.mae, arm.mae, natural.groups, rng_for("boot-sens", label))
        reduction["helps"] = bool(reduction["low"] > 0 and reduction["percent"] >= 20)
        base_s, arm_s = base.summary(), arm.summary()
        return {"flash_mae_median": base_s["mae_median"], "arm_mae_median": arm_s["mae_median"], "flash_mae_p90": base_s["mae_p90"], "arm_mae_p90": arm_s["mae_p90"],
                "flash_share": base_s["share_within_targets"], "arm_share": arm_s["share_within_targets"],
                "independent_directions": operator_rank(lights_all, camera_c), "reduction": reduction}
    sensitivity = out["sensitivity"]
    for name in ("narrow", "medium", "wide", "Apple Studio Display (measured)"):
        sensitivity[f"screen: {name}"] = contrast(f"screen-{name}", camera, flash, noise, exposed(screens[name]), 3.0)
    for factor in (1.0, 5.0):
        sensitivity[f"screen noise x{factor:g}"] = contrast(f"factor-{factor}", camera, flash, noise, exposed(screens["medium"]), factor)
    for level in ("low", "high"):
        sensitivity[f"noise: {level}"] = contrast(f"noise-{level}", camera, flash, noises[level], exposed(screens["medium"]), 3.0)
    for name in ("LED-B1", "LED-B5"):
        sensitivity[f"flash: {name}"] = contrast(f"flash-{name}", camera, flashes[name], noise, exposed(screens["medium"], camera, flashes[name]), 3.0)
    sigma = cameras["Sigma SD Merrill"]
    sensitivity["camera: Sigma SD Merrill"] = contrast("camera-sigma", sigma, flash, noise, exposed(screens["medium"], sigma, flash), 3.0)
    return out


def question_5(natural, mushrooms, camera, flash, screen, noise, factor=3.0):
    def responses(spectra, light):
        return (spectra * light) @ camera.T

    def threshold(a, b, multiplier=1.0):
        level = (a + b) / 2
        return 2 * np.sqrt(2) * np.sqrt((noise.additive * multiplier) ** 2 + (noise.gain * level) ** 2)

    def within(a, b, multiplier=1.0):
        return np.all(np.abs(a - b) <= threshold(a, b, multiplier), axis=-1)

    y = responses(natural.sim, flash)
    exposed = [match_exposure(curve, flash, camera) for curve in screen]
    extra = [responses(natural.sim, curve) for curve in exposed]
    i, j = np.triu_indices(len(y), 1)
    independent = natural.groups[i] != natural.groups[j]
    i, j = i[independent], j[independent]
    look_alike = within(y[i], y[j])
    i, j = i[look_alike], j[look_alike]
    sam, mae = errors(natural.metric[i], natural.metric[j])
    differ = (sam > 10) | (mae > 0.05)
    separable = np.zeros(len(i), dtype=bool)
    for table in extra:
        separable |= ~within(table[i], table[j], factor)
    # Stricter test: photo errors are shared across a photo's channels, so compare all three at once.
    limit = chi2_quantile(0.95, 3)
    strict = np.zeros(len(i), dtype=bool)
    for k, (a, b) in enumerate(zip(y[i], y[j])):
        covariance = 2 * noise.additive ** 2 * np.eye(3) + noise.gain ** 2 * (np.outer(a, a) + np.outer(b, b))
        gap = a - b
        strict[k] = gap @ np.linalg.solve(covariance, gap) <= limit
    mean_a, mean_b = natural.metric[i].mean(axis=1), natural.metric[j].mean(axis=1)
    bright = (mean_a > 0.15) & (mean_b > 0.15)
    dark = (mean_a < 0.15) | (mean_b < 0.15)
    # Illustration only: the largest spectral-shape difference that passes the stricter test,
    # among objects not so dark that their shape is mostly noise.
    candidates = np.flatnonzero(differ & strict & (mean_a > 0.08) & (mean_b > 0.08))
    top = candidates[np.argmax(sam[candidates])] if len(candidates) else None

    def describe(index):
        return {"id": natural.ids[index], "domain": str(natural.domain[index]), "object": natural.objects[index], "color": natural.colors[index]}

    out = {"pairs_from_independent_groups": int(independent.sum()),
           "look_alike_pairs": int(look_alike.sum()),
           "look_alike_above_a_draft_target": int(differ.sum()),
           "above_target_by_angle_only": int(((sam > 10) & (mae <= 0.05)).sum()),
           "above_target_by_mae": int((mae > 0.05).sum()),
           "median_mae_of_those": float(np.median(mae[differ])) if differ.any() else None,
           "of_those_both_bright": int((differ & bright).sum()),
           "of_those_with_a_dark_patch": int((differ & dark).sum()),
           "of_those_passing_strict_test": int((differ & strict).sum()),
           "of_those_separated_by_screen_light": int((differ & separable).sum()),
           "share_separated": float((differ & separable).sum() / max(1, differ.sum())),
           "note": "Noise-free responses compared against a per-channel noise threshold at moderate noise. The strict test also accounts for gain errors shared across a photo's channels."}
    if top is not None:
        a, b = int(i[top]), int(j[top])
        out["example_pair"] = {"a": describe(a), "b": describe(b), "sam_deg": float(sam[top]), "mae": float(mae[top]),
                               "flash_response_a": y[a].tolist(), "flash_response_b": y[b].tolist(),
                               "bands_nm": METRIC_NM.astype(int).tolist(),
                               "spectrum_a": natural.metric[a].round(5).tolist(), "spectrum_b": natural.metric[b].round(5).tolist()}
    ym = responses(mushrooms.sim, flash)
    pairs_a, pairs_b = np.meshgrid(np.arange(len(ym)), np.arange(len(y)), indexing="ij")
    pairs_a, pairs_b = pairs_a.ravel(), pairs_b.ravel()
    alike = within(ym[pairs_a], y[pairs_b])
    sam_m, mae_m = errors(mushrooms.metric[pairs_a[alike]], natural.metric[pairs_b[alike]])
    out["mushroom_vs_natural"] = {"look_alike_pairs": int(alike.sum()), "look_alike_above_a_draft_target": int(((sam_m > 10) | (mae_m > 0.05)).sum())}
    return out


def question_7(natural, chart, phones, flash, noise, shift_curves, screen):
    """Repeat the natural-object tests for each phone (see the plan's addendum)."""
    out = {}
    for phone, camera in phones.items():
        def one_photo(spectra, rng, camera=camera):
            return capture(spectra, flash, camera, noise, rng)
        affine = run_cv(natural, one_photo, f"{phone}-one-photo")
        kernel = run_cv(natural, one_photo, f"{phone}-one-photo", "kernel")
        polynomial = run_cv(natural, one_photo, f"{phone}-one-photo", "polynomial")
        best_name, best = min((("polynomial", polynomial), ("kernel", kernel)), key=lambda item: np.median(item[1].mae))
        gain = paired_reduction(affine.mae, best.mae, natural.groups, rng_for("boot-q7", phone), statistic="ratio")
        chart_eval = run_chart(chart, natural, one_photo, f"{phone}-transfer")
        entry = {
            "affine": with_share_interval(affine, natural.groups, phone, "affine"),
            "best_nonlinear": best_name, "best_nonlinear_summary": best.summary(), "nonlinear_gain": gain,
            "limit_holds": bool(gain["percent"] <= 10), "limit_resolved": bool(not gain["low"] <= 10 <= gain["high"]),
            "chart": with_share_interval(chart_eval, natural.groups, phone, "chart"),
            "chart_transfer_ratio": chart_eval.summary()["mae_median"] / affine.summary()["mae_median"],
        }
        white = chart.names.index("white 9.5 (.05 D)") if "white 9.5 (.05 D)" in chart.names else 18
        lights = {}
        for name, curve in shift_curves.items():
            light = match_exposure(curve, flash, camera)
            sams = {key: [] for key in ("none", "white_patch", "chart_under_test_light")}
            maes = {key: [] for key in sams}
            for draw in range(DRAWS):
                rng = rng_for(phone, "shift", name, draw)
                chart_flash = capture(chart.sim, flash, camera, noise, rng)
                chart_test = capture(chart.sim, light, camera, noise, rng)
                test = capture(natural.sim, light, camera, noise, rng)
                base = fit_selected(chart_flash, chart.metric, chart.groups, rng)
                retrained = fit_selected(chart_test, chart.metric, chart.groups, rng)
                gains = chart_flash[white] / chart_test[white]
                estimates = {"none": base.predict(test), "white_patch": base.predict(test * gains), "chart_under_test_light": retrained.predict(test)}
                for key, estimate in estimates.items():
                    sam, mae = errors(natural.metric, estimate)
                    sams[key].append(sam); maes[key].append(mae)
            evaluations = {key: Evaluation(sams[key], maes[key]) for key in sams}
            row = {key: evaluation.summary() for key, evaluation in evaluations.items()}
            for key in ("white_patch", "chart_under_test_light"):
                reduction = paired_reduction(evaluations["none"].mae, evaluations[key].mae, natural.groups, rng_for("boot-q7", phone, name, key))
                reduction["helps"] = bool(reduction["low"] > 0 and reduction["percent"] >= 20)
                row[f"{key}_reduction"] = reduction
            lights[name] = row
        entry["lights"] = lights
        primaries = [(match_exposure(curve, flash, camera), 3.0) for curve in screen]
        all_lights = [(flash, 1.0), *primaries]
        for label, runner, base_eval in (("similar_objects", lambda name: run_cv(natural, lambda s, r: capture_arm(s, all_lights, camera, noise, r), name), affine),
                                         ("chart_only", lambda name: run_chart(chart, natural, lambda s, r: capture_arm(s, all_lights, camera, noise, r), name), chart_eval)):
            arm = runner(f"{phone}-flash-screen")
            reduction = paired_reduction(base_eval.mae, arm.mae, natural.groups, rng_for("boot-q7-arm", phone, label))
            reduction["helps"] = bool(reduction["low"] > 0 and reduction["percent"] >= 20)
            entry[f"flash_plus_screen_{label}"] = {**arm.summary(), "reduction_vs_flash": reduction, "independent_directions": operator_rank(all_lights, camera)}
        out[phone] = entry
    return out


def main():
    tables = load_colour_tables()
    raw_natural = load_natural_objects()
    raw_mushrooms = load_mushrooms()
    natural = Dataset("natural", raw_natural["wavelengths"], raw_natural["spectra"], raw_natural["group"])
    natural.domain = np.asarray(raw_natural["domain"])
    natural.ids, natural.objects, natural.colors = raw_natural["ids"], raw_natural["object"], raw_natural["color"]
    mushrooms = Dataset("mushrooms", raw_mushrooms["wavelengths"], raw_mushrooms["spectra"], raw_mushrooms["group"])
    mixed_n = len(tables["mixed"])
    mixed = Dataset("mixed", tables["mixed_wavelengths"], tables["mixed"], [f"patch{i}" for i in range(mixed_n)])
    chart = Dataset("chart", tables["chart_wavelengths"], tables["chart"], tables["chart_names"])
    chart.names = tables["chart_names"]
    cameras = {name: camera_curves(*pair) for name, pair in tables["cameras"].items()}
    flashes = {name: light_curve(*tables["illuminants"][name]) for name in ("LED-B1", "LED-B4", "LED-B5")}
    camera, flash, noise = cameras["Nikon D5100"], flashes["LED-B4"], NOISE["moderate"]
    d65 = light_curve(*tables["illuminants"]["D65"])
    leaf = np.median(natural.sim[natural.domain == "Leaves"], axis=0)
    shift_curves = {"D65 daylight": d65, "A incandescent": light_curve(*tables["illuminants"]["A"]),
                    "LED-B1 warm white": flashes["LED-B1"], "LED-B5 cool white": flashes["LED-B5"],
                    "FL2 fluorescent": light_curve(*tables["illuminants"]["FL2"]),
                    "Green shade (illustrative)": light_curve(SIM_NM, d65 * leaf)}
    shift_lights = {name: match_exposure(curve, flash, camera) for name, curve in shift_curves.items()}
    display_wl, display_values = tables["apple_studio_display"]
    gaussian = {"narrow": ((458, 22), (528, 32), (630, 20)), "medium": ((458, 30), (530, 45), (625, 30)), "wide": ((455, 40), (535, 70), (615, 60))}
    screens = {name: [gaussian_light(*p) for p in primaries] for name, primaries in gaussian.items()}
    screens["Apple Studio Display (measured)"] = [light_curve(display_wl, display_values[:, c]) for c in range(3)]
    repeats_27 = [resample(raw_natural["wavelengths"], repeats, METRIC_NM) for repeats in raw_natural["repeats"]]

    results = {
        "plan": "study/PLAN.md", "seed": SEED, "noise_draws": DRAWS, "alpha_grid": list(ALPHAS),
        "metric_grid_nm": [420, 680, 10], "simulation_grid_nm": [400, 700, 5],
        "noise_levels": {name: {"additive": n.additive, "gain": n.gain} for name, n in NOISE.items()},
        "data": {
            "natural": {"n": len(natural.metric), "groups": int(len(np.unique(natural.groups))), "largest_group": int(np.unique(natural.groups, return_counts=True)[1].max()),
                        "clipped_band_values": natural.clipped, "by_domain": {d: int(np.sum(natural.domain == d)) for d in sorted(set(natural.domain.tolist()))}},
            "mushrooms": {"n": len(mushrooms.metric), "clipped_band_values": mushrooms.clipped,
                          "dry_matter_min": float(raw_mushrooms["dry_matter"].min()), "dry_matter_max": float(raw_mushrooms["dry_matter"].max()),
                          "dry_matter_median": float(np.median(raw_mushrooms["dry_matter"]))},
            "mixed": {"n": mixed_n, "clipped_band_values": mixed.clipped}, "chart": {"n": len(chart.metric), "clipped_band_values": chart.clipped},
        },
    }
    print("Q1 ...", flush=True)
    results["q1"], kept = question_1(natural, mushrooms, mixed, camera, flash, noise, cameras["Sigma SD Merrill"], repeats_27)
    print("Q2 ...", flush=True)
    results["q2"] = question_2(natural, mushrooms, chart, camera, flash, noise)
    print("Q3 ...", flush=True)
    results["q3"] = question_3(natural, mushrooms, chart, camera, flash, noise, shift_lights)
    print("Q4 ...", flush=True)
    results["q4"] = question_4(natural, mushrooms, chart, cameras, flashes, d65, screens, NOISE, kept)
    print("Q5 ...", flush=True)
    results["q5"] = question_5(natural, mushrooms, camera, flash, screens["medium"], noise)
    print("Q7 ...", flush=True)
    phones = {name: camera_curves(*pair) for name, pair in load_phone_curves().items()}
    results["q7"] = question_7(natural, chart, phones, flash, noise, shift_curves, screens["medium"])
    out = ROOT / "results" / "study.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1, allow_nan=False) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
