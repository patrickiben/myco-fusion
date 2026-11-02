"""The spectral bench: one camera model, measured spectra, four ways of taking the photo.

Reflectances are measured. The camera and lights are illustrative curves, so the bench
is self-contained; study/run_study.py repeats this with measured camera sensitivities.
"""
from __future__ import annotations

import numpy as np

from .data import natural_objects
from .spectra import BANDS_NM, _validate_seed, mean_absolute_error, spectral_angle
from .synthetic import _camera, run_synthetic
from .validation import DRAFT_TARGETS

ALPHA = 1e-4
_W = BANDS_NM.astype(float)


def _gauss(center, sd):
    return np.exp(-0.5 * ((_W - center) / sd) ** 2)


def _normalized(curve):
    return curve / curve.mean()


def illustrative_lights():
    """A white-LED-like flash, an incandescent-like light and three screen colors.

    Each light is scaled as exposure would scale it: the strongest channel reads the
    same on a white target as it does under flash. Screen photos then pay for their
    dimness through extra noise, not through a smaller signal.
    """
    camera = _camera()
    flash = _normalized(0.55 * _gauss(452, 11) + _gauss(560, 62))

    def exposed(curve):
        return curve * (camera @ flash).max() / (camera @ curve).max()

    warm = exposed(np.exp(0.0078 * (_W - 420)))
    screen = [exposed(_gauss(c, s)) for c, s in ((458, 13), (530, 19), (625, 13))]
    return {"flash": flash, "warm": warm, "screen": screen}


def group_split(groups, n_folds, seed):
    """Fold label per sample, keeping every sample of a species in the same fold."""
    names = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(names))
    fold = {names[index]: position % n_folds for position, index in enumerate(order)}
    return np.array([fold[group] for group in groups])


def _ridge(x, y, alpha=ALPHA):
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    u, s, vt = np.linalg.svd(x - x_mean, full_matrices=False)
    weights = (vt.T * (s / (s ** 2 + alpha))) @ (u.T @ (y - y_mean))
    return lambda new: (new - x_mean) @ weights + y_mean


def _describe(data, index):
    parts = [data["domain"][index], data["object"][index]]
    if data["color"][index]:
        parts.append(data["color"][index].lower())
    return " · ".join(parts)


def run_measured(seed=7, noise=0.005):
    """Held-out errors on measured reflectance for each capture scenario."""
    seed = _validate_seed(seed)
    if (isinstance(noise, (bool, np.bool_)) or not isinstance(noise, (int, float, np.integer, np.floating))
            or not np.isfinite(noise) or not 0 <= noise <= 0.1):
        raise ValueError("noise must be finite and between 0 and 0.1")
    data = natural_objects()
    spectra = data["spectra"]
    camera = _camera()
    lights = illustrative_lights()
    flash = lights["flash"]
    rng = np.random.default_rng(seed)
    folds = group_split(data["group"], 5, seed)

    def photo(light, factor=1.0):
        clean = (spectra * light) @ camera.T
        return clean + rng.normal(0, noise * factor, clean.shape)

    def held_out(train_x, test_x):
        estimate = np.empty_like(spectra)
        for fold in range(5):
            train, test = folds != fold, folds == fold
            estimate[test] = _ridge(train_x[train], spectra[train])(test_x[test])
        return estimate

    base = photo(flash)
    three_brightness = [0.5 * flash, flash, 1.5 * flash]
    flash_screen = [flash, *lights["screen"]]
    cases = [
        ("flash", "One photo · same light as calibration", base, base, [flash],
         "The fitted map reuses the light it was calibrated under. This is the best case for one photo."),
        ("changed_light", "One photo · changed light, no refit", base, photo(lights["warm"]), [lights["warm"]],
         "The same map, applied under warmer light without recalibrating. A real scene's light shifts like this all the time."),
        ("brightness", "Three brightness levels", None, None, three_brightness,
         "Nine numbers are recorded, but scaling one light repeats the same three measurements. Noise falls a little; no new spectral information arrives."),
        ("screen", "Flash plus three screen colors", None, None, flash_screen,
         "Twelve numbers from four different lights. Different light shapes can reveal spectral directions one flash cannot see, but screen light is dimmer, so these photos carry more noise."),
    ]
    scenarios = []
    for identifier, label, train_x, test_x, used, description in cases:
        if train_x is None:
            factors = [3.0 if identifier == "screen" and i > 0 else 1.0 for i in range(len(used))]
            train_x = test_x = np.column_stack([photo(light, factor) for light, factor in zip(used, factors)])
        estimate = held_out(train_x, test_x)
        sam, mae = spectral_angle(spectra, estimate), mean_absolute_error(spectra, estimate)
        within = (sam <= DRAFT_TARGETS["sam_deg"]) & (mae <= DRAFT_TARGETS["mae"])
        operator = np.vstack([camera * light for light in used])
        values = np.linalg.svd(operator, compute_uv=False)
        typical = int(np.argsort(mae)[len(mae) // 2])
        worst = int(np.argmax(mae))
        scenarios.append({
            "id": identifier, "label": label, "description": description,
            "channels": 3 * len(used), "independent": int(np.sum(values > 0.02 * values[0])),
            "median_sam_deg": float(np.median(sam)), "median_mae": float(np.median(mae)),
            "p90_mae": float(np.percentile(mae, 90)), "worst_mae": float(mae.max()),
            "share_within_targets": float(within.mean()), "n_test_targets": len(mae),
            "examples": {
                "typical": {"name": _describe(data, typical), "reference": spectra[typical].tolist(), "estimate": estimate[typical].tolist(),
                            "sam_deg": float(sam[typical]), "mae": float(mae[typical])},
                "worst": {"name": _describe(data, worst), "reference": spectra[worst].tolist(), "estimate": estimate[worst].tolist(),
                          "sam_deg": float(sam[worst]), "mae": float(mae[worst])},
            },
        })
    return {
        "kind": "bench", "source": "measured", "seed": seed, "noise": float(noise), "bands_nm": BANDS_NM.tolist(),
        "scenarios": scenarios, "metamer": measured_metamer(),
        "notes": [
            "Reflectance spectra are measured (532 samples of natural objects, Gutiérrez et al. 2025). The camera and lights are illustrative curves, and RGB values are simulated.",
            "Every estimate is held out: 5-fold cross-validation with all samples of a species or specimen in one fold.",
            "Each sample is captured once per light, with independent additive noise. Each light is exposed to match flash on a white target, and screen photos carry three times the noise.",
            "No phone or mushroom was measured for this bench. Draft targets (SAM ≤ 10°, MAE ≤ 0.05) are unvalidated placeholders.",
            "Independent measurements count singular values of the stacked sensitivity operator above 2% of the largest.",
        ],
    }


def measured_metamer():
    """The real pair from the study: same flash RGB within noise, different spectra."""
    from .data import findings
    try:
        pair = findings()["metamer"]
    except (OSError, KeyError):
        return None
    return {
        "bands_nm": pair["bands_nm"], "first": pair["spectrum_a"], "second": pair["spectrum_b"],
        "first_name": pair["name_a"], "second_name": pair["name_b"],
        "sam_deg": pair["sam_deg"], "mae": pair["mae"],
        "rgb_a": pair["flash_response_a"], "rgb_b": pair["flash_response_b"],
        "description": pair["description"],
    }


def run_idealized(seed=7, noise=0.003):
    """The original synthetic bench, kept to show what a model-matched test claims."""
    report = run_synthetic(seed=seed, noise=noise)
    scenarios = []
    for row in report["scenarios"]:
        channels = {"matched": 3, "shifted": 3, "intensity_only": 9, "spectrally_distinct": 9}[row["id"]]
        scenarios.append({
            "id": row["id"], "label": row["label"], "description": row["description"],
            "channels": channels, "independent": row["rank"],
            "median_sam_deg": row["median_sam_deg"], "median_mae": row["median_mae"],
            "p90_mae": None, "worst_mae": None, "share_within_targets": None, "n_test_targets": row["n_test_targets"],
            "examples": {"typical": {"name": "Generated spectrum", "reference": row["reference"], "estimate": row["estimate"],
                                     "sam_deg": None, "mae": None}},
        })
    metamer = dict(report["metamer"], first_name="Constructed spectrum A", second_name="Constructed spectrum B")
    return {"kind": "bench", "source": "idealized", "seed": report["seed"], "noise": report["noise"], "bands_nm": report["bands_nm"],
            "scenarios": scenarios, "metamer": metamer,
            "notes": ["Spectra are generated from the same four-coefficient model the solver assumes. Treat this as a check of the code and a best case, not a performance estimate.", *report["notes"]]}


def run_bench(seed=7, noise=None, source="measured"):
    if source == "measured":
        return run_measured(seed, 0.005 if noise is None else noise)
    if source == "idealized":
        return run_idealized(seed, 0.003 if noise is None else noise)
    raise ValueError("source must be 'measured' or 'idealized'")
