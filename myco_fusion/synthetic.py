"""Reproducible, explicitly idealized experiments in spectral identifiability."""
from __future__ import annotations

import numpy as np

from .spectra import BANDS_NM, _validate_seed, apply_tm, fit_tm, mean_absolute_error, spectral_angle


def _basis():
    t = np.linspace(0, np.pi, len(BANDS_NM))
    return np.stack([np.cos(k * t) for k in range(4)], axis=1)


def _camera():
    centers = np.array([460., 540., 620.])[:, None]
    widths = np.array([38., 43., 42.])[:, None]
    sensitivity = np.exp(-0.5 * ((BANDS_NM[None, :] - centers) / widths) ** 2)
    return sensitivity / sensitivity.sum(axis=1, keepdims=True)


def _physical_spectra(rng, basis, n):
    coefficients = rng.uniform(-0.055, 0.055, size=(n, 4))
    coefficients[:, 0] = rng.uniform(0.25, 0.75, size=n)
    # Sum of three oscillating terms <= .165: every spectrum lies in [.085,.915].
    return coefficients @ basis.T


def run_synthetic(seed: int = 7, noise: float = 0.003) -> dict:
    """Return a JSON-safe report; synthetic errors are observations, never gates.

    The known-operator scenarios assume exact sensitivity, illumination and a
    four-coefficient reflectance basis. They do not estimate illumination from
    a tile and are not evidence of real camera or fungal performance.
    """
    seed = _validate_seed(seed)
    if (isinstance(noise, (bool, np.bool_))
            or not isinstance(noise, (int, float, np.integer, np.floating))
            or not np.isfinite(noise) or not 0 <= noise <= 0.1):
        raise ValueError("noise must be finite and between 0 and 0.1")
    rng = np.random.default_rng(seed)
    basis, camera = _basis(), _camera()
    calibration = _physical_spectra(rng, basis, 96)
    reference = _physical_spectra(rng, basis, 40)
    flash = np.ones(len(BANDS_NM))
    shifted = 0.65 + 0.9 * np.exp(-0.5 * ((BANDS_NM - 550) / 32) ** 2)
    distinct = [flash,
                0.3 + 1.2 * np.exp(-0.5 * ((BANDS_NM - 450) / 25) ** 2),
                0.3 + 1.2 * np.exp(-0.5 * ((BANDS_NM - 625) / 25) ** 2)]

    def observe(spectra, illuminant):
        response = (spectra * illuminant) @ camera.T
        return response + rng.normal(0, noise, size=response.shape)

    weights = fit_tm(observe(calibration, flash), calibration)
    matched = apply_tm(observe(reference, flash), weights)
    changed = apply_tm(observe(reference, shifted), weights)
    scenarios = []

    def record(identifier, label, estimate, operators, description):
        operator = np.vstack(operators)
        rank = int(np.linalg.matrix_rank(operator))
        scenarios.append({
            "id": identifier, "label": label, "rank": rank, "basis_dim": 4,
            "condition_number": float(np.linalg.cond(operator)) if rank == 4 else None,
            "median_sam_deg": float(np.median(spectral_angle(reference, estimate))),
            "median_mae": float(np.median(mean_absolute_error(reference, estimate))),
            "reference": reference[0].tolist(), "estimate": estimate[0].tolist(),
            "description": description, "n_test_targets": len(reference),
            "n_out_of_range_estimates": int(((estimate < 0) | (estimate > 1)).sum()),
        })

    record("matched", "Matched illumination · RGB regression", matched, [camera @ basis],
           "Affine ridge calibration uses 96 independent synthetic targets; 40 different targets are evaluated under the same illumination. The learned prior can yield low error despite a rank-deficient observation operator.")
    record("shifted", "Changed illumination · same RGB regression", changed,
           [camera @ np.diag(shifted) @ basis],
           "The calibration map is reused without refitting under a changed spectral illuminant. The measured synthetic degradation is not a required failure or a universal bound.")
    for identifier, label, illuminants, description in [
        ("intensity_only", "Three brightness levels · known operator", [0.5 * flash, flash, 1.5 * flash],
         "Scalar multiples of one illumination stack redundant operators. More exposures may reduce noise, but their rank remains 3 for the four-coefficient reflectance basis."),
        ("spectrally_distinct", "Three spectral illuminants · known operator", distinct,
         "Different known spectral illumination shapes yield a full-rank operator in this deliberately low-dimensional model. Exact camera sensitivity and illumination are assumed, not estimated."),
    ]:
        operators = [camera @ np.diag(e) @ basis for e in illuminants]
        responses = np.concatenate([observe(reference, e) for e in illuminants], axis=1)
        coefficients = np.linalg.lstsq(np.vstack(operators), responses.T, rcond=None)[0].T
        estimate = coefficients @ basis.T
        record(identifier, label, estimate, operators, description)

    # Exhibit two distinct physical 27-band spectra with indistinguishable RGB.
    _, _, right = np.linalg.svd(camera, full_matrices=True)
    delta = right[-1] * (0.25 / np.max(np.abs(right[-1])))
    first, second = 0.5 + delta, 0.5 - delta
    metamer = {
        "rgb_difference": float(np.max(np.abs(camera @ first - camera @ second))),
        "spectral_rmse": float(np.sqrt(np.mean((first - second) ** 2))),
        "bands_nm": BANDS_NM.tolist(), "first": first.tolist(), "second": second.tolist(),
        "description": "A noiseless algebraic metamer pair under the illustrative three-channel camera and flat illumination; not measured materials.",
    }
    return {
        "kind": "synthetic", "seed": int(seed), "noise": float(noise),
        "bands_nm": BANDS_NM.tolist(), "scenarios": scenarios, "metamer": metamer,
        "notes": [
            "All data and errors in this report are synthetic. No phone, spectrometer or mushroom was measured.",
            "Illuminants and camera sensitivities are assumed known; the four-coefficient basis is an idealized prior.",
            "The estimated 27-band outputs are not 27 independently measured channels.",
            "Rank refers to the stacked known measurement operator on the four-dimensional basis; scalar brightness changes do not add spectral rank.",
            "Rank sufficiency here does not prove identifiability for an unknown illuminant, nonlinear ISP or real reflectance distribution.",
            "Curves show the first held-out target; reported medians use all 40 held-out targets. Predictions are not clipped.",
        ],
    }
