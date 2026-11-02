"""Forward model, estimators, grouped cross-validation and metrics for the study."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from myco_fusion.spectra import BANDS_NM, mean_absolute_error, spectral_angle
from myco_fusion.validation import DRAFT_TARGETS

SIM_NM = np.arange(400, 701, 5.0)
METRIC_NM = BANDS_NM.astype(float)
ALPHAS = (0.0, 1e-6, 1e-4, 1e-2, 1.0)


@dataclass(frozen=True)
class Noise:
    additive: float
    gain: float


NOISE = {"low": Noise(0.002, 0.005), "moderate": Noise(0.005, 0.02), "high": Noise(0.015, 0.05)}


def resample(wavelengths, values, grid):
    """Linear interpolation along the last axis."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.interp(grid, wavelengths, values)
    return np.array([np.interp(grid, wavelengths, row) for row in values])


def light_curve(wavelengths, values):
    curve = np.maximum(resample(wavelengths, values, SIM_NM), 0)
    return curve / curve.mean()


def camera_curves(wavelengths, values):
    """Sensitivities as rows, normalized so a perfect white under equal energy reads 1."""
    table = np.asarray(values, dtype=float)
    curves = np.array([np.interp(SIM_NM, wavelengths, table[:, c]) for c in range(3)])
    return curves / curves.sum(axis=1, keepdims=True)


def gaussian_light(center, fwhm):
    sd = fwhm / 2.3548200450309493
    return light_curve(SIM_NM, np.exp(-0.5 * ((SIM_NM - center) / sd) ** 2))


def match_exposure(curve, reference, camera):
    """Scale a light as exposure would: the same strongest-channel reading on white as the reference."""
    return curve * (camera @ reference).max() / (camera @ curve).max()


def capture(spectra, light, camera, noise, rng, additive_factor=1.0):
    """One photo per sample: shared gain error plus independent additive channel noise."""
    clean = (spectra * light) @ camera.T
    gain = 1 + rng.normal(0, noise.gain, (len(clean), 1))
    return clean * gain + rng.normal(0, noise.additive * additive_factor, clean.shape)


def capture_arm(spectra, lights, camera, noise, rng):
    """Stack independent captures. ``lights`` is a list of (curve, additive_factor)."""
    return np.column_stack([capture(spectra, curve, camera, noise, rng, factor) for curve, factor in lights])


def operator_rank(lights, camera, threshold=0.02):
    """Independent measurement directions: singular values above 2% of the largest."""
    operator = np.vstack([camera * curve for curve, _ in lights])
    values = np.linalg.svd(operator, compute_uv=False)
    return int(np.sum(values > threshold * values[0]))


class Ridge:
    """Affine ridge regression with an unpenalized intercept, like the app's calibration."""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, x, y):
        self.x_mean, self.y_mean = x.mean(axis=0), y.mean(axis=0)
        u, s, vt = np.linalg.svd(x - self.x_mean, full_matrices=False)
        shrink = s / (s ** 2 + self.alpha)
        shrink[s < 1e-12 * max(s.max(initial=0.0), 1e-300)] = 0
        self.weights = (vt.T * shrink) @ (u.T @ (y - self.y_mean))
        return self

    def predict(self, x):
        return (x - self.x_mean) @ self.weights + self.y_mean


def polynomial_features(x):
    columns = [x]
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]):
            columns.append((x[:, i] * x[:, j])[:, None])
    return np.column_stack(columns)


class KernelRidge:
    """RBF kernel ridge on standardized channels."""

    def __init__(self, alpha, gamma):
        self.alpha, self.gamma = alpha, gamma

    def _kernel(self, a, b):
        distance = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-self.gamma * distance)

    def fit(self, x, y):
        self.x_mean, self.x_sd = x.mean(axis=0), x.std(axis=0) + 1e-12
        self.y_mean = y.mean(axis=0)
        self.train = (x - self.x_mean) / self.x_sd
        k = self._kernel(self.train, self.train)
        self.dual = np.linalg.solve(k + self.alpha * np.eye(len(k)), y - self.y_mean)
        return self

    def predict(self, x):
        return self._kernel((x - self.x_mean) / self.x_sd, self.train) @ self.dual + self.y_mean


def group_folds(groups, k, rng):
    """Assign whole groups to folds, largest first, always filling the smallest fold."""
    names, inverse, counts = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    k = min(k, len(names))
    order = rng.permutation(len(names))
    order = order[np.argsort(-counts[order], kind="stable")]
    sizes = np.zeros(k, dtype=int)
    assignment = np.empty(len(names), dtype=int)
    for index in order:
        fold = int(np.argmin(sizes))
        assignment[index] = fold
        sizes[fold] += counts[index]
    return assignment[inverse]


def _inner_score(x, y, groups, build, rng, k=4):
    folds = group_folds(groups, k, rng)
    errors = []
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        if train.sum() < 2:
            continue
        estimate = build().fit(x[train], y[train]).predict(x[test])
        errors.append(np.mean(np.abs(estimate - y[test])))
    return float(np.mean(errors)) if errors else np.inf


def fit_selected(x, y, groups, rng, kind="ridge"):
    """Fit one estimator family, choosing its penalty by inner grouped cross-validation."""
    if kind == "ridge":
        candidates = [lambda a=a: Ridge(a) for a in ALPHAS]
    elif kind == "polynomial":
        x = polynomial_features(x)
        candidates = [lambda a=a: Ridge(a) for a in ALPHAS]
    elif kind == "kernel":
        candidates = [lambda a=a, g=g: KernelRidge(a, g) for a in (1e-3, 1e-2, 1e-1, 1.0) for g in (0.03, 0.1, 0.3, 1.0)]
    else:
        raise ValueError(kind)
    scores = [_inner_score(x, y, groups, build, rng) for build in candidates]
    model = candidates[int(np.argmin(scores))]().fit(x, y)
    if kind == "polynomial":
        class Wrapped:
            predict = staticmethod(lambda new: model.predict(polynomial_features(new)))
        return Wrapped
    return model


def cv_predict(x, y, folds, groups, rng, kind="ridge"):
    """Held-out predictions using the supplied fold labels."""
    groups = np.asarray(groups)
    estimate = np.empty_like(y)
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        estimate[test] = fit_selected(x[train], y[train], groups[train], rng, kind).predict(x[test])
    return estimate


def errors(reference, estimate):
    return spectral_angle(reference, estimate), mean_absolute_error(reference, estimate)


class Evaluation:
    """Errors per noise draw and sample. Summaries use each sample's error averaged over draws."""

    def __init__(self, sams, maes):
        self.sams, self.maes = np.asarray(sams), np.asarray(maes)
        self.mae = self.maes.mean(axis=0)
        self.sam = self.sams.mean(axis=0)
        self.within = ((self.sams <= DRAFT_TARGETS["sam_deg"]) & (self.maes <= DRAFT_TARGETS["mae"])).mean(axis=0)

    def summary(self, mask=None):
        mask = np.ones(len(self.mae), dtype=bool) if mask is None else mask
        sam, mae = self.sam[mask], self.mae[mask]
        return {"n": int(mask.sum()),
                "sam_median": float(np.median(sam)), "sam_p90": float(np.percentile(sam, 90)), "sam_worst": float(np.max(sam)),
                "mae_median": float(np.median(mae)), "mae_p90": float(np.percentile(mae, 90)), "mae_worst": float(np.max(mae)),
                "share_within_targets": float(self.within[mask].mean())}


def _cluster_members(groups):
    names, inverse = np.unique(np.asarray(groups), return_inverse=True)
    return [np.flatnonzero(inverse == i) for i in range(len(names))]


def paired_reduction(base_mae, arm_mae, groups, rng, resamples=2000, statistic="paired"):
    """Percent reduction in MAE with a 95% cluster bootstrap by group.

    ``paired`` is the median per-sample drop as a percent of the baseline median, the
    statistic the plan names for comparing arms. ``ratio`` is the drop in the median itself.
    """
    base_mae, arm_mae = np.asarray(base_mae), np.asarray(arm_mae)
    members = _cluster_members(groups)

    def reduction(index):
        if statistic == "paired":
            return 100 * np.median(base_mae[index] - arm_mae[index]) / np.median(base_mae[index])
        return 100 * (1 - np.median(arm_mae[index]) / np.median(base_mae[index]))

    point = float(reduction(np.arange(len(base_mae))))
    draws = np.empty(resamples)
    for b in range(resamples):
        chosen = rng.integers(0, len(members), len(members))
        draws[b] = reduction(np.concatenate([members[i] for i in chosen]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"percent": point, "low": float(low), "high": float(high), "n_clusters": len(members), "statistic": statistic}


def share_bootstrap(within, groups, rng, resamples=2000):
    """95% cluster bootstrap for the share of samples meeting both draft targets."""
    within = np.asarray(within)
    members = _cluster_members(groups)
    draws = np.empty(resamples)
    for b in range(resamples):
        chosen = rng.integers(0, len(members), len(members))
        draws[b] = within[np.concatenate([members[i] for i in chosen])].mean()
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}
