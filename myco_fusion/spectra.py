"""Small, explicit numerical primitives for estimated reflectance spectra.

RGB-to-spectrum regression is a learned estimate. No function in this module
turns three observed channels into 27 independent physical measurements.
"""
from __future__ import annotations

import numpy as np

BANDS_NM = np.arange(420, 681, 10, dtype=int)
BANDS = BANDS_NM


def _validate_seed(seed) -> int:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or not 0 <= seed <= 2**32 - 1):
        raise ValueError("seed must be an integer between 0 and 4294967295")
    return int(seed)


def _finite_array(value, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def fit_tm(rgb, reference, alpha: float = 1e-4) -> np.ndarray:
    """Fit affine ridge regression, returning coefficients shaped (4, 27).

    Rows are independent calibration patches. The intercept is not penalized.
    Calibration and evaluation targets must be separated by the caller.
    """
    x = _finite_array(rgb, "rgb")
    y = _finite_array(reference, "reference")
    if x.ndim != 2 or x.shape[1] != 3 or x.shape[0] < 4:
        raise ValueError("rgb must have shape (n, 3), with at least 4 calibration rows")
    if y.shape != (len(x), len(BANDS_NM)):
        raise ValueError("reference must have shape (n, 27), matching rgb rows")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("reference reflectance must lie in [0, 1]")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite nonnegative number")
    design = np.column_stack((x, np.ones(len(x))))
    penalty = np.diag([np.sqrt(alpha)] * 3 + [0.0])
    return np.linalg.lstsq(
        np.vstack((design, penalty)),
        np.vstack((y, np.zeros((4, len(BANDS_NM))))), rcond=None
    )[0]


def second_derivative_matrix(n_bands: int) -> np.ndarray:
    """Discrete second derivative operator matrix of shape (n_bands - 2, n_bands)."""
    if not isinstance(n_bands, (int, np.integer)) or n_bands < 3:
        raise ValueError("n_bands must be an integer >= 3")
    d2 = np.zeros((n_bands - 2, n_bands), dtype=float)
    for i in range(n_bands - 2):
        d2[i, i] = 1.0
        d2[i, i + 1] = -2.0
        d2[i, i + 2] = 1.0
    return d2


def fit_smooth_tm(rgb, reference, alpha: float = 1e-4, beta: float = 1e-3) -> np.ndarray:
    """Fit affine ridge regression with Tikhonov second-derivative roughness regularization.

    Solves the Sylvester system: (X^T X + alpha * D) W + W (beta * D_2^T D_2) = X^T Y
    via eigendecomposition of the discrete second-derivative roughness operator.
    When beta <= 0, this is identical to fit_tm.
    """
    x = _finite_array(rgb, "rgb")
    y = _finite_array(reference, "reference")
    if x.ndim != 2 or x.shape[1] != 3 or x.shape[0] < 4:
        raise ValueError("rgb must have shape (n, 3), with at least 4 calibration rows")
    if y.shape != (len(x), len(BANDS_NM)):
        raise ValueError("reference must have shape (n, 27), matching rgb rows")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("reference reflectance must lie in [0, 1]")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite nonnegative number")
    if not np.isscalar(beta) or not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be a finite nonnegative number")
    if beta <= 0:
        return fit_tm(x, y, alpha=alpha)
    n = len(x)
    design = np.column_stack((x, np.ones(n)))
    diag_alpha = np.array([alpha] * 3 + [0.0])
    A = design.T @ design + np.diag(diag_alpha)
    d2 = second_derivative_matrix(len(BANDS_NM))
    B = beta * (d2.T @ d2)
    C = design.T @ y
    vals, V = np.linalg.eigh(B)
    C_tilde = C @ V
    W_tilde = np.zeros_like(C_tilde)
    I4 = np.eye(4)
    for j in range(len(BANDS_NM)):
        W_tilde[:, j] = np.linalg.solve(A + vals[j] * I4, C_tilde[:, j])
    return W_tilde @ V.T


def apply_tm(rgb, weights) -> np.ndarray:
    """Apply a fitted map to (..., 3) values without clipping predictions."""
    x = _finite_array(rgb, "rgb")
    w = _finite_array(weights, "weights")
    if x.ndim < 1 or x.shape[-1] != 3:
        raise ValueError("rgb must have final dimension 3")
    if w.shape != (4, len(BANDS_NM)):
        raise ValueError("weights must have shape (4, 27)")
    flat = x.reshape(-1, 3)
    predicted = np.column_stack((flat, np.ones(len(flat)))) @ w
    return predicted.reshape(x.shape[:-1] + (len(BANDS_NM),))


def spectral_angle(reference, estimate):
    """Spectral angle in degrees; reject zero norms rather than invent an angle.

    Supports vectors and arrays of equal shape, with bands on the last axis.
    """
    a = _finite_array(reference, "reference")
    b = _finite_array(estimate, "estimate")
    if a.shape != b.shape or a.ndim < 1 or a.shape[-1] < 2:
        raise ValueError("spectra must have equal shapes and at least two bands")
    # Normalize each input before products to avoid norm-product overflow.
    scale_a = np.max(np.abs(a), axis=-1, keepdims=True)
    scale_b = np.max(np.abs(b), axis=-1, keepdims=True)
    if np.any(scale_a == 0) or np.any(scale_b == 0):
        raise ValueError("SAM is undefined for a zero-norm spectrum")
    a = a / scale_a
    b = b / scale_b
    cosine = np.sum(a * b, axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return float(angles) if angles.ndim == 0 else angles


sam = spectral_angle


def mean_absolute_error(reference, estimate):
    a = _finite_array(reference, "reference")
    b = _finite_array(estimate, "estimate")
    if a.shape != b.shape or a.ndim < 1 or a.shape[-1] < 2:
        raise ValueError("spectra must have equal shapes and at least two bands")
    error = np.mean(np.abs(a - b), axis=-1)
    return float(error) if error.ndim == 0 else error
