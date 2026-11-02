"""Small-sample intervals that report the confidence they actually reach.

A percentile bootstrap labeled "95%" can fall far short with few independent
units (docs/FINDINGS.md has the simulation). The order-statistic interval below
makes no shape assumption and states its true level, which may be below 95%.
"""
from __future__ import annotations

from math import comb, exp, lgamma, log, sqrt

import numpy as np


def median_interval(values, level=0.95):
    """Exact interval for a population median from independent units.

    Uses the narrowest symmetric pair of order statistics whose coverage is at
    least ``level``. With too few units to reach ``level`` (fewer than six at
    95%), it returns the full range and that range's lower, true level. Returns
    None for fewer than two units, because no interval can be formed.
    """
    data = np.sort(np.asarray(values, dtype=float))
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("values must be a finite one-dimensional array")
    n = len(data)
    if n < 2:
        return None
    total = 2 ** n
    best_rank, achieved, tail = 1, 1 - 2 * comb(n, 0) / total, comb(n, 0)
    for rank in range(2, n // 2 + 1):
        tail += comb(n, rank - 1)
        coverage = 1 - 2 * tail / total
        if coverage < level:
            break
        best_rank, achieved = rank, coverage
    return {"low": float(data[best_rank - 1]), "high": float(data[n - best_rank]),
            "level": float(achieved), "n_units": n, "reaches_target": bool(achieved >= level)}


def share_interval(count, n, z=1.959963984540054):
    """Wilson score interval for a share of independent units."""
    if n < 1:
        return None
    p = count / n
    denominator = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator
    return {"low": max(0.0, center - half), "high": min(1.0, center + half), "level": 0.95}


def _lower_gamma_regularized(a, x):
    """Regularized lower incomplete gamma P(a, x)."""
    if x <= 0:
        return 0.0
    log_front = a * log(x) - x - lgamma(a)
    if x < a + 1:  # Series expansion.
        term = total = 1 / a
        k = a
        for _ in range(10_000):
            k += 1
            term *= x / k
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * exp(log_front)
    # Lentz continued fraction for the upper tail.
    b = x + 1 - a
    c = 1e300
    d = 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c
        c = c if abs(c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return 1 - exp(log_front) * h


def chi2_quantile(p, df):
    """Chi-square quantile by bisection; df need not be an integer."""
    if not 0 < p < 1 or df <= 0:
        raise ValueError("p must lie in (0, 1) and df must be positive")
    low, high = 0.0, max(10.0, 2 * df)
    while _lower_gamma_regularized(df / 2, high / 2) < p:
        high *= 2
    for _ in range(200):
        middle = (low + high) / 2
        if _lower_gamma_regularized(df / 2, middle / 2) < p:
            low = middle
        else:
            high = middle
    return (low + high) / 2
