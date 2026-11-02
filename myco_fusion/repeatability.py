"""Analyze a phone-only test: repeatability across sessions and independent measurements.

Input is one row per photo, with mean region-of-interest values per channel. Objects
are photographed again in every session, so objects and sessions are crossed. Variance
components come from REML with Fisher scoring, which also handles missing photos.
"""
from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter

import numpy as np

from .intervals import chi2_quantile

REQUIRED = {"object_id", "session_id", "light", "repeat", "r", "g", "b"}
CHANNELS = ("r", "g", "b")
MAX_PHOTOS = 4000
MAX_PHOTOS_PER_LIGHT = 1000
COMPONENTS = ("object", "session", "object_by_session", "residual")


def parse_rows(text):
    """Read a photo table. Accepts comma, tab or semicolon separators and tidy-able headers."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("CSV text is empty")
    clean = text.lstrip("﻿").lstrip()
    first_line = clean.split("\n", 1)[0]
    delimiter = max(",\t;", key=first_line.count)
    reader = csv.reader(io.StringIO(clean), delimiter=delimiter)
    header = [name.strip().lower() for name in next(reader)]
    if not REQUIRED.issubset(header):
        raise ValueError("CSV requires columns: " + ", ".join(sorted(REQUIRED)))
    if len(set(header)) != len(header):
        raise ValueError("CSV contains duplicate column names")
    rows, seen = [], set()
    for line, cells in enumerate(reader, start=2):
        if not any(cell.strip() for cell in cells):
            continue
        if len(rows) >= MAX_PHOTOS:
            raise ValueError(f"This analysis accepts at most {MAX_PHOTOS} photos")
        if len(cells) != len(header):
            raise ValueError(f"Malformed CSV row at line {line}")
        row = dict(zip(header, cells))
        identity = (row["object_id"].strip(), row["session_id"].strip(), row["light"].strip().lower(), row["repeat"].strip())
        if not all(identity):
            raise ValueError(f"object_id, session_id, light and repeat must be nonempty (line {line})")
        if identity in seen:
            raise ValueError(f"Duplicate photo {identity} at line {line}")
        seen.add(identity)
        try:
            values = [float(row[channel]) for channel in CHANNELS]
        except ValueError as exc:
            raise ValueError(f"Non-numeric channel value at line {line}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"Channel values must be finite (line {line})")
        rows.append({"object": identity[0], "session": identity[1], "light": identity[2], "repeat": identity[3], "values": values,
                     "synthetic": row.get("data_kind", "").strip() == "synthetic"})
    if not rows:
        raise ValueError("CSV has no photos")
    return rows


def _indicator(labels):
    names = sorted(set(labels))
    position = {name: i for i, name in enumerate(names)}
    z = np.zeros((len(labels), len(names)))
    z[np.arange(len(labels)), [position[label] for label in labels]] = 1
    return z


def variance_components(y, objects, sessions, iterations=200):
    """REML estimates for object, session, object-by-session and residual variance.

    Returns (estimates, information-based variances, residual degrees of freedom,
    estimable flags), each in the order object, session, object_by_session, residual.
    A component the design cannot separate is held at zero and flagged. Components at
    the zero boundary stay there only while the likelihood does not want them back.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    # Work on a unit-variance scale so very dim channels stay numerically well behaved.
    scale = float(np.std(y, ddof=1)) if n > 1 else 0.0
    if scale > 0:
        y = (y - y.mean()) / scale
    cells = [f"{a}␟{b}" for a, b in zip(objects, sessions)]
    n_objects, n_sessions, n_cells = len(set(objects)), len(set(sessions)), len(set(cells))
    x = np.ones((n, 1))
    designs = {"object": _indicator(objects), "session": _indicator(sessions), "object_by_session": _indicator(cells), "residual": np.eye(n)}
    estimable = {
        "object": n_objects >= 2,
        "session": n_sessions >= 2,
        # Needs both factors, replicate photos to separate it from residual, and more cells than main effects.
        "object_by_session": n_objects >= 2 and n_sessions >= 2 and n_cells < n and n_cells > n_objects + n_sessions - 1,
    }
    fixed = [x] + [designs[name] for name in ("object", "session", "object_by_session") if estimable[name]]
    residual_df = n - int(np.linalg.matrix_rank(np.column_stack(fixed)))
    estimable["residual"] = residual_df > 0
    active = [name for name in COMPONENTS if estimable[name]]
    if "residual" not in active:
        raise ValueError("These photos cannot separate noise from real differences. Take at least two photos of each object, or add a session.")
    kernels = [designs[name] @ designs[name].T for name in active]
    if scale == 0:
        return np.zeros(4), np.zeros(4), residual_df, [estimable[name] for name in COMPONENTS]
    total = 1.0
    floor = 1e-8

    def evaluate(theta):
        # A trial step can be wildly off; it is rejected below if its likelihood is not finite.
        with np.errstate(all="ignore"):
            try:
                v = sum(t * k for t, k in zip(theta, kernels)) + floor * np.eye(n)
                vi = np.linalg.inv(v)
                vix = vi @ x
                information = x.T @ vix
                p = vi - vix @ np.linalg.inv(information) @ vix.T
                _, log_v = np.linalg.slogdet(v)
                _, log_x = np.linalg.slogdet(information)
            except np.linalg.LinAlgError:
                return None, -np.inf
            return p, -0.5 * (log_v + log_x + y @ p @ y)

    theta = np.full(len(active), total / len(active))
    p, likelihood = evaluate(theta)
    for _ in range(iterations):
        py = p @ y
        pk = [p @ k for k in kernels]
        score = np.array([0.5 * (py @ k @ py - np.trace(q)) for k, q in zip(kernels, pk)])
        info = 0.5 * np.array([[np.sum(a * b.T) for b in pk] for a in pk])
        free = (theta > 0) | (score > 0)
        step = np.zeros(len(theta))
        if free.any():
            step[free] = np.linalg.pinv(info[np.ix_(free, free)]) @ score[free]
        moved = False
        for shrink in (1, .5, .25, .125, .0625, .03125):
            candidate = np.maximum(theta + shrink * step, 0)
            new_p, new_likelihood = evaluate(candidate)
            if np.isfinite(new_likelihood) and new_likelihood >= likelihood - 1e-10:
                moved = np.max(np.abs(candidate - theta)) > 1e-11 * total
                theta, p, likelihood = candidate, new_p, new_likelihood
                break
        if not moved:
            break
    pk = [p @ k for k in kernels]
    info = 0.5 * np.array([[np.sum(a * b.T) for b in pk] for a in pk])
    positive = theta > 0
    covariance = np.zeros((len(theta), len(theta)))
    if positive.any():
        covariance[np.ix_(positive, positive)] = np.linalg.pinv(info[np.ix_(positive, positive)])
    estimates = np.zeros(4)
    variances = np.zeros(4)
    for index, name in enumerate(active):
        position = COMPONENTS.index(name)
        estimates[position] = theta[index] * scale ** 2
        variances[position] = max(covariance[index, index], 0.0) * scale ** 4
    return estimates, variances, residual_df, [estimable[name] for name in COMPONENTS]


def _sd_interval(variance, variance_of_estimate):
    """Approximate 95% interval for an SD using Satterthwaite degrees of freedom."""
    if variance <= 0 or variance_of_estimate <= 0:
        return None
    df = 2 * variance ** 2 / variance_of_estimate
    if df < 1:  # Too little information; the upper limit would be astronomically large.
        return None
    return {"low": float(np.sqrt(df * variance / chi2_quantile(0.975, df))),
            "high": float(np.sqrt(df * variance / chi2_quantile(0.025, df))), "df": float(df)}


def analyze_channel(y, objects, sessions):
    theta, variances, residual_df, estimable = variance_components(y, objects, sessions)
    var_object, var_session, var_cross, var_residual = (float(t) for t in theta)
    has_object, has_session, has_cross, _ = estimable
    sd = lambda variance, known: float(np.sqrt(variance)) if known else None
    between = var_session + var_cross + var_residual
    total = var_object + between
    return {
        "mean": float(np.mean(y)),
        "sd_object": sd(var_object, has_object), "sd_session": sd(var_session, has_session),
        "sd_object_by_session": sd(var_cross, has_cross), "sd_residual": float(np.sqrt(var_residual)),
        "sd_session_interval": _sd_interval(var_session, float(variances[1])),
        "sd_residual_interval": _sd_interval(var_residual, float(variances[3])),
        "repeatability_coefficient": float(1.959963984540054 * np.sqrt(2 * var_residual)),
        "reproducibility_coefficient": float(1.959963984540054 * np.sqrt(2 * between)) if has_session else None,
        "icc_agreement": float(var_object / total) if has_object and total > 0 else None,
        "residual_df": int(residual_df), "residual_includes_object_by_session": bool(has_object and has_session and not has_cross),
    }


def independent_measurements(rows, lights, seed=20261101, draws=300):
    """Count measurement directions that rise above the noise floor of the object means.

    Averages photos into one row per object and one column per light and channel, then
    compares singular values with simulated noise built by resampling each object's own
    session means (parallel analysis). Resampling whole sessions keeps errors shared by a
    session's photos and by a photo's channels, so they are not counted as new directions.
    With one session it falls back to resampling photos, which cannot see session drift.
    A direction counts only if it beats the 95th percentile of the strongest direction
    noise alone produces. The count can never exceed objects minus one, or the channels.
    """
    objects = sorted({row["object"] for row in rows})
    sessions = sorted({row["session"] for row in rows})
    columns = len(lights) * 3
    table = np.full((len(objects), columns), np.nan)
    pools = {}
    for i, name in enumerate(objects):
        for j, light in enumerate(lights):
            photos = [row for row in rows if row["object"] == name and row["light"] == light]
            if len(photos) < 2:
                continue
            means = np.array([np.mean([p["values"] for p in photos if p["session"] == s], axis=0) for s in sessions if any(p["session"] == s for p in photos)])
            units = means if len(means) >= 2 else np.array([p["values"] for p in photos])
            table[i, 3 * j:3 * j + 3] = units.mean(axis=0)
            # Scale up because residuals around their own mean are slightly too small.
            pools[(i, j)] = (units - units.mean(axis=0)) * np.sqrt(len(units) / (len(units) - 1))
    complete = [i for i in range(len(objects)) if not np.isnan(table[i]).any()]
    if len(complete) < 3:
        return None
    table = table[complete]
    rng = np.random.default_rng(seed)

    def noise_table():
        noise = np.empty_like(table)
        for row, i in enumerate(complete):
            for j in range(len(lights)):
                pool = pools[(i, j)]
                noise[row, 3 * j:3 * j + 3] = pool[rng.integers(0, len(pool), len(pool))].mean(axis=0)
        return noise

    samples = np.array([noise_table() for _ in range(draws)])
    noise_sd = samples.std(axis=(0, 1))
    if np.any(noise_sd <= 0):
        return None
    observed = np.linalg.svd((table - table.mean(axis=0)) / noise_sd, compute_uv=False)
    strongest = [np.linalg.svd((sample - sample.mean(axis=0)) / noise_sd, compute_uv=False)[0] for sample in samples]
    floor = float(np.percentile(strongest, 95))
    return {"lights": list(lights), "n_objects": len(table), "n_objects_excluded": len(objects) - len(table), "channels": columns,
            "independent": int(np.sum(observed > floor)), "singular_values": observed.tolist(), "noise_floor": floor}


def analyze_repeatability(text, independent_sets=None):
    """Full report for one phone test. ``independent_sets`` names groups of lights to compare."""
    rows = parse_rows(text)
    lights = sorted({row["light"] for row in rows})
    results = []
    for light in lights:
        subset = [row for row in rows if row["light"] == light]
        if len(subset) > MAX_PHOTOS_PER_LIGHT:
            raise ValueError(f"Light {light!r} has {len(subset)} photos; this analysis accepts at most {MAX_PHOTOS_PER_LIGHT} per light")
        objects = [row["object"] for row in subset]
        sessions = [row["session"] for row in subset]
        if len({(o, s) for o, s in zip(objects, sessions)}) == len(subset) and len(set(sessions)) < 2:
            raise ValueError(f"Light {light!r} has one photo per object in one session, which cannot separate noise from real differences. Take at least two photos of each object.")
        channels = {channel: analyze_channel(np.array([row["values"][index] for row in subset]), objects, sessions) for index, channel in enumerate(CHANNELS)}
        results.append({"light": light, "n_photos": len(subset), "n_objects": len(set(objects)), "n_sessions": len(set(sessions)), "channels": channels})
    counts = Counter((row["object"], row["session"], row["light"]) for row in rows)
    objects = sorted({row["object"] for row in rows})
    sessions = sorted({row["session"] for row in rows})
    expected = len(objects) * len(sessions) * len(lights)
    missing = expected - len(counts)
    sessions_per_object = Counter()
    for name in objects:
        sessions_per_object[name] = len({row["session"] for row in rows if row["object"] == name})
    crossed = any(value >= 2 for value in sessions_per_object.values())
    if independent_sets is None:
        independent_sets = [[light] for light in lights] + ([lights] if len(lights) > 1 else [])
    rank_results, notes = [], [
        "Values are used as supplied. Tone-mapped JPEG or HEIC values are not sensor-linear, and tone curves can add directions that are not real measurements. Use linear or RAW-derived values before relying on the independent-measurement count.",
        "This test measures how stable the phone is. It says nothing about spectral accuracy, which needs a reference instrument.",
        "Objects are real signal, not noise. A high ICC with low session and photo-to-photo SD is the goal.",
    ]
    for group in independent_sets:
        result = independent_measurements(rows, group)
        label = ", ".join(group)
        if result is None:
            notes.append(f"Independent measurements not counted for {label}: at least 3 objects need 2 or more photos under every one of these lights.")
            continue
        rank_results.append(result)
        if result["n_objects_excluded"]:
            notes.append(f"Independent measurements for {label} leave out {result['n_objects_excluded']} objects without enough photos under every one of these lights.")
    if len(sessions) < 2:
        notes.append("With one session, session-to-session drift cannot be estimated, so session SD and the reproducibility coefficient are not reported, and the independent-measurement count cannot see drift. Add sessions.")
    elif len(sessions) < 6:
        notes.append(f"With {len(sessions)} sessions, session variance rests on {len(sessions) - 1} degrees of freedom. Treat the session SD as a rough guide.")
    if not crossed and len(sessions) >= 2:
        notes.append("No object appears in more than one session, so object and session effects cannot be separated. Photograph the same objects in every session.")
    if any(result["channels"]["r"]["residual_includes_object_by_session"] for result in results):
        notes.append("With one photo per object and session, photo-to-photo noise cannot be separated from object-by-session differences. Their total is reported as photo-to-photo SD.")
    if missing:
        notes.append(f"{missing} object/session/light combinations have no photos. REML handles this, but check that the gaps are not related to the object or the light.")
    return {
        "kind": "repeatability", "schema_version": "0.1",
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "data_kind": "synthetic" if all(row["synthetic"] for row in rows) else "user_supplied_unverified",
        "counts": {"n_photos": len(rows), "n_objects": len(objects), "n_sessions": len(sessions), "n_lights": len(lights),
                   "n_missing_combinations": missing, "photos_per_combination_median": float(np.median(list(counts.values()))), "crossed": crossed},
        "lights": results, "independent_measurements": rank_results, "notes": notes,
    }
