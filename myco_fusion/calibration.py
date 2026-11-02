"""Leakage-checked affine spectral calibration on explicitly partitioned targets.

This module consumes already extracted, normalized linear RGB and reference
reflectance. It neither decodes DNG nor authenticates instrument provenance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json

import numpy as np

from .intervals import median_interval
from .spectra import BANDS_NM, apply_tm, mean_absolute_error
from .validation import evaluate_csv

SCHEMA = "myco-fusion.calibration/0.2"
ALPHAS = (0.0, 1e-6, 1e-4, 1e-2, 1.0)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _parse(data):
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA:
        raise ValueError(f"schema_version must be {SCHEMA}")
    observations = data.get("observations")
    if not isinstance(observations, list) or not 1 <= len(observations) <= 5000:
        raise ValueError("observations must contain 1 to 5000 records")
    if data.get("rgb_encoding") not in {"linear_normalized", "synthetic_linear"}:
        raise ValueError("rgb_encoding must be linear_normalized or synthetic_linear; sRGB/ProRAW rendering is not implicitly sensor-linear")
    if data.get("data_kind") == "synthetic" and data["rgb_encoding"] != "synthetic_linear":
        raise ValueError("Synthetic data must declare synthetic_linear RGB")
    if data["rgb_encoding"] == "synthetic_linear" and data.get("data_kind") != "synthetic":
        raise ValueError("synthetic_linear RGB must be labeled data_kind=synthetic")
    for field in ("device", "pipeline_id"):
        if not isinstance(data.get(field), str) or not data[field].strip() or len(data[field]) > 200:
            raise ValueError(f"{field} must identify the fixed acquisition/processing configuration in at most 200 characters")
    if "bands_nm" in data and data["bands_nm"] != BANDS_NM.tolist():
        raise ValueError("bands_nm must be 420 through 680 nm in 10 nm steps")
    rows, ids, specimens, sessions = [], set(), defaultdict(set), defaultdict(set)
    for i, row in enumerate(observations):
        if not isinstance(row, dict):
            raise ValueError(f"Observation {i} must be an object")
        item = dict(row)
        for field in ("specimen_id", "session_id", "regime", "observation_id", "split"):
            if not isinstance(item.get(field), str) or not item[field].strip() or len(item[field]) > 200:
                raise ValueError(f"Observation {i}: {field} must be nonempty text of at most 200 characters")
            item[field] = item[field].strip()
        if item["split"] not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if item["observation_id"] in ids:
            raise ValueError("Duplicate observation_id")
        ids.add(item["observation_id"])
        for field, size in (("rgb", 3), ("reference", 27)):
            value = np.asarray(item.get(field), dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
                raise ValueError(f"Observation {i}: {field} must be {size} finite values in [0,1]")
            if field == "reference" and not np.any(value):
                raise ValueError(f"Observation {i}: zero reference spectrum has undefined SAM")
            item[field] = value
        specimens[item["specimen_id"]].add(item["split"])
        sessions[item["session_id"]].add(item["split"])
        rows.append(item)
    leaking = [k for k, v in specimens.items() if len(v) > 1]
    if leaking:
        raise ValueError("Specimen leakage across splits: " + ", ".join(sorted(leaking)[:10]))
    session_overlap = sorted(k for k, v in sessions.items() if len(v) > 1)
    holdout = data.get("require_session_holdout", True)
    if not isinstance(holdout, bool):
        raise ValueError("require_session_holdout must be true or false")
    if holdout and session_overlap:
        raise ValueError("Session leakage across splits: " + ", ".join(session_overlap[:10]))
    for split in ("train", "test"):
        if not any(r["split"] == split for r in rows):
            raise ValueError(f"At least one {split} observation is required")
    return rows, session_overlap


def _weights(rows):
    counts = Counter(r["specimen_id"] for r in rows)
    # Total weight is one per physical specimen, independent of repeat count.
    return np.array([1.0 / counts[r["specimen_id"]] for r in rows])


def _fit(rows, alpha):
    x = np.array([r["rgb"] for r in rows]); y = np.array([r["reference"] for r in rows])
    design = np.column_stack((x, np.ones(len(rows))))
    weights = _weights(rows)
    weighted = design * np.sqrt(weights[:, None])
    rank = int(np.linalg.matrix_rank(weighted))
    if rank < 4:
        raise ValueError("Calibration RGB design is rank deficient; acquire at least four affinely independent calibration responses")
    penalty = np.diag([np.sqrt(alpha)] * 3 + [0.0])
    W = np.linalg.lstsq(np.vstack((weighted, penalty)), np.vstack((y * np.sqrt(weights[:, None]), np.zeros((4, 27)))), rcond=None)[0]
    return W, rank, float(np.linalg.cond(weighted))


def _specimen_mae(rows, predictions):
    by_specimen = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        by_specimen[row["specimen_id"]].append(mean_absolute_error(row["reference"], prediction))
    return {specimen: float(np.mean(values)) for specimen, values in by_specimen.items()}


def _mean_specimen_mae(rows, predictions):
    return float(np.mean(list(_specimen_mae(rows, predictions).values())))


def fit_calibration(data):
    """Tune on validation only; preserve test data exclusively for evaluation."""
    rows, overlaps = _parse(data)
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]
    if len({r["specimen_id"] for r in train}) < 4:
        raise ValueError("At least four distinct training specimens are required")
    candidates = []
    selected_alpha = 1e-4
    if validation:
        for alpha in ALPHAS:
            W, _, _ = _fit(train, alpha)
            predictions = apply_tm([r["rgb"] for r in validation], W)
            candidates.append({"alpha": alpha, "validation_mean_specimen_mae": _mean_specimen_mae(validation, predictions)})
        selected_alpha = min(candidates, key=lambda c: (c["validation_mean_specimen_mae"], c["alpha"]))["alpha"]
    W, rank, condition = _fit(train, selected_alpha)
    predictions = apply_tm([r["rgb"] for r in test], W)
    train_weights = _weights(train)
    baseline_spectrum = np.average([r["reference"] for r in train], axis=0, weights=train_weights)
    baseline = np.broadcast_to(baseline_spectrum, predictions.shape)
    # For an MAE objective, a weighted median is a stronger constant comparator
    # than the training mean. Preserve the mean baseline for comparability.
    targets = np.array([r["reference"] for r in train])
    median_spectrum = []
    for band in range(27):
        order = np.argsort(targets[:, band], kind="stable")
        halfway = train_weights.sum() / 2
        index = np.searchsorted(np.cumsum(train_weights[order]), halfway, side="left")
        median_spectrum.append(float(targets[order[index], band]))
    median_baseline = np.broadcast_to(median_spectrum, predictions.shape)
    data_kind = "synthetic" if data.get("data_kind") == "synthetic" else "user_supplied_unverified"
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["specimen_id", "regime", "repeat", "session_id", "wavelength_nm", "reference", "estimate", "data_kind", "data_origin"])
    for row, prediction in zip(test, predictions):
        for band, ref, estimate in zip(BANDS_NM, row["reference"], prediction):
            writer.writerow([row["specimen_id"], row["regime"], row["observation_id"], row["session_id"], int(band), float(ref), float(estimate), data_kind, "held_out_calibration_prediction"])
    prediction_csv = stream.getvalue()
    evaluation = evaluate_csv(prediction_csv, uncertainty=data.get("uncertainty", "auto"))
    # Model minus stronger baseline, specimen by specimen; negative means the model is better.
    model_errors = _specimen_mae(test, predictions)
    baseline_errors = _specimen_mae(test, median_baseline)
    differences = [model_errors[s] - baseline_errors[s] for s in model_errors]
    x_train = np.array([r["rgb"] for r in train]); x_test = np.array([r["rgb"] for r in test])
    outside = ((x_test < x_train.min(0)) | (x_test > x_train.max(0))).any(axis=1)
    model = {
        "schema_version": "myco-fusion.affine-model/0.2", "model_kind": "affine_ridge_linear_rgb_to_estimated_reflectance",
        "bands_nm": BANDS_NM.tolist(), "weights": W.tolist(), "alpha": selected_alpha,
        "rgb_encoding": data["rgb_encoding"], "data_kind": data_kind,
        "device": data["device"], "pipeline_id": data["pipeline_id"],
        "training_specimens": sorted({r["specimen_id"] for r in train}),
        "training_sessions": sorted({r["session_id"] for r in train}),
        "validation_specimens": sorted({r["specimen_id"] for r in validation}),
        "rgb_training_min": x_train.min(0).tolist(), "rgb_training_max": x_train.max(0).tolist(),
        "training_weighting": "equal physical-specimen weight; equal observations within specimen",
    }
    model["model_sha256"] = canonical_digest(model)
    notes = [
        "RGB and reference spectra are supplied by the caller. Instrument calibration and linearity are not independently authenticated.",
        "All observations of a physical specimen must stay within one split; overlapping identifiers are rejected.",
        "Validation selects alpha by mean specimen MAE. Test references never select alpha or fit the model; the final model uses training observations only.",
        "The comparison baseline predicts the specimen-balanced training mean spectrum for every held-out observation.",
        "A second constant baseline uses the specimen-weighted per-band training median, which minimizes weighted training MAE. Both baselines are evaluated without test-data fitting.",
        "RGB range flags are diagnostics, not a calibrated out-of-distribution detector.",
        "Predictions are retained without clipping. This is spectral-estimation validation, not a biological classifier.",
    ]
    if not validation:
        notes.append("No validation split supplied: alpha=0.0001 was fixed in advance rather than selected on test data.")
    if overlaps:
        notes.append("Session holdout explicitly disabled. Shared sessions: " + ", ".join(overlaps))
    return {
        "kind": "calibration_evaluation", "schema_version": "0.2", "data_kind": data_kind,
        "input_sha256": canonical_digest(data), "model": model, "evaluation": evaluation,
        "counts": {split: {"observations": sum(r["split"] == split for r in rows), "specimens": len({r["specimen_id"] for r in rows if r["split"] == split}), "sessions": len({r["session_id"] for r in rows if r["split"] == split})} for split in ("train", "validation", "test")},
        "diagnostics": {"design_rank": rank, "design_condition_number": condition, "test_rgb_outside_training_range": int(outside.sum()), "test_estimated_bands_outside_0_1": int(((predictions < 0) | (predictions > 1)).sum()), "session_overlap": overlaps},
        "tuning": candidates,
        "test_mean_specimen_mae": _mean_specimen_mae(test, predictions),
        "baseline_mean_specimen_mae": _mean_specimen_mae(test, baseline),
        "median_baseline_mean_specimen_mae": _mean_specimen_mae(test, median_baseline),
        "model_vs_median_baseline": {
            "median_difference": float(np.median(differences)), "interval": median_interval(differences),
            "note": "Model minus training-median baseline, per test specimen. Negative favors the model. The interval treats test specimens as independent.",
        },
        "prediction_csv": prediction_csv, "notes": notes,
    }


def predict_calibration(model, capture):
    """Apply an exported model without refitting or treating predictions as truth.

    The hash detects accidental changes; it is not a signature or proof that
    acquisition metadata, reference measurements, or training labels are true.
    """
    if not isinstance(model, dict) or model.get("schema_version") != "myco-fusion.affine-model/0.2":
        raise ValueError("Supply an exported myco-fusion.affine-model/0.2 model")
    unsigned = {k: v for k, v in model.items() if k != "model_sha256"}
    if model.get("model_sha256") != canonical_digest(unsigned):
        raise ValueError("Model SHA-256 mismatch; the exported model was changed")
    if model.get("bands_nm") != BANDS_NM.tolist():
        raise ValueError("Model wavelength grid does not match 420–680 nm / 10 nm")
    if not isinstance(capture, dict):
        raise ValueError("capture must be an object with device, pipeline_id, rgb_encoding and observations")
    for field in ("device", "pipeline_id", "rgb_encoding"):
        if capture.get(field) != model.get(field):
            raise ValueError(f"Capture {field} must exactly match the calibration model; do not silently transfer across pipelines")
    synthetic = model.get("data_kind") == "synthetic"
    if synthetic != (capture.get("data_kind") == "synthetic"):
        raise ValueError("Synthetic calibration models and captures cannot be mixed with user-supplied physical data")
    weights = np.asarray(model.get("weights"), dtype=float)
    if weights.shape != (4, 27) or not np.isfinite(weights).all():
        raise ValueError("Model weights must be finite with shape (4,27)")
    low = np.asarray(model.get("rgb_training_min"), dtype=float)
    high = np.asarray(model.get("rgb_training_max"), dtype=float)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or (low > high).any():
        raise ValueError("Model training bounds are invalid")
    observations = capture.get("observations")
    if not isinstance(observations, list) or not 1 <= len(observations) <= 5000:
        raise ValueError("capture observations must contain 1 to 5000 records")
    parsed, seen = [], set()
    for row in observations:
        if not isinstance(row, dict):
            raise ValueError("Each capture observation must be an object")
        identifier = row.get("observation_id")
        if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 200 or identifier in seen:
            raise ValueError("Capture observation_id values must be unique, nonempty text of at most 200 characters")
        seen.add(identifier)
        rgb = np.asarray(row.get("rgb"), dtype=float)
        if rgb.shape != (3,) or not np.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
            raise ValueError("Each capture rgb must contain three finite linear values in [0,1]")
        estimate = apply_tm(rgb[None, :], weights)[0]
        if not np.isfinite(estimate).all():
            raise ValueError("Model produced nonfinite predictions; inspect its coefficients and RGB inputs")
        parsed.append({"observation_id": identifier, "estimated_reflectance": estimate.tolist(), "rgb_outside_training_range": bool(((rgb < low) | (rgb > high)).any()), "estimated_bands_outside_0_1": int(((estimate < 0) | (estimate > 1)).sum())})
    return {"kind": "calibrated_predictions", "schema_version": "0.2", "data_kind": "synthetic" if synthetic else "user_supplied_unverified", "evidence_type": "inferred", "model_sha256": model["model_sha256"], "capture_sha256": canonical_digest(capture), "bands_nm": BANDS_NM.tolist(), "predictions": parsed, "notes": ["Predictions are unclipped estimates, not independent spectral measurements.", "A matching digest verifies content integrity, not authenticity or physical calibration.", "Matching device/pipeline labels cannot establish actual acquisition equivalence; verify the capture log.", "Range flags are diagnostics; their absence does not establish in-distribution performance."]}


def calibration_example():
    """Measured reflectance with simulated RGB; see myco_fusion.examples."""
    from .examples import calibration_dataset
    return calibration_dataset()
