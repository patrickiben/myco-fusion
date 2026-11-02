"""Evaluate paired spectra with specimens, not repeated captures, as units."""
from __future__ import annotations

import csv
import hashlib
import io
from collections import defaultdict

import numpy as np

from .intervals import median_interval, share_interval
from .spectra import BANDS_NM, mean_absolute_error, spectral_angle

REQUIRED_COLUMNS = {"specimen_id", "regime", "repeat", "wavelength_nm", "reference", "estimate"}
DRAFT_TARGETS = {"sam_deg": 10.0, "mae": 0.05}
UNCERTAINTY_METHODS = {"auto", "specimen", "session_cluster"}


def evaluate_csv(text: str, uncertainty: str = "auto") -> dict:
    """Evaluate complete 420–680 nm spectra supplied in long CSV format.

    Each (specimen_id, regime, repeat, optional session_id, optional spot_id)
    is one 27-band observation. Supplied session/spot columns must be complete.
    Repeat/spot errors are averaged within specimen and regime, then the median
    operates on physical specimens.

    Intervals are exact order-statistic intervals for a median, computed over
    independent units. ``specimen`` treats specimens as those units. ``session_cluster``
    uses one median per session and requires specimens nested within exactly one
    session; crossed designs are rejected in that mode. ``auto`` uses sessions
    when they are supplied, nested and at least two per regime, otherwise specimens.
    The reported level is the true one and can be below 95% when units are few.

    Draft targets (SAM <= 10 degrees, MAE <= 0.05) are unvalidated placeholders.
    Each regime reports how many specimens meet both, because a median can sit
    within the targets while many specimens do not.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("CSV text is empty")
    if not isinstance(uncertainty, str) or uncertainty not in UNCERTAINTY_METHODS:
        raise ValueError("uncertainty must be 'auto', 'specimen' or 'session_cluster'")
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
        raise ValueError("CSV requires columns: " + ", ".join(sorted(REQUIRED_COLUMNS)))
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("CSV contains duplicate column names")
    metadata_columns = [name for name in ("session_id", "spot_id") if name in reader.fieldnames]
    has_sessions = "session_id" in metadata_columns
    if uncertainty == "session_cluster" and not has_sessions:
        raise ValueError("session_cluster uncertainty requires a complete session_id column")
    observations = defaultdict(dict)
    out_of_range = 0
    source_labels = set()
    all_declared_synthetic = True
    for line_number, row in enumerate(reader, start=2):
        if None in row or any(row.get(name) is None for name in REQUIRED_COLUMNS | set(metadata_columns)):
            raise ValueError(f"Malformed CSV row at line {line_number}")
        identity = tuple(row[name].strip() for name in ("specimen_id", "regime", "repeat"))
        if not all(identity):
            raise ValueError(f"specimen_id, regime and repeat must be nonempty (line {line_number})")
        metadata = {name: row[name].strip() for name in metadata_columns}
        for name, value in metadata.items():
            if not value:
                raise ValueError(f"Optional {name} column must be complete and nonempty on every band row (line {line_number})")
        key = (*identity, metadata.get("session_id", ""), metadata.get("spot_id", ""))
        try:
            wave, reference, estimate = (
                float(row[name]) for name in ("wavelength_nm", "reference", "estimate")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric spectrum value at line {line_number}") from exc
        if not np.isfinite([wave, reference, estimate]).all():
            raise ValueError(f"Nonfinite spectrum value at line {line_number}")
        if wave not in BANDS_NM:
            raise ValueError(f"Wavelength must be 420:10:680 nm (line {line_number})")
        if not 0 <= reference <= 1:
            raise ValueError(f"Reference reflectance outside [0, 1] (line {line_number})")
        if int(wave) in observations[key]:
            raise ValueError(f"Duplicate wavelength for observation {key} (line {line_number})")
        observations[key][int(wave)] = (reference, estimate)
        out_of_range += int(not 0 <= estimate <= 1)
        if (row.get("data_origin") or "").strip():
            source_labels.add(row["data_origin"].strip())
        if (row.get("data_kind") or "").strip() != "synthetic":
            all_declared_synthetic = False
    if not observations:
        raise ValueError("CSV has no observations")
    rows = []
    specimen_metrics = defaultdict(list)
    specimen_sessions = defaultdict(set)
    specimen_regime_sessions = defaultdict(set)
    session_specimen_metrics = defaultdict(list)
    regime_observation_counts = defaultdict(int)
    session_observation_counts = defaultdict(int)
    for (specimen, regime, repeat, session, spot), bands in sorted(observations.items()):
        missing = sorted(set(BANDS_NM.tolist()) - set(bands))
        if missing:
            raise ValueError(
                f"Incomplete spectrum for {(specimen, regime, repeat, session, spot)}; "
                f"missing {missing}. Session/spot metadata must be consistent across all bands."
            )
        reference, estimate = np.asarray([bands[int(w)] for w in BANDS_NM]).T
        try:
            angle = spectral_angle(reference, estimate)
        except ValueError as exc:
            raise ValueError(f"Observation {(specimen, regime, repeat)}: {exc}") from exc
        error = mean_absolute_error(reference, estimate)
        observation = {"specimen_id": specimen, "regime": regime, "repeat": repeat,
                       "sam_deg": angle, "mae": error,
                       "reference": reference.tolist(), "estimate": estimate.tolist()}
        if has_sessions:
            observation["session_id"] = session
            specimen_sessions[specimen].add(session)
            specimen_regime_sessions[(regime, specimen)].add(session)
            session_specimen_metrics[(regime, session, specimen)].append([angle, error])
            session_observation_counts[(regime, session)] += 1
        if "spot_id" in metadata_columns:
            observation["spot_id"] = spot
        rows.append(observation)
        specimen_metrics[(regime, specimen)].append([angle, error])
        regime_observation_counts[regime] += 1
    crossed = sorted(specimen for specimen, sessions in specimen_sessions.items() if len(sessions) > 1)
    if uncertainty == "session_cluster" and crossed:
        raise ValueError(
            "session_cluster requires each physical specimen to belong to exactly one session "
            "across all regimes. Crossed specimens were found (including "
            + ", ".join(crossed[:5])
            + "). Use specimen mode as a descriptive analysis with session correlation unmodeled, "
            "or the repeatability analysis, which is built for crossed designs."
        )
    by_regime = defaultdict(list)
    specimen_summaries = []
    for (regime, specimen), values in sorted(specimen_metrics.items()):
        mean = np.mean(values, axis=0)
        by_regime[regime].append(mean)
        sessions = sorted(specimen_regime_sessions[(regime, specimen)])
        specimen_summaries.append({
            "specimen_id": specimen, "regime": regime, "n_observations": len(values),
            "session_ids": sessions, "n_sessions": len(sessions) if has_sessions else None,
            "mean_sam_deg": float(mean[0]), "mean_mae": float(mean[1]),
        })
    by_session = defaultdict(list)
    for (regime, session, specimen), values in sorted(session_specimen_metrics.items()):
        by_session[(regime, session)].append(np.mean(values, axis=0))
    session_summaries = []
    for (regime, session), values in sorted(by_session.items()):
        median = np.median(values, axis=0)
        session_summaries.append({
            "session_id": session, "regime": regime, "n_specimens": len(values),
            "n_observations": session_observation_counts[(regime, session)],
            "median_sam_deg": float(median[0]), "median_mae": float(median[1]),
        })
    method = uncertainty
    if method == "auto":
        enough = has_sessions and not crossed and all(
            sum(1 for r, _ in by_session if r == regime) >= 2 for regime in by_regime)
        method = "session_cluster" if enough else "specimen"
    regimes = []
    interval_notes = []
    for regime, values in sorted(by_regime.items()):
        values = np.asarray(values)
        angle, error = np.median(values, axis=0)
        sessions = sorted(session for r, session in by_session if r == regime)
        if method == "session_cluster":
            units = np.asarray([np.median(by_session[(regime, session)], axis=0) for session in sessions])
            unit_name, missing_reason = "session", "At least two observed sessions are required for a session-based interval."
        else:
            units = values
            unit_name, missing_reason = "specimen", "At least two specimens are required for an interval."
        sam_interval = median_interval(units[:, 0])
        mae_interval = median_interval(units[:, 1])
        level = sam_interval["level"] if sam_interval else None
        note = None
        if sam_interval is None:
            note = missing_reason
            interval_notes.append(f"{regime}: {missing_reason}")
        elif not sam_interval["reaches_target"]:
            note = (f"Only {len(units)} {unit_name}s: the full range is a {100 * level:.1f}% interval. "
                    f"No {unit_name}-based interval can reach 95% without assuming a distribution; that takes at least 6.")
            interval_notes.append(f"{regime}: {note}")
        within = (values[:, 0] <= DRAFT_TARGETS["sam_deg"]) & (values[:, 1] <= DRAFT_TARGETS["mae"])
        regimes.append({
            "regime": regime, "n_specimens": len(values),
            "n_observations": regime_observation_counts[regime],
            "n_sessions": len(sessions) if has_sessions else None,
            "median_sam_deg": float(angle), "sam_interval": sam_interval,
            "median_mae": float(error), "mae_interval": mae_interval,
            "interval_level": level, "interval_units": len(units), "interval_unit": unit_name,
            "interval_note": note,
            "worst_specimen_mae": float(np.max(values[:, 1])),
            "p90_specimen_mae": float(np.percentile(values[:, 1], 90)),
            "n_within_targets": int(within.sum()),
            "share_within_targets": float(within.mean()),
            "share_interval": share_interval(int(within.sum()), len(values)) if method == "specimen" else None,
            "median_within_targets": bool(angle <= DRAFT_TARGETS["sam_deg"] and error <= DRAFT_TARGETS["mae"]),
        })
    # Pair specimen-level means before taking medians; never subtract unrelated
    # regime medians or inflate n by pairing individual repeated captures.
    means = {(r["regime"], r["specimen_id"]): r for r in specimen_summaries}
    r1_ids = {s for regime, s in means if regime == "R1"}
    r3_ids = {s for regime, s in means if regime == "R3"}
    pairs = []
    for specimen in sorted(r1_ids & r3_ids):
        first, third = means[("R1", specimen)], means[("R3", specimen)]
        pairs.append({
            "specimen_id": specimen,
            "r1_mean_mae": first["mean_mae"], "r3_mean_mae": third["mean_mae"],
            "delta_mae_r3_minus_r1": third["mean_mae"] - first["mean_mae"],
            "r1_mean_sam_deg": first["mean_sam_deg"], "r3_mean_sam_deg": third["mean_sam_deg"],
            "delta_sam_deg_r3_minus_r1": third["mean_sam_deg"] - first["mean_sam_deg"],
        })
    paired = {
        "n_specimens": len(pairs), "rows": pairs,
        "r1_only_specimens": sorted(r1_ids - r3_ids), "r3_only_specimens": sorted(r3_ids - r1_ids),
        "median_delta_mae": float(np.median([r["delta_mae_r3_minus_r1"] for r in pairs])) if pairs else None,
        "median_delta_sam_deg": float(np.median([r["delta_sam_deg_r3_minus_r1"] for r in pairs])) if pairs else None,
        "delta_mae_interval": median_interval([r["delta_mae_r3_minus_r1"] for r in pairs]) if pairs else None,
        "delta_sam_interval": median_interval([r["delta_sam_deg_r3_minus_r1"] for r in pairs]) if pairs else None,
        "note": "R3 minus R1 deltas pair physical-specimen mean errors. Their interval treats specimens as independent and does not model shared session effects.",
    }
    if method == "specimen":
        interval_notes.append(
            "Exact order-statistic interval over physical-specimen means, assuming specimens are independent. "
            + ("Session identifiers and descriptive session summaries are available." if has_sessions else "No session identifiers were supplied.")
        )
        if crossed:
            interval_notes.append("Some physical specimens occur in multiple sessions. This interval ignores shared session effects and can be too narrow; see the repeatability analysis.")
        elif has_sessions:
            interval_notes.append("Sessions were supplied, but this interval ignores them. It can be too narrow when session effects are large.")
    else:
        interval_notes.append(
            "Exact order-statistic interval over one median per session, assuming independent sessions and specimens "
            "nested within one session. It describes the typical session's median. Site clustering is unmodeled."
        )
    notes = [
        "Input spectra are user-supplied. This report does not establish instrument provenance or independent ground truth.",
        "Repeat/spot metrics are averaged within each specimen/regime; point medians give each physical specimen equal weight.",
        "Draft targets (SAM ≤ 10°, MAE ≤ 0.05) are unvalidated placeholders, not an acceptance rule or a field go/no-go decision.",
        "Calibration/evaluation separation must be established in the study design; this CSV contains no training provenance.",
        "Predictions are retained without clipping; zero-norm spectra are rejected because SAM is undefined.",
    ]
    notes.extend(interval_notes)
    if out_of_range:
        notes.append(f"{out_of_range} estimated band values fall outside [0, 1] and remain in the error metrics.")
    if source_labels:
        notes.append("User-supplied data_origin labels: " + ", ".join(sorted(source_labels)) + ". These labels have not been independently verified.")
    return {
        "kind": "paired_spectra", "bands_nm": BANDS_NM.tolist(),
        "data_kind": "synthetic" if all_declared_synthetic else "user_supplied_unverified",
        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "regimes": regimes, "rows": rows, "notes": notes,
        "uncertainty_method": method, "uncertainty_requested": uncertainty, "interval_notes": interval_notes,
        "specimen_summaries": specimen_summaries, "session_summaries": session_summaries,
        "paired_r1_r3": paired,
        "counts": {"n_observations": len(rows), "n_specimens": len({r["specimen_id"] for r in rows}),
                   "n_sessions": len({r["session_id"] for r in rows}) if has_sessions else None,
                   "n_specimen_regime_units": len(specimen_summaries)},
        "source_labels": sorted(source_labels),
        "draft_targets": DRAFT_TARGETS,
    }
