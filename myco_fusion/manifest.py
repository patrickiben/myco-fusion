"""Draft capture records. Creating a record never asserts acquisition occurred."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone


def build_manifest(data):
    if not isinstance(data, dict):
        raise ValueError("manifest input must be an object")
    required = ("specimen_id", "session_id", "device", "regime", "reference_id")
    for field in required:
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f"{field} is required")
        if len(data[field]) > 200:
            raise ValueError(f"{field} must be 200 characters or fewer")
    if data["regime"] not in {"R1", "R2", "R3", "R4"}:
        raise ValueError("regime must be R1, R2, R3, or R4")
    notes = data.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 5000:
        raise ValueError("notes must be text of at most 5000 characters")
    record = {
        "schema_version": "myco-fusion.capture/0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "draft_capture_record",
        **{field: data[field].strip() for field in required},
        "notes": notes,
        "bands_nm": list(range(420, 681, 10)),
        "measurement_kind": "not_acquired",
        "calibration_id": None,
        "sensor_response_id": None,
        "illuminant_spectrum_id": None,
        "exposure_settings": None,
        "linearization_pipeline": None,
        "contact_instrument_id": None,
        "contact_spot_registration": None,
        "pose_convention": "+x right, +y down, +z forward; world_to_camera after reconstruction refinement",
        "coordinate_units": None,
        "split_assignment": None,
        "taxonomic_evidence": None,
        "files": [],
        "checks": {
            "dark_white_references": False,
            "exposure_linearization_recorded": False,
            "contact_reference_same_spot": False,
            "camera_intrinsics_recorded": False,
            "spectral_pose_in_final_world": False,
            "held_out_by_specimen": False,
            "rights_recorded_per_file": False,
        },
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    record["record_sha256"] = hashlib.sha256(canonical).hexdigest()
    record["digest_scope"] = "UTF-8 canonical JSON with sorted keys, compact separators and ensure_ascii=False, excluding record_sha256 and digest_scope"
    return {"manifest": record, "warnings": [
        "This draft does not confirm that any specimen, image, spectrum, or pose was acquired.",
        "Complete acquisition metadata and file SHA-256 digests before using this record in an experiment.",
    ]}
