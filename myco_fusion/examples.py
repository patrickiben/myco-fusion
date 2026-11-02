"""Worked examples built from bundled measured reflectance and a simulated camera.

Reference spectra are real measurements. RGB values are simulated, so every example
is labeled synthetic and none of them validates a phone.
"""
from __future__ import annotations

import csv
import io

import numpy as np

from .bench import group_split, illustrative_lights, _ridge
from .calibration import SCHEMA
from .data import natural_objects
from .spectra import BANDS_NM
from .synthetic import _camera

SEED = 20261101
SESSIONS = 6


def _simulate(spectra, light, session_gain, rng, noise=0.005):
    clean = (spectra * light) @ _camera().T
    return clean * session_gain + rng.normal(0, noise, clean.shape)


def calibration_dataset():
    """Train, validation and test specimens with separate sessions, under two lights."""
    data = natural_objects()
    rng = np.random.default_rng(SEED)
    lights = illustrative_lights()
    regimes = {"R1": lights["flash"], "R3": lights["flash"] * (1 + 0.15 * (BANDS_NM - 550) / 130)}
    folds = group_split(data["group"], 3, SEED)
    plan = (("train", 0, 60, 4), ("validation", 1, 24, 4), ("test", 2, 36, SESSIONS))
    observations = []
    for split, fold, count, sessions in plan:
        members = np.flatnonzero(folds == fold)[:count]
        for position, index in enumerate(members):
            session = position % sessions
            gain = 1 + np.random.default_rng([SEED, fold, session]).normal(0, 0.02)
            specimen = data["ids"][index]
            for regime, light in regimes.items():
                if split != "test" and regime != "R1":
                    continue
                for repeat in (1, 2):
                    rgb = _simulate(data["spectra"][index][None, :], light, gain, rng)[0]
                    observations.append({
                        "observation_id": f"{split}-{specimen}-{regime}-{repeat}", "specimen_id": specimen,
                        "session_id": f"{split}-session-{session + 1}", "split": split, "regime": regime,
                        "rgb": np.clip(rgb, 0, 1).round(6).tolist(), "reference": data["spectra"][index].round(5).tolist(),
                    })
    return {"schema_version": SCHEMA, "data_kind": "synthetic", "rgb_encoding": "synthetic_linear",
            "device": "simulated camera; measured reflectance; not phone data", "pipeline_id": "bundled-example-v1",
            "require_session_holdout": True, "uncertainty": "auto", "observations": observations}


def validation_csv():
    """Held-out estimates for paired-spectrum validation, in the app's CSV format."""
    data = natural_objects()
    rng = np.random.default_rng(SEED + 1)
    lights = illustrative_lights()
    spectra = data["spectra"]
    folds = group_split(data["group"], 4, SEED)
    train, test = folds != 0, np.flatnonzero(folds == 0)[:48]
    model = _ridge(_simulate(spectra[train], lights["flash"], 1.0, rng), spectra[train])
    shifted = lights["flash"] * (1 + 0.15 * (BANDS_NM - 550) / 130)
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["specimen_id", "regime", "repeat", "session_id", "wavelength_nm", "reference", "estimate", "data_kind", "data_origin"])
    for position, index in enumerate(test):
        session = position % SESSIONS
        gain = 1 + np.random.default_rng([SEED, session]).normal(0, 0.03)
        for regime, light in (("R1", lights["flash"]), ("R3", shifted)):
            for repeat in (1, 2):
                estimate = model(_simulate(spectra[index][None, :], light, gain, rng))[0]
                for band, reference, value in zip(BANDS_NM, spectra[index], estimate):
                    writer.writerow([data["ids"][index], regime, repeat, f"session-{session + 1}", int(band),
                                     f"{reference:.5f}", f"{value:.5f}", "synthetic", "simulated_camera_on_measured_reflectance"])
    return output.getvalue()


def repeatability_csv():
    """A simulated phone test with known variance, so the analysis can be checked by eye."""
    data = natural_objects()
    rng = np.random.default_rng(SEED + 2)
    lights = illustrative_lights()
    camera = _camera()
    picks = np.linspace(0, len(data["ids"]) - 1, 10).astype(int)
    used = {"flash": lights["flash"], "screen-red": lights["screen"][2], "screen-green": lights["screen"][1], "screen-blue": lights["screen"][0]}
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["object_id", "session_id", "light", "repeat", "r", "g", "b", "data_kind"])
    for session in range(1, 4):
        session_gain = 1 + rng.normal(0, 0.015)
        for name, light in used.items():
            for index in picks:
                object_gain = 1 + rng.normal(0, 0.01)
                clean = (data["spectra"][index] * light) @ camera.T
                # Screen photos are dimmer, so exposure is longer and noise is larger.
                noise = 0.009 if name.startswith("screen") else 0.003
                for repeat in (1, 2, 3):
                    values = clean * session_gain * object_gain + rng.normal(0, noise, 3)
                    writer.writerow([f"object-{index:03d}", f"session-{session}", name, repeat, *[f"{max(v, 0):.5f}" for v in values], "synthetic"])
    return output.getvalue()
