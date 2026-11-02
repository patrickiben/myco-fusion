"""What should the phone test find? Simulate it with known truth, so results can be read.

Uses the bundled natural-object spectra with the illustrative camera and lights. The
control restricts spectra to three dimensions, so any count above 3 is a false positive.
Writes results/phone_power.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from myco_fusion.bench import illustrative_lights
from myco_fusion.data import natural_objects
from myco_fusion.repeatability import independent_measurements
from myco_fusion.synthetic import _camera

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261101
REPLICATES = 30


def simulated_rows(spectra, count, lights, camera, rng, additive=0.005, gain=0.02, screen_factor=3.0, sessions=3, repeats=3, session_object_gain=0.0):
    picks = rng.choice(len(spectra), count, replace=False)
    rows = []
    for session in range(sessions):
        session_gain = 1 + rng.normal(0, 0.015)
        for name, light in lights.items():
            noise = additive * (screen_factor if name != "flash" else 1.0)
            for index in picks:
                clean = (spectra[index] * light) @ camera.T
                placement = 1 + rng.normal(0, session_object_gain)
                for repeat in range(repeats):
                    values = clean * session_gain * placement * (1 + rng.normal(0, gain)) + rng.normal(0, noise, 3)
                    rows.append({"object": str(index), "session": str(session), "light": name, "repeat": str(repeat), "values": list(values)})
    return rows


def main():
    spectra = natural_objects()["spectra"]
    centered = spectra - spectra.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    three_dimensional = spectra.mean(axis=0) + centered @ vt[:3].T @ vt[:3]
    lights_curves = illustrative_lights()
    lights = {"flash": lights_curves["flash"], "red": lights_curves["screen"][2], "green": lights_curves["screen"][1], "blue": lights_curves["screen"][0]}
    camera = _camera()
    rng = np.random.default_rng(SEED)
    table = []
    for count in (10, 20, 30):
        row = {"n_objects": count}
        for label, data, group, placement in (("flash_only", spectra, ["flash"], 0.0), ("all_lights", spectra, list(lights), 0.0),
                                              ("control_three_dimensional", three_dimensional, list(lights), 0.0),
                                              ("control_with_placement_drift", three_dimensional, list(lights), 0.03)):
            counts = [independent_measurements(simulated_rows(data, count, lights, camera, rng, session_object_gain=placement), group, draws=150)["independent"] for _ in range(REPLICATES)]
            row[label] = {"min": int(np.min(counts)), "median": float(np.median(counts)), "max": int(np.max(counts)), "share_above_3": float(np.mean(np.array(counts) > 3))}
        table.append(row)
        print(row, flush=True)
    out = ROOT / "results" / "phone_power.json"
    out.write_text(json.dumps({"seed": SEED, "replicates": REPLICATES, "design": "3 sessions, 3 repeats, flash plus 3 screen colors; additive noise 0.005, gain error 2%, screen noise x3; lights exposure-matched to flash; one control adds 3% object-by-session drift", "rows": table}, indent=1) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
