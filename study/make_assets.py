"""Write the 27-band natural-object table bundled with the package (CC BY 4.0 source)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .pipeline import METRIC_NM, resample
from .sources import load_natural_objects

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "myco_fusion" / "data" / "natural_objects_27band.csv"


def main():
    data = load_natural_objects()
    spectra = resample(data["wavelengths"], np.clip(data["spectra"], 0, 1), METRIC_NM)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["sample_id", "domain", "object", "color", "group", *[str(int(w)) for w in METRIC_NM]])
        for i, sample in enumerate(data["ids"]):
            writer.writerow([sample, data["domain"][i], data["object"][i], data["color"][i], data["group"][i],
                             *[f"{value:.5f}" for value in spectra[i]]])
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes, {len(data['ids'])} samples)")


if __name__ == "__main__":
    main()
