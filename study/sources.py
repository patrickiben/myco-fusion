"""Fetch, verify and load the measured datasets used by the study.

Natural-object spectra (Gutiérrez et al. 2025) are CC BY 4.0. The SpectroFood CSV
page states no license, so it is downloaded on demand and never redistributed here.
Color chart, camera, illuminant and display curves come from colour-science.
"""
from __future__ import annotations

import csv
import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FILES = {
    "Database_Plants.csv": ("https://ndownloader.figshare.com/files/51761975", "bdadfd60623f60983f85fc23bc01e8c7901266ba2b7a6df5782ec7711da963d3"),
    "Database_Animals.csv": ("https://ndownloader.figshare.com/files/51761978", "aabb1d4eea382f23758766105aa6f9bf89f46e6c59d25cb7fa27ad79c820ae40"),
    "Database_Stones.csv": ("https://ndownloader.figshare.com/files/51761981", "e7a369164cf02d92d547a028af121ed2e1959f6c4ded17751716d6117e9a7691"),
    "SpectroFood_dataset.csv": ("https://zenodo.org/api/records/8362947/files/SpectroFood_dataset.csv/content", "7e2f2eb0b44a1e1cfcd580ca2bc13473d22cba0529b175904d43983e31fb707d"),
}
# Measured phone sensitivities from Tominaga et al. 2021 (http://ohlab.kic.ac.jp/). The source
# states no license, so these are downloaded and checked here and never stored in the repository.
GOOGLE_DOWNLOAD = "https://docs.google.com/uc?export=download&id="
PHONES = {
    "iPhone 6s": ("1dQExKthDjb45N7o5T5QWUTeKELxM03AH", "d1511879d1b5c40548805cc46e3a9a53584526c9acdd69ed9d269ba29dda7f93"),
    "iPhone SE": ("1dMec6PkUiH8Hk38Mp-om-3Dy5j40xtp7", "0dd185167a4d95e861b905880bc0be49e5a7739e3c822450c37afeae18f40e88"),
    "iPhone 8": ("1dOp0-yXR9MnDbuc5mVOO2izRFkqQICGD", "9a121a8ba49adb74b42eaeffe03efa435ed369baa3f7bb81b46ae0ba6ea82f15"),
    "iPhone X": ("1dKYksvsFi5BUUA_8aRG4uHUv-eZV-6k9", "63f992d333794e639f4b4f28f44cab8054ebb27acadfa651633d384f81ac71ce"),
    "iPhone 11": ("1d_lhS5r2uW3JuFV58Bn93hjtV_RLPQoV", "d29a5b62b7015c2f246236d8e2a495ff3c4f1a51512bd60ff8f5cbade9a38bf2"),
    "iPhone 12 Pro Max": ("1dSuouS7o-7DPx_EgXkNyUijngSVYY8dj", "9bf20d601f38cebb55048c4a0d5135071b5d1651b732f5c03bb68beb98ebc30a"),
}
for _phone, (_identifier, _digest) in PHONES.items():
    FILES[f"phone_curves/{_phone.replace(' ', '_')}.txt"] = (GOOGLE_DOWNLOAD + _identifier, _digest)


def fetch(name: str) -> Path:
    """Return a verified local copy of one raw file, downloading it if needed."""
    url, expected = FILES[name]
    path = RAW / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{name}: checksum {digest} does not match the expected {expected}")
    return path


def _missing(value: str) -> bool:
    return value.strip() in {"", "N/A", "N/D", "N/S"}


def _independent_groups(sample_ids, taxa):
    """Group samples that share a species or a specimen, so neither crosses a fold.

    A numbered file-name prefix such as Bu13 in Bu13_AspBl identifies one physical
    specimen. Plant prefixes such as Fl name a domain, not a specimen, and are ignored.
    """
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for sample, taxon in zip(sample_ids, taxa):
        keys = [f"sample:{sample}"]
        prefix = sample.split("_")[0]
        if any(ch.isdigit() for ch in prefix):
            keys.append(f"specimen:{prefix}")
        if taxon:
            keys.append(f"taxon:{taxon}")
        for key in keys[1:]:
            parent[find(key)] = find(keys[0])
    return [find(f"sample:{sample}") for sample in sample_ids]


def load_natural_objects() -> dict:
    """One row per sample: mean reflectance across repeats, plus the repeats themselves."""
    samples, order = {}, []
    wavelengths = None
    for name in ("Plants", "Animals", "Stones"):
        raw = fetch(f"Database_{name}.csv").read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
        rows = list(csv.reader(io.StringIO(text.replace("\xa0", " ")), delimiter=";"))
        header = rows[1]
        start = header.index("400")
        wl = np.array([float(x) for x in header[start:] if x.strip()])
        if wavelengths is None:
            wavelengths = wl
        elif not np.array_equal(wl, wavelengths):
            raise ValueError(f"{name}: wavelength grid differs between files")
        for row in rows[2:]:
            if not row or not row[0].strip():
                continue
            sample_id = row[0].strip()
            if sample_id not in samples:
                domain = row[4].split("(")[0].strip()
                genus, species = row[7].strip(), row[8].strip()
                has_species = not _missing(genus) and not _missing(species)
                taxon = f"{genus} {species}".replace("Aenea archidona", "Anaea archidona") if has_species else None
                samples[sample_id] = {
                    "domain": domain,
                    "object": row[5].strip(),
                    "family": "" if _missing(row[6]) else row[6].strip(),
                    "color": "" if _missing(row[9]) else row[9].strip().replace("Yelow", "Yellow"),
                    "taxon": taxon,
                    "repeats": [],
                }
                order.append(sample_id)
            samples[sample_id]["repeats"].append([float(x) for x in row[start:start + len(wl)]])
    repeats = [np.asarray(samples[s]["repeats"]) for s in order]
    groups = _independent_groups(order, [samples[s]["taxon"] for s in order])
    return {
        "ids": order,
        "domain": [samples[s]["domain"] for s in order],
        "object": [samples[s]["object"] for s in order],
        "family": [samples[s]["family"] for s in order],
        "color": [samples[s]["color"] for s in order],
        "group": groups,
        "wavelengths": wavelengths,
        "spectra": np.array([r.mean(axis=0) for r in repeats]),
        "repeats": repeats,
    }


def load_mushrooms() -> dict:
    """SpectroFood mushroom block: 250 mean reflectance spectra, species not named."""
    with fetch("SpectroFood_dataset.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    start = next(i for i, row in enumerate(rows) if row[0].strip().lower().startswith("mush"))
    wavelengths = np.array([float(x) for x in rows[start][2:] if x.strip()])
    ids, spectra, dry_matter = [], [], []
    for row in rows[start + 1:]:
        label = row[0].strip()
        if not label or not label.startswith("M") or not label[1:].isdigit():
            break
        ids.append(label)
        dry_matter.append(float(row[1]))
        spectra.append([float(x) for x in row[2:2 + len(wavelengths)]])
    return {"ids": ids, "group": list(ids), "wavelengths": wavelengths, "spectra": np.array(spectra),
            "dry_matter": np.array(dry_matter)}


def load_phone_curves() -> dict:
    """Measured sensitivities as (wavelengths, table) pairs: 400-700 nm, 10 nm steps, 3 columns."""
    wavelengths = np.arange(400.0, 701.0, 10.0)
    curves = {}
    for phone in PHONES:
        table = np.loadtxt(fetch(f"phone_curves/{phone.replace(' ', '_')}.txt"))
        if table.shape != (31, 3):
            raise ValueError(f"{phone}: expected 31 rows and 3 columns, found {table.shape}")
        curves[phone] = (wavelengths, table)
    return curves


def load_colour_tables() -> dict:
    """Curves bundled with colour-science, as (wavelengths, values) pairs."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import colour
        from colour.characterisation import MSDS_CAMERA_SENSITIVITIES, read_training_data_rawtoaces_v1

        def sd_pair(sd):
            return np.asarray(sd.wavelengths, dtype=float), np.asarray(sd.values, dtype=float)

        def msds_pair(msds):
            return np.asarray(msds.wavelengths, dtype=float), np.asarray(msds.values, dtype=float)

        chart = colour.SDS_COLOURCHECKERS["BabelColor Average"]
        chart_names = list(chart.keys())
        chart_wl = np.asarray(chart[chart_names[0]].wavelengths, dtype=float)
        training = read_training_data_rawtoaces_v1()
        illuminants = {name: sd_pair(colour.SDS_ILLUMINANTS[name])
                       for name in ("LED-B1", "LED-B4", "LED-B5", "D65", "A", "FL2")}
        return {
            "chart_wavelengths": chart_wl,
            "chart_names": chart_names,
            "chart": np.array([np.asarray(chart[name].values, dtype=float) for name in chart_names]),
            "mixed_wavelengths": np.asarray(training.wavelengths, dtype=float),
            "mixed": np.asarray(training.values, dtype=float).T,
            "cameras": {"Nikon D5100": msds_pair(MSDS_CAMERA_SENSITIVITIES["Nikon 5100 (NPL)"]),
                        "Sigma SD Merrill": msds_pair(MSDS_CAMERA_SENSITIVITIES["Sigma SDMerill (NPL)"])},
            "illuminants": illuminants,
            "apple_studio_display": msds_pair(colour.MSDS_DISPLAY_PRIMARIES["Apple Studio Display"]),
        }
