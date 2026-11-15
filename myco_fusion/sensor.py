"""Sensor radiometry, heteroscedastic noise and polarization optics.

Captures real physical phenomena absent from idealized additive models:
1. Heteroscedastic noise (signal-dependent Poisson shot noise + Gaussian dark read noise).
2. Analog-to-digital converter (ADC) quantization at finite bit depths (e.g. 10, 12, 14 bits).
3. Gloss and cross-polarization extinction: specular highlights maintain the polarization
   of the illuminant and can be filtered by an orthogonal camera polarizer, leaving pure
   subsurface diffuse reflection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Optional

import numpy as np

from .spectra import _finite_array


@dataclass(frozen=True)
class RealisticNoise:
    """Parameters for physical camera sensor simulation.

    Attributes
    ----------
    read_noise_sd : float
        Standard deviation of sensor readout and dark noise (signal-independent).
    shot_noise_factor : float
        Scaling factor for photon shot noise (variance proportional to signal).
    gain_error_sd : float
        Standard deviation of multiplicative exposure/gain variation across shots.
    bit_depth : int, optional
        ADC quantization bit depth (e.g. 10, 12, 14, 16 bits). None retains floating point.
    """
    read_noise_sd: float = 0.002
    shot_noise_factor: float = 0.005
    gain_error_sd: float = 0.02
    bit_depth: Optional[int] = None


def simulate_sensor(clean_rgb, noise: RealisticNoise, rng: np.random.Generator, clip: bool = True) -> np.ndarray:
    """Simulate realistic CMOS sensor capture with Poisson-Gaussian noise and ADC quantization.

    Parameters
    ----------
    clean_rgb : array-like of shape (..., 3)
        Noiseless sensor linear response in [0, 1].
    noise : RealisticNoise
        Noise and quantization parameters.
    rng : np.random.Generator
        NumPy random number generator.
    clip : bool
        Whether to clip outputs to [0, 1].
    """
    rgb = _finite_array(clean_rgb, "clean_rgb")
    if rgb.shape[-1] != 3:
        raise ValueError("clean_rgb last dimension must be 3")

    flat = rgb.reshape(-1, 3)
    n = len(flat)

    gain = 1.0 + rng.normal(0, noise.gain_error_sd, (n, 1))
    scaled = flat * gain

    # Heteroscedastic noise: shot noise variance proportional to signal, plus read noise
    var_shot = (noise.shot_noise_factor ** 2) * np.maximum(scaled, 0.0)
    var_read = noise.read_noise_sd ** 2
    sd_total = np.sqrt(var_shot + var_read)
    noisy = scaled + rng.normal(0, sd_total)

    if clip:
        noisy = np.clip(noisy, 0.0, 1.0)

    if noise.bit_depth is not None:
        if not 8 <= noise.bit_depth <= 16:
            raise ValueError("bit_depth must be between 8 and 16")
        levels = (1 << noise.bit_depth) - 1
        noisy = np.round(noisy * levels) / levels

    return noisy.reshape(rgb.shape)


def simulate_cross_polarization(diffuse_rgb, specular_ratio, illuminant_rgb, cross_polarized: bool = True) -> np.ndarray:
    """Model specular gloss vs cross-polarization glare extinction.

    Biological specimens often have waxy, wet, or gelatinous surfaces (viscid mushroom
    caps, leaf cuticles, fruit skins) that reflect a specular highlight of the light
    source.

    Under unpolarized flash:
        RGB_observed = (1 - k_s) * RGB_diffuse + k_s * RGB_flash

    Under cross-polarization (linear polarizer over the flash, perpendicular polarizer
    over the camera lens):
        Specular reflection preserves polarization and is extinguished by the lens polarizer.
        Only light undergoing multiple internal scattering events depolarizes and exits:
        RGB_observed = (1 - k_s) * RGB_diffuse

    Parameters
    ----------
    diffuse_rgb : array-like of shape (..., 3)
        Intrinsic diffuse subsurface color.
    specular_ratio : float
        Fraction of surface reflection that is specular (k_s in [0, 1]).
    illuminant_rgb : array-like of shape (3,)
        Color of the illuminant (white flash).
    cross_polarized : bool
        If True, extinguishes the specular highlight.
    """
    diff = _finite_array(diffuse_rgb, "diffuse_rgb")
    illum = _finite_array(illuminant_rgb, "illuminant_rgb")
    if diff.shape[-1] != 3 or illum.shape != (3,):
        raise ValueError("diffuse_rgb and illuminant_rgb must have 3 channels")
    ks = float(specular_ratio)
    if not 0.0 <= ks <= 1.0:
        raise ValueError("specular_ratio must be in [0, 1]")

    if cross_polarized:
        return (1.0 - ks) * diff
    return (1.0 - ks) * diff + ks * illum


def srgb_to_linear(value: float) -> float:
    """IEC 61966-2-1 sRGB byte value [0, 255] to linear float in [0, 1]."""
    v = float(value) / 255.0
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


RAW_EXTENSIONS = {".dng"}
RAW_DECODER_HINT = (
    "DNG files need a true raw decoder (rawpy) so that the values stay sensor-linear. "
    "Without it they would pass through an 8-bit tone-mapped conversion, which is not linear, "
    "so they are refused. Install rawpy in your own environment (python3 -m pip install rawpy) "
    "or capture locked-exposure JPEG/PNG instead."
)


def raw_decoder_available() -> bool:
    """True when rawpy (LibRaw) can be imported."""
    try:
        import rawpy  # noqa: F401
    except ImportError:
        return False
    return True


def _linear_raw_rgb(image_path: Path) -> np.ndarray:
    """Decode a raw file to linear camera RGB in [0, 1] with rawpy.

    No gamma, no automatic brightening, unit white-balance multipliers and the camera's own
    color space, so the values stay proportional to the black-subtracted sensor signal.
    """
    import rawpy

    with rawpy.imread(str(image_path)) as raw:
        rgb16 = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=[1.0, 1.0, 1.0, 1.0],
            output_color=rawpy.ColorSpace.raw,
            user_flip=0,
        )
    return rgb16.astype(float) / 65535.0


def open_image(image_path: Path):
    """Open a non-raw image file, with a macOS sips fallback for HEIC.

    DNG files are refused here; ``extract_box_rgb`` decodes them with rawpy when available.
    """
    from PIL import Image

    image_path = Path(image_path)
    if image_path.suffix.lower() in RAW_EXTENSIONS:
        raise ValueError(RAW_DECODER_HINT)
    try:
        img = Image.open(image_path)
        img.load()
        return img
    except Exception:
        if sys.platform == "darwin" and image_path.suffix.lower() in {".heic"}:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                res = subprocess.run(
                    ["sips", "-s", "format", "png", str(image_path), "--out", str(tmp_path)],
                    capture_output=True,
                )
                if res.returncode == 0:
                    img = Image.open(tmp_path)
                    img.load()
                    return img
            finally:
                tmp_path.unlink(missing_ok=True)
        raise


def extract_exif(image_path: Path) -> dict:
    """Extract exposure and camera metadata from an image if available."""
    try:
        from PIL import ExifTags, Image

        with Image.open(image_path) as img:
            exif_raw = img.getexif()
            if not exif_raw:
                return {}
            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
            return {
                "iso": exif.get("ISOSpeedRatings") or exif.get("PhotographicSensitivity"),
                "exposure_time": str(exif.get("ExposureTime")),
                "f_number": float(exif["FNumber"]) if "FNumber" in exif else None,
                "focal_length": float(exif["FocalLength"]) if "FocalLength" in exif else None,
                "model": str(exif.get("Model", "")),
            }
    except Exception:
        return {}


def _box_from_mode(box_mode: str, w: int, h: int) -> tuple[int, int, int, int]:
    if box_mode.startswith("center:"):
        pct = float(box_mode.split(":")[1]) / 100.0
        bw, bh = max(1, int(w * pct)), max(1, int(h * pct))
        return (w - bw) // 2, (h - bh) // 2, bw, bh
    if "," in box_mode:
        parts = [float(p.strip()) for p in box_mode.split(",")]
        if max(parts) <= 1.0:
            return int(parts[0] * w), int(parts[1] * h), max(1, int(parts[2] * w)), max(1, int(parts[3] * h))
        return int(parts[0]), int(parts[1]), max(1, int(parts[2])), max(1, int(parts[3]))
    raise ValueError(f"Unrecognized box mode: {box_mode}")


def extract_box_rgb(image_path: Path, box_mode: str = "center:30") -> tuple[list[float], float, dict]:
    """Extract mean linear RGB from a region of interest and compute the fraction of clipped pixels.

    JPEG, PNG, TIFF and HEIC files are linearized with the inverse sRGB curve, which is only an
    approximation for tone-mapped phone images. DNG files are decoded with rawpy when it is
    installed (sensor-linear camera RGB, no inverse sRGB) and refused otherwise.
    """
    image_path = Path(image_path)
    exif = extract_exif(image_path)
    if image_path.suffix.lower() in RAW_EXTENSIONS:
        if not raw_decoder_available():
            raise ValueError(RAW_DECODER_HINT)
        linear = _linear_raw_rgb(image_path)
        h, w = linear.shape[:2]
        bx, by, bw, bh = _box_from_mode(box_mode, w, h)
        crop = linear[by:by + bh, bx:bx + bw, :3].reshape(-1, 3)
        clip_ratio = float(np.mean((crop >= 0.999).any(axis=1))) if len(crop) else 0.0
        return np.mean(crop, axis=0).tolist(), clip_ratio, exif
    with open_image(image_path) as img:
        rgb_img = img.convert("RGB")
        w, h = rgb_img.size
        bx, by, bw, bh = _box_from_mode(box_mode, w, h)
        crop = rgb_img.crop((bx, by, bx + bw, by + bh))
        arr = np.asarray(crop, dtype=np.uint8)

    flat = arr.reshape(-1, 3)
    clipped_pixels = np.sum((flat >= 254).any(axis=1))
    clip_ratio = float(clipped_pixels) / max(1, len(flat))

    v = flat.astype(float) / 255.0
    linear_arr = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    mean_linear = np.mean(linear_arr, axis=0).tolist()
    return mean_linear, clip_ratio, exif
