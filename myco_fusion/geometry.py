"""Point-centroid projection prototype, NOT a Gaussian splat renderer.

Coordinates: right-handed OpenCV camera (+x right, +y down, +z forward).
The supplied 4x4 matrix maps final reconstruction-world points to the camera.
Pixels use integer centers and nearest-center sampling, with ties rounded up.
Depth tolerance has the same units as points and the translation vector.
"""
from __future__ import annotations

import numpy as np


def project_spectra(points, cube, intrinsics, world_to_camera, depth_tolerance=0.005):
    """Associate a finite spectral cube with visible point centroids.

    Missing values are NaN with zero counts. Whole spectra with nonfinite values
    are omitted. A nearest-centroid z-buffer handles only point occlusion; extended
    Gaussian covariance, opacity, surface normals and uncertainty are not modeled.
    """
    points = np.asarray(points, dtype=float)
    cube = np.asarray(cube, dtype=float)
    K = np.asarray(intrinsics, dtype=float)
    T = np.asarray(world_to_camera, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite N x 3 array")
    if cube.ndim != 3 or min(cube.shape) < 1:
        raise ValueError("cube must be H x W x bands with nonempty dimensions")
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("intrinsics must be finite 3 x 3")
    if not np.allclose(K[2], [0, 0, 1]) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("intrinsics must use pinhole convention with positive focal lengths")
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("world_to_camera must be a finite homogeneous 4 x 4 transform")
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1):
        raise ValueError("world_to_camera rotation must be proper and orthonormal")
    if not np.isfinite(depth_tolerance) or depth_tolerance < 0:
        raise ValueError("depth_tolerance must be finite and nonnegative")
    h, w, bands = cube.shape
    camera = points @ R.T + T[:3, 3]
    positive = np.flatnonzero(camera[:, 2] > 0)
    uvw = camera[positive] @ K.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    # Mask in floating point BEFORE integer conversion; then check actual indices.
    valid = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= -0.5) & (uv[:, 0] < w - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < h - 0.5)
    idx = positive[valid]
    pixels = np.floor(uv[valid] + 0.5).astype(np.int64)
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    idx, pixels = idx[inside], pixels[inside]
    u, v = pixels[:, 0], pixels[:, 1]
    z = camera[idx, 2]
    zbuf = np.full((h, w), np.inf)
    np.minimum.at(zbuf, (v, u), z)
    keep = (z - zbuf[v, u] <= depth_tolerance) & np.isfinite(cube[v, u]).all(axis=1)
    spectra = np.full((len(points), bands), np.nan)
    counts = np.zeros(len(points), dtype=np.int64)
    spectra[idx[keep]] = cube[v[keep], u[keep]]
    counts[idx[keep]] = 1
    return {"spectra": spectra, "n_views": counts, "method": "point_centroid_zbuffer_proxy"}


def combine_views(results):
    """Equal-view mean, retaining NaN when no valid spectrum is available."""
    if not results:
        raise ValueError("at least one view is required")
    shape = np.asarray(results[0]["spectra"]).shape
    if len(shape) != 2:
        raise ValueError("spectra must be N x bands")
    total, counts = np.zeros(shape), np.zeros(shape[0], dtype=np.int64)
    for result in results:
        spec, n = np.asarray(result["spectra"], float), np.asarray(result["n_views"])
        if spec.shape != shape or n.shape != (shape[0],):
            raise ValueError("all views must have matching point and band dimensions")
        if not np.isfinite(n).all() or (n < 0).any() or (n != np.floor(n)).any():
            raise ValueError("view counts must be nonnegative integers")
        valid = (n > 0) & np.isfinite(spec).all(axis=1)
        total[valid] += spec[valid] * n[valid, None]
        counts[valid] += n[valid].astype(np.int64)
    out = np.full(shape, np.nan)
    np.divide(total, counts[:, None], out=out, where=counts[:, None] > 0)
    return {"spectra": out, "n_views": counts, "method": "equal_view_point_centroid_proxy"}
