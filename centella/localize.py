"""Where on the disk the change is (experimental — not validated against
flare catalogues yet, docs/model-card.md).

The brightest connected region of the 131 Å difference **in DN/s**
(current/EXPTIME − mean of the baselines/EXPTIME, on the 512 grid), in
helioprojective arcsec computed from the synoptic header. Not the model's
difference channel: that one is in raw DN, and AIA's exposure control cuts
131 Å exposures during flares, so the raw difference is often negative right
where the flare is. Array row 0
is the first FITS row, i.e. the *south* edge for a north-up image: y grows
with the row index (the v3 pipeline's localiser appears to have had y flipped;
this module is written from the FITS convention instead)."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def rate_map(data: np.ndarray, header: dict) -> np.ndarray:
    """131 Å image in DN/s on the 512 grid (same resampling as the model)."""
    from .preprocess import resize_512

    exptime = float(header.get("EXPTIME", 0) or 0)
    img = np.clip(np.nan_to_num(data, nan=0.0), 0, None) / (exptime if exptime > 0 else 1.0)
    return resize_512(img).numpy()


def localize(diff: np.ndarray | torch.Tensor, header: dict, src_size: int | None = None) -> dict | None:
    """``diff``: 2-D difference map on the 512 grid (DN/s)."""
    src_size = int(src_size or header.get("NAXIS1") or 1024)
    d = diff.float().numpy() if isinstance(diff, torch.Tensor) else np.asarray(diff, dtype=np.float32)
    d = ndimage.gaussian_filter(d, 2.0)
    peak = float(d.max())
    if peak <= 0:
        return None
    thr = max(0.5 * peak, float(d.mean() + 5 * d.std()))
    lab, n = ndimage.label(d >= thr)
    if n == 0:
        return None
    sums = ndimage.sum(np.clip(d, 0, None), lab, index=range(1, n + 1))
    k = int(np.argmax(sums)) + 1
    region = lab == k
    w = np.where(region, np.clip(d, 0, None), 0)
    rows, cols = np.indices(d.shape)
    r = float((rows * w).sum() / w.sum())
    c = float((cols * w).sum() / w.sum())
    scale = src_size / d.shape[0]
    # 512-grid centre (0-based) → 1024-grid pixel (1-based FITS)
    px = (c + 0.5) * scale + 0.5
    py = (r + 0.5) * scale + 0.5
    cdelt1 = float(header.get("CDELT1") or 0.6 * 4096 / src_size)
    cdelt2 = float(header.get("CDELT2", cdelt1) or cdelt1)
    xa = (px - float(header.get("CRPIX1", src_size / 2 + 0.5))) * cdelt1
    ya = (py - float(header.get("CRPIX2", src_size / 2 + 0.5))) * cdelt2
    rsun = float(header.get("RSUN_OBS", 960.0) or 960.0)
    rad = float(np.hypot(xa, ya))
    area_px = int(region.sum())
    # the region's bounding box, pixel edges → helioprojective arcsec (x grows west,
    # y north: rows are FITS rows, so the first row is the south edge)
    rr, cc = np.nonzero(region)

    def hpc(col_edge: float, row_edge: float) -> tuple[float, float]:
        return ((col_edge * scale + 0.5 - float(header.get("CRPIX1", src_size / 2 + 0.5))) * cdelt1,
                (row_edge * scale + 0.5 - float(header.get("CRPIX2", src_size / 2 + 0.5))) * cdelt2)

    x0, y0 = hpc(float(cc.min()), float(rr.min()))
    x1, y1 = hpc(float(cc.max()) + 1, float(rr.max()) + 1)
    return {
        "centroid_arcsec": [round(xa, 1), round(ya, 1)],
        "area_px": area_px,
        "area_arcsec2": round(area_px * (cdelt1 * scale) * (cdelt2 * scale), 1),
        "bbox_arcsec": [round(min(x0, x1), 1), round(min(y0, y1), 1), round(max(x0, x1), 1), round(max(y0, y1), 1)],
        "on_limb": bool(rad > 0.9 * rsun),
        "method": "diff131_rate_brightest_region",
        "confidence": "experimental",
    }
