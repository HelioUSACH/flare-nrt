"""Where on the disk the change is (experimental — not validated against
flare catalogues yet, docs/model-card.md).

The brightest connected region of the 131 Å difference channel, in
helioprojective arcsec computed from the 1024² synoptic header. Array row 0
is the first FITS row, i.e. the *south* edge for a north-up image: y grows
with the row index (the v3 pipeline's localiser appears to have had y flipped;
this module is written from the FITS convention instead)."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

DIFF_131 = 4  # channel index of 131 Å difference in the 8-ch tensor


def localize(x: torch.Tensor, header: dict, src_size: int | None = None) -> dict | None:
    src_size = int(src_size or header.get("NAXIS1") or 1024)
    d = x[DIFF_131].float().numpy()
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
    return {
        "centroid_arcsec": [round(xa, 1), round(ya, 1)],
        "area_px": area_px,
        "area_arcsec2": round(area_px * (cdelt1 * scale) * (cdelt2 * scale), 1),
        "on_limb": bool(rad > 0.9 * rsun),
        "method": "diff131_brightest_region",
        "confidence": "experimental",
    }
