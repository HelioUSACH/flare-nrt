"""PNG previews of one slot for the portal: each AIA channel as an image
(DN/s, asinh stretch, SDO-style colour table) or as the difference the
model looks at (DN/s, current − mean of its baselines, grey ±).

Rendered on request from the staging FITS and cached on disk for a day;
nothing extra is stored by the loop. North up (FITS row 0 is south, so the
array is flipped for display), 512 px."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from .qc import read_frame

SIZE = 512


def _lut(wave: int) -> np.ndarray:
    """256×3 colour table in the style of the SDO/AIA tables (built from
    IDL table 3 and power curves, as SunPy does); an approximation for
    display, not a calibrated product."""
    i = np.arange(256, dtype=float)
    r0 = np.clip(i * 255 / 176, 0, 255)
    g0 = np.clip((i - 120) * 255 / 135, 0, 255)
    b0 = np.clip((i - 190) * 255 / 65, 0, 255)
    c0 = i
    c1 = np.sqrt(i) * np.sqrt(255.0)
    c2 = i**2 / 255.0
    c3 = (c1 + c2 / 2.0) * 255.0 / (c1.max() + c2.max() / 2.0)
    table = {131: (g0, r0, r0), 171: (r0, c0, b0), 304: (r0, g0, b0), 1600: (c3, c3, c2)}
    r, g, b = table.get(wave, (c0, c0, c0))
    return np.stack([r, g, b], axis=1).clip(0, 255).astype(np.uint8)


def _downsample(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    if n % SIZE == 0:
        k = n // SIZE
        return a.reshape(SIZE, k, SIZE, k).mean(axis=(1, 3))
    idx = (np.arange(SIZE) * n / SIZE).astype(int)
    return a[np.ix_(idx, idx)]


def rate(path: Path) -> np.ndarray:
    fr = read_frame(path)
    exptime = float(fr.header.get("EXPTIME", 0) or 0)
    data = np.nan_to_num(fr.data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0, None) / (exptime if exptime > 0 else 1.0)
    return _downsample(data)


def _png(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb[::-1].copy(), "RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def image_png(path: Path, wave: int) -> bytes:
    v = rate(path)
    hi = float(np.percentile(v, 99.7)) or 1.0
    k = 0.05
    x = np.arcsinh(v / (k * hi)) / np.arcsinh(1 / k)
    idx = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return _png(_lut(wave)[idx])


def diff_png(path: Path, baseline_paths: list[Path]) -> bytes:
    cur = rate(path)
    bl = np.mean([rate(p) for p in baseline_paths], axis=0)
    d = cur - bl
    s = float(np.percentile(np.abs(d), 99.5)) or 1.0
    g = (127.5 + 127.5 * np.clip(d / s, -1, 1)).astype(np.uint8)
    return _png(np.stack([g, g, g], axis=-1))
