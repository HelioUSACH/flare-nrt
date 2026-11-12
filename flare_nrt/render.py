"""Display images of one slot for the portal. Nothing here touches the model
or its inputs: it only turns FITS and stored Grad-CAM maps into pictures.

- each AIA channel as an image (DN/s, per-channel limits and stretch,
  SDO-style colour table), 512 px or 1024 px;
- the difference the model looks at (DN/s, current − mean of its
  baselines, grey ±), 512 px;
- the Grad-CAM as a heat map (``gradcam_heat``): the map alone, normalised
  to its own maximum, inferno colours, transparent below ``HEAT_FLOOR``,
  the solar limb as a thin circle.

PNG or WebP (``FORMATS``). North up (FITS row 0 is south, so arrays are
flipped for display)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from .qc import read_frame

SIZE = 512
SIZES = (512, 1024)
FORMATS = {"png": "image/png", "webp": "image/webp"}
WEBP_QUALITY = 90

# Per-channel scaling of the DN/s image: (stretch, high percentile, asinh softening).
# 131/171/304: asinh up to the 99.7th percentile, as in 1.0.0 (byte-identical output).
# 1600: linear from 0 to the 99.9th percentile. The UV continuum disk spans only
# ~2× between limb and centre, so the asinh stretch put the whole disk at
# 0.66–0.98 of the colour range (a pale uniform disk). Linear keeps limb
# darkening (limb ≈ 0.25, centre ≈ 0.4 of the range) and plage (≈ 0.75–1).
SCALING = {
    131: ("asinh", 99.7, 0.05),
    171: ("asinh", 99.7, 0.05),
    304: ("asinh", 99.7, 0.05),
    1600: ("linear", 99.9, None),
}

# Grad-CAM heat style: matplotlib's "inferno" (perceptually uniform), 17 stops
# at 0, 1/16, …, 1, linearly interpolated; the same stops are in INTEGRATION.md
# for the portal's legend.
INFERNO_STOPS = ("#000004", "#0b0724", "#210c4a", "#3d0965", "#57106e", "#71196e", "#8a226a", "#a32c61",
                 "#bc3754", "#d24644", "#e45a31", "#f1731d", "#f98e09", "#fcac11", "#f9cb35", "#f2ea69",
                 "#fcffa4")
HEAT_FLOOR = 0.05          # map / its maximum below this: fully transparent
HEAT_RAMP = 0.05           # then opacity rises linearly to 1 over this much
LIMB_RGBA = (200, 200, 200, 210)
LIMB_WIDTH_PX = 1.5


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


def heat_lut() -> np.ndarray:
    """256×3 uint8 from ``INFERNO_STOPS``."""
    stops = np.array([[int(h[k:k + 2], 16) for k in (1, 3, 5)] for h in INFERNO_STOPS], dtype=float)
    pos = np.linspace(0, 1, len(stops))
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, pos, stops[:, c]) for c in range(3)], axis=1).round().astype(np.uint8)


def _downsample(a: np.ndarray, size: int = SIZE) -> np.ndarray:
    n = a.shape[0]
    if n % size == 0:          # also n == size (float64 out, as in 1.0.0)
        k = n // size
        return a.reshape(size, k, size, k).mean(axis=(1, 3))
    idx = (np.arange(size) * n / size).astype(int)
    return a[np.ix_(idx, idx)]


def rate(path: Path, size: int = SIZE) -> np.ndarray:
    fr = read_frame(path)
    exptime = float(fr.header.get("EXPTIME", 0) or 0)
    data = np.nan_to_num(fr.data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0, None) / (exptime if exptime > 0 else 1.0)
    return _downsample(data, size)


def encode(pixels: np.ndarray, fmt: str = "png", quality: int = WEBP_QUALITY, flip: bool = True) -> bytes:
    """uint8 (H, W, 3) RGB or (H, W, 4) RGBA → PNG (optimised, lossless) or
    WebP (lossy, ``quality``). ``flip``: the array is in FITS row order."""
    from PIL import Image

    a = pixels[::-1].copy() if flip else pixels
    img = Image.fromarray(a, "RGBA" if a.shape[-1] == 4 else "RGB")
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG", optimize=True)
    elif fmt == "webp":
        img.save(buf, format="WEBP", quality=quality, method=4)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    return buf.getvalue()


def scale(v: np.ndarray, wave: int) -> np.ndarray:
    """DN/s → [0, 1] with the channel's limits and stretch (``SCALING``)."""
    stretch, pct, k = SCALING.get(wave, ("asinh", 99.7, 0.05))
    hi = float(np.percentile(v, pct)) or 1.0
    if stretch == "linear":
        return np.clip(v / hi, 0, 1)
    return np.clip(np.arcsinh(v / (k * hi)) / np.arcsinh(1 / k), 0, 1)


def image(path: Path, wave: int, size: int = SIZE, fmt: str = "png", quality: int = WEBP_QUALITY) -> bytes:
    idx = (scale(rate(path, size), wave) * 255).astype(np.uint8)
    return encode(_lut(wave)[idx], fmt, quality)


def diff(path: Path, baseline_paths: list[Path], fmt: str = "png", quality: int = WEBP_QUALITY) -> bytes:
    cur = rate(path)
    bl = np.mean([rate(p) for p in baseline_paths], axis=0)
    d = cur - bl
    s = float(np.percentile(np.abs(d), 99.5)) or 1.0
    g = (127.5 + 127.5 * np.clip(d / s, -1, 1)).astype(np.uint8)
    return encode(np.stack([g, g, g], axis=-1), fmt, quality)


def image_png(path: Path, wave: int) -> bytes:
    return image(path, wave)


def diff_png(path: Path, baseline_paths: list[Path]) -> bytes:
    return diff(path, baseline_paths)


def limb_from_header(header: dict, size: int = SIZE) -> list[float] | None:
    """Solar limb on the ``size`` grid, FITS row order: [x_centre, y_centre,
    radius] in pixels (0-based pixel centres), or None without the keys."""
    try:
        s = float(header.get("NAXIS1") or 1024) / size
        cdelt = float(header["CDELT1"])
        return [round((float(header["CRPIX1"]) - 0.5) / s - 0.5, 2),
                round((float(header["CRPIX2"]) - 0.5) / s - 0.5, 2),
                round(float(header["RSUN_OBS"]) / cdelt / s, 2)]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def heat_rgba(cam: np.ndarray, limb: list[float] | None) -> np.ndarray:
    """The Grad-CAM map alone, FITS row order (float in [0, 1], or uint8
    0–255) → (H, W, 4) uint8: normalised to its own maximum, inferno, alpha 0
    below ``HEAT_FLOOR`` rising to 1 over ``HEAT_RAMP``, the limb circle
    composited on top. No AIA underlay."""
    c = np.asarray(cam)
    c = c.astype(np.float32) / 255.0 if c.dtype.kind in "ui" else c.astype(np.float32)
    mx = float(c.max())
    c = np.clip(c / mx if mx > 0 else c, 0, 1)
    heat = heat_lut()[(c * 255).astype(np.uint8)].astype(np.float32)
    a = np.clip((c - HEAT_FLOOR) / HEAT_RAMP, 0, 1)
    a[c < HEAT_FLOOR] = 0.0
    rgb = heat
    if limb:
        cx, cy, r = limb
        yy, xx = np.mgrid[:c.shape[0], :c.shape[1]]
        la = np.clip(1.0 - np.abs(np.hypot(xx - cx, yy - cy) - r) / LIMB_WIDTH_PX, 0, 1) * (LIMB_RGBA[3] / 255)
        out_a = la + a * (1 - la)                                   # limb "over" heat
        num = np.array(LIMB_RGBA[:3], np.float32) * la[..., None] + heat * (a * (1 - la))[..., None]
        rgb = np.where(out_a[..., None] > 0, num / np.maximum(out_a, 1e-6)[..., None], 0)
        a = out_a
    return np.dstack([np.clip(rgb, 0, 255), a * 255]).round().astype(np.uint8)


def gradcam_heat(cam: np.ndarray, limb: list[float] | None, fmt: str = "png",
                 quality: int = WEBP_QUALITY) -> bytes:
    return encode(heat_rgba(cam, limb), fmt, quality)


def reencode(data: bytes, fmt: str, quality: int = WEBP_QUALITY) -> bytes:
    """A stored PNG in another format (the Grad-CAM overlay as WebP)."""
    from PIL import Image

    if fmt == "png":
        return data
    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).save(buf, format="WEBP", quality=quality, method=4)
    return buf.getvalue()
