"""Frame quality control: decides whether a synoptic slot may be used, as
current image or as part of a baseline. Every rejection carries a reason code
so the archive records why a slot is missing.

Rules (a slot is rejected if any channel fails):

- ``unreadable``: FITS can't be opened or has no 2-D image.
- ``quality``: header QUALITY != 0 (e.g. the daily 21:00 UT synoptic frame,
  QUALITY=65536, or the July 2026 spacecraft flags, QUALITY=2097152).
- ``exptime``: EXPTIME <= 0.
- ``wavelength``: WAVELNTH doesn't match the file's channel.
- ``nonfinite``: more than 1 % of pixels NaN/inf.
- ``time_offset``: T_OBS more than 90 s from the slot time.
- ``channel_spread``: channels' T_OBS more than 120 s apart.
- ``empty``: blank readout — no positive pixel, or zero variance.
- ``median_jump``: on-disk median **per second of exposure** outside
  [0.25, 4]× the channel's recent reference (dimming, partial readout).
  Synoptic images are in DN, not DN/s, and AIA's automatic exposure control
  shortens exposures during flares (131 Å drops to a few % of its normal
  value), so raw counts can't be compared across frames — and a short
  exposure can leave the raw disk median at 0, which is why ``empty`` is
  not a median test. Loose on purpose: a flare barely moves the disk median.
"""

from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grid import parse_datetime

MAX_NONFINITE = 0.01
MAX_TIME_OFFSET_S = 90
MAX_CHANNEL_SPREAD_S = 120
MEDIAN_RATIO = (0.25, 4.0)


@dataclass
class Frame:
    data: np.ndarray
    header: dict


@dataclass
class SlotQC:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    frames: dict[int, Frame] = field(default_factory=dict)
    t_obs: dict[int, str] = field(default_factory=dict)


def _header_time(h) -> dt.datetime | None:
    for key in ("T_OBS", "DATE-OBS"):
        v = h.get(key)
        if v:
            try:
                return parse_datetime(str(v).replace("_TAI", ""))
            except ValueError:
                continue
    return None


def read_frame(path: Path) -> Frame:
    """First HDU with data, as float32 (same as the validated pipeline)."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        for idx in (1, 0):
            if idx < len(hdul) and hdul[idx].data is not None:
                hdu = hdul[idx]
                return Frame(hdu.data.astype(np.float32), dict(hdu.header))
    raise ValueError("no image data")


def disk_median(data: np.ndarray, header: dict) -> float:
    """Median inside 0.8 R_sun around the disk centre (falls back to the
    central half of the image)."""
    ny, nx = data.shape
    cx = float(header.get("CRPIX1", nx / 2 + 0.5)) - 1
    cy = float(header.get("CRPIX2", ny / 2 + 0.5)) - 1
    cdelt = float(header.get("CDELT1", 0) or 0)
    rsun = float(header.get("RSUN_OBS", 0) or 0)
    r_pix = 0.8 * rsun / cdelt if cdelt > 0 and rsun > 0 else nx / 4
    yy, xx = np.ogrid[:ny, :nx]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_pix**2
    vals = data[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else 0.0


class QualityControl:
    def __init__(self, history: int = 30):
        self.ref: dict[int, deque] = {}
        self.history = history

    def check_slot(self, slot: dt.datetime, paths: dict[int, Path]) -> SlotQC:
        res = SlotQC(ok=True)
        times = []
        medians = {}
        for wave, path in sorted(paths.items()):
            try:
                fr = read_frame(path)
            except Exception:
                res.reasons.append(f"unreadable:{wave}")
                continue
            h, d = fr.header, fr.data
            if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 512:
                res.reasons.append(f"unreadable:{wave}")
                continue
            if int(h.get("QUALITY", 0) or 0) != 0:
                res.reasons.append(f"quality:{wave}={int(h.get('QUALITY'))}")
            if float(h.get("EXPTIME", 1) or 0) <= 0:
                res.reasons.append(f"exptime:{wave}")
            wl = h.get("WAVELNTH")
            if wl is not None and int(round(float(wl))) != wave:
                res.reasons.append(f"wavelength:{wave}")
            if (~np.isfinite(d)).mean() > MAX_NONFINITE:
                res.reasons.append(f"nonfinite:{wave}")
            t = _header_time(h)
            if t is not None:
                res.t_obs[wave] = t.isoformat()
                times.append(t)
                if abs((t - slot).total_seconds()) > MAX_TIME_OFFSET_S:
                    res.reasons.append(f"time_offset:{wave}")
            finite = d[np.isfinite(d)]
            if finite.size == 0 or not (finite > 0).any() or float(finite.std()) == 0.0:
                res.reasons.append(f"empty:{wave}")
            exptime = float(h.get("EXPTIME", 0) or 0)
            med = disk_median(d, h) / exptime if exptime > 0 else 0.0
            medians[wave] = med
            if med > 0:
                ref = self.ref.get(wave)
                if ref and len(ref) >= 5:
                    r = med / float(np.median(ref))
                    if not MEDIAN_RATIO[0] <= r <= MEDIAN_RATIO[1]:
                        res.reasons.append(f"median_jump:{wave}")
            res.frames[wave] = fr
        if len(times) >= 2 and (max(times) - min(times)).total_seconds() > MAX_CHANNEL_SPREAD_S:
            res.reasons.append("channel_spread")
        res.ok = not res.reasons and len(res.frames) == len(paths)
        if res.ok:
            for wave, med in medians.items():
                self.ref.setdefault(wave, deque(maxlen=self.history)).append(med)
        else:
            res.frames = {}
        return res
