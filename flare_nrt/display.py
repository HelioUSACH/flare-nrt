"""Pre-rendered display files (1.1.0): what the portal plays back without
the API rendering anything on request.

- ``render/YYYYMMDD/HHMM/`` per result, written by the live loop right
  after the result: ``171.webp`` (the 171 Å image, 512 px), ``171-diff.webp``
  (171 Å current − mean of the baselines the result used) and
  ``gradcam-heat.webp`` (the Grad-CAM heat style; only when the loop
  computed a Grad-CAM, i.e. the newest frame and alerts, not each slot of a
  backlog). WebP at ``webp_quality`` (90). Kept like the raw FITS
  (``staging_days``, 4) and pruned with them (flare_nrt/retention.py).

Nothing here feeds back into a result: a failed render is logged and the
result stands."""

from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

from . import render
from .artifacts import _write
from .grid import day_key, file_name

log = logging.getLogger(__name__)

RENDER_CHANNELS = ("171", "171-diff", "gradcam-heat")


def render_slot_dir(root: Path, slot: dt.datetime) -> Path:
    return root / f"{slot:%Y%m%d}" / f"{slot:%H%M}"


def render_path(root: Path, slot: dt.datetime, channel: str) -> Path:
    return render_slot_dir(root, slot) / f"{channel}.webp"


class FitsLocator:
    """Where the FITS of a slot is: ``staging/YYYYMMDD/`` first, then the
    read-only trees of ``FLARE_NRT_FITS_DIRS`` by file name (indexed, rebuilt
    at most every 10 min)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._built = 0.0
        self._paths: dict[str, Path] = {}

    def _indexed(self, name: str) -> Path | None:
        if not self.cfg.fits_dirs:
            return None
        if time.time() - self._built > 600:
            paths: dict[str, Path] = {}
            for root in self.cfg.fits_dirs:
                for p in Path(root).rglob("AIA*.fits"):
                    paths.setdefault(p.name, p)
            self._built, self._paths = time.time(), paths
        return self._paths.get(name)

    def path(self, slot: dt.datetime, wave: int) -> Path:
        """The file, or its (missing) staging path."""
        p = self.cfg.staging_dir / day_key(slot) / file_name(slot, wave)
        return p if p.exists() else (self._indexed(p.name) or p)

    def existing(self, slot: dt.datetime, wave: int) -> Path | None:
        p = self.path(slot, wave)
        return p if p.exists() else None


def write_result_renders(cfg, slot: dt.datetime, fits171: Path | None, baselines171: list[Path],
                         cam: dict | None) -> list[str]:
    """The three WebP files of one result (whichever can be made). Never
    raises; returns the channels written."""
    out = []
    q = cfg.webp_quality
    jobs = []
    if fits171 is not None and Path(fits171).exists():
        jobs.append(("171", lambda: render.image(fits171, 171, fmt="webp", quality=q)))
        bls = [b for b in baselines171 if b is not None and Path(b).exists()]
        if bls and len(bls) >= cfg.min_baseline:
            jobs.append(("171-diff", lambda: render.diff(fits171, bls, fmt="webp", quality=q)))
    if cam is not None:
        jobs.append(("gradcam-heat", lambda: render.gradcam_heat(cam["cam"], cam["limb"], fmt="webp", quality=q)))
    for name, make in jobs:
        try:
            _write(render_path(cfg.render_dir, slot, name), make())
            out.append(name)
        except Exception:
            log.exception("rendering %s for %s failed", name, slot)
    return out
