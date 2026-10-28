"""Where FITS files come from. ``LiveSource`` fills ``staging/YYYYMMDD/`` over
HTTPS from two JSOC places: ``mostrecent/`` (the newest image of each channel,
updated in real time; each is filed under every 2-min slot whose time is
within 90 s of its own T_OBS, the same tolerance QC applies)
and the dated archive (``YYYY/MM/DD/HHH00/``, which fills gaps but can lag
behind real time); ``ReplaySource`` reads a
directory that already holds files (validation, tests). Both expose
``paths(slot)`` → {wave: Path} or None while a channel is missing."""

from __future__ import annotations

import datetime as dt
import logging
import os
import shutil
from pathlib import Path

from .grid import CADENCE, CHANNELS, day_key, file_name, file_url, floor_slot, parse_datetime, parse_file_name

log = logging.getLogger(__name__)

MAX_OFFSET = dt.timedelta(seconds=90)  # = qc.MAX_TIME_OFFSET_S


def slots_for(t: dt.datetime) -> list[dt.datetime]:
    """The 2-min slots an image taken at ``t`` can stand for: those whose
    time is within 90 s of it (one or two slots, nearest first)."""
    s0 = floor_slot(t)
    cands = sorted((s0, s0 + CADENCE), key=lambda s: abs(t - s))
    return [s for s in cands if abs(t - s) <= MAX_OFFSET]


class LiveSource:
    def __init__(self, staging_dir: Path, base_url: str, lookback_min: int, timeout: float):
        import requests

        self.dir = Path(staging_dir)
        self.base = base_url.rstrip("/")
        self.lookback = dt.timedelta(minutes=lookback_min)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Centella (HelioUSACH)"
        self.last_download: dt.datetime | None = None
        self._missed: dict[Path, dt.datetime] = {}  # 404s: retry at most every 5 min

    def path_for(self, slot: dt.datetime, wave: int) -> Path:
        return self.dir / day_key(slot) / file_name(slot, wave)

    def refresh_mostrecent(self) -> int:
        """The newest image of each channel, filed under the slot(s) it stands for."""
        from astropy.io import fits

        n = 0
        for w in CHANNELS:
            url = f"{self.base}/mostrecent/AIAsynoptic{w:04d}.fits"
            try:
                r = self.session.get(url, timeout=self.timeout)
            except Exception as e:
                log.warning("mostrecent %d failed: %s", w, e)
                continue
            if r.status_code != 200 or not r.content:
                log.warning("mostrecent %d: HTTP %s", w, r.status_code)
                continue
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.dir / f".mostrecent_{w}.part"
            tmp.write_bytes(r.content)
            try:
                with fits.open(tmp) as h:
                    hdr = next((x.header for x in h if x.data is not None), h[0].header)
                    t_obs = hdr.get("T_OBS") or hdr.get("DATE-OBS")
                t = parse_datetime(str(t_obs).replace("_TAI", ""))
            except Exception as e:
                log.warning("mostrecent %d unreadable: %s", w, e)
                tmp.unlink(missing_ok=True)
                continue
            new = False
            for slot in slots_for(t):
                dest = self.path_for(slot, w)
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(tmp, dest)       # same bytes under both slots, no extra disk
                except OSError:
                    shutil.copyfile(tmp, dest)
                new = True
            tmp.unlink(missing_ok=True)
            if new:
                n += 1
                self.last_download = dt.datetime.now(dt.timezone.utc)
        return n

    def refresh(self, now: dt.datetime) -> int:
        """Newest images first, then every missing file of [now − lookback,
        now] from the dated archive. Network errors are logged and retried
        on the next poll, never raised."""
        n = self.refresh_mostrecent()
        s, end = floor_slot(now - self.lookback), floor_slot(now)
        while s <= end:
            for w in CHANNELS:
                p = self.path_for(s, w)
                if p.exists():
                    continue
                miss = self._missed.get(p)
                if miss and now - miss < dt.timedelta(minutes=5):
                    continue
                try:
                    r = self.session.get(file_url(self.base, s, w), timeout=self.timeout)
                except Exception as e:  # network down: try again next poll
                    log.warning("download failed %s: %s", p.name, e)
                    return n
                if r.status_code == 200 and r.content:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_suffix(".part")
                    tmp.write_bytes(r.content)
                    tmp.rename(p)
                    n += 1
                    self.last_download = dt.datetime.now(dt.timezone.utc)
                    self._missed.pop(p, None)
                else:
                    self._missed[p] = now
            s += CADENCE
        cutoff = now - 2 * self.lookback
        self._missed = {k: v for k, v in self._missed.items() if v > cutoff}
        return n

    def paths(self, slot: dt.datetime) -> dict[int, Path] | None:
        ps = {w: self.path_for(slot, w) for w in CHANNELS}
        return ps if all(p.exists() for p in ps.values()) else None


class ReplaySource:
    def __init__(self, root: Path):
        self.frames: dict[dt.datetime, dict[int, Path]] = {}
        for p in Path(root).rglob("AIA*.fits"):
            parsed = parse_file_name(p.name)
            if parsed:
                self.frames.setdefault(parsed[0], {})[parsed[1]] = p
        self.slots = sorted(self.frames)

    def refresh(self, now: dt.datetime) -> int:
        return 0

    def paths(self, slot: dt.datetime) -> dict[int, Path] | None:
        d = self.frames.get(slot)
        return d if d and all(w in d for w in CHANNELS) else None
