"""Where FITS files come from. ``LiveSource`` downloads the dated JSOC synoptic
archive over HTTPS into ``staging/YYYYMMDD/``; ``ReplaySource`` reads a
directory that already holds files (validation, tests). Both expose
``paths(slot)`` → {wave: Path} or None while a channel is missing."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from .grid import CADENCE, CHANNELS, day_key, file_name, file_url, floor_slot, parse_file_name

log = logging.getLogger(__name__)


class LiveSource:
    def __init__(self, staging_dir: Path, base_url: str, lookback_min: int, timeout: float):
        import requests

        self.dir = Path(staging_dir)
        self.base = base_url.rstrip("/")
        self.lookback = dt.timedelta(minutes=lookback_min)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "flare-nrt (HelioUSACH)"
        self.last_download: dt.datetime | None = None
        self._missed: dict[Path, dt.datetime] = {}  # 404s: retry at most every 5 min

    def path_for(self, slot: dt.datetime, wave: int) -> Path:
        return self.dir / day_key(slot) / file_name(slot, wave)

    def refresh(self, now: dt.datetime) -> int:
        """Fetch every missing file in [now − lookback, now]. Network errors
        are logged and retried on the next poll, never raised."""
        n = 0
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
                    self.last_download = dt.datetime.now(dt.UTC)
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
