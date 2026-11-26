"""Where FITS files come from. ``LiveSource`` fills ``staging/YYYYMMDD/`` over
HTTPS from JSOC: ``mostrecent/`` (the newest image of each channel,
updated in real time; each is filed under every 2-min slot whose time is
within 90 s of its own T_OBS, the same tolerance QC applies), and, to fill
gaps, the near-real-time archive (1.3.1; ``nrt/YYYY/MM/DD/HHH00/``, current
to ~20 min, 3-min cadence, filed the same way) or, for an hour ``nrt/``
doesn't have, the dated archive (``YYYY/MM/DD/HHH00/``, which lags real
time by weeks). ``StagingSource`` reads ``staging/`` without network (the
reprocess command); ``ReplaySource`` reads a directory that already holds
files (validation, tests). All expose ``paths(slot)`` → {wave: Path} or
None while a channel is missing."""

from __future__ import annotations

import datetime as dt
import logging
import shutil
import time
from collections import Counter
from pathlib import Path

from .grid import (
    CADENCE,
    CHANNELS,
    day_key,
    file_name,
    file_url,
    floor_slot,
    iso,
    nrt_dir_url,
    parse_datetime,
    parse_file_name,
    parse_nrt_listing,
)

log = logging.getLogger(__name__)

MAX_OFFSET = dt.timedelta(seconds=90)  # = qc.MAX_TIME_OFFSET_S
HTTP_REPORT_EVERY = dt.timedelta(hours=1)


def slots_for(t: dt.datetime) -> list[dt.datetime]:
    """The 2-min slots an image taken at ``t`` can stand for: those whose
    time is within 90 s of it (one or two slots, nearest first)."""
    s0 = floor_slot(t)
    cands = sorted((s0, s0 + CADENCE), key=lambda s: abs(t - s))
    return [s for s in cands if abs(t - s) <= MAX_OFFSET]


class StagingSource:
    """FITS already in ``staging/YYYYMMDD/`` (the live layout), no network."""

    def __init__(self, staging_dir: Path):
        self.dir = Path(staging_dir)

    def refresh(self, now: dt.datetime) -> int:
        return 0

    def path_for(self, slot: dt.datetime, wave: int) -> Path:
        return self.dir / day_key(slot) / file_name(slot, wave)

    def paths(self, slot: dt.datetime) -> dict[int, Path] | None:
        ps = {w: self.path_for(slot, w) for w in CHANNELS}
        return ps if all(p.exists() for p in ps.values()) else None

    def missing_channels(self, slot: dt.datetime) -> list[int]:
        return [w for w in CHANNELS if not self.path_for(slot, w).exists()]


class LiveSource(StagingSource):
    def __init__(self, staging_dir: Path, base_url: str, lookback_min: int, timeout: float,
                 archive_every_sec: float = 600, archive_budget_sec: float = 20, max_sec: float = 60):
        import requests

        super().__init__(staging_dir)
        self.base = base_url.rstrip("/")
        self.lookback = dt.timedelta(minutes=lookback_min)
        self.timeout = timeout
        self.max_sec = max_sec        # whole download, headers and body (1.3.1)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "flare-nrt (HelioUSACH)"
        self.last_download: dt.datetime | None = None
        self._missed: dict[Path, dt.datetime] = {}  # failed: retry at most every 30 min
        self.archive_every = dt.timedelta(seconds=archive_every_sec)
        self.archive_budget = archive_budget_sec
        self._last_archive: dt.datetime | None = None
        # archive answers other than 200, logged at most once an hour (1.3.1)
        self._http_fails: Counter = Counter()
        self._http_example: str | None = None
        self._http_since: dt.datetime | None = None
        self._http_reported: dt.datetime | None = None

    def get(self, url: str) -> tuple[int, bytes]:
        """(HTTP status, body); the body only for 200. ``timeout`` bounds
        the connection and each read, ``max_sec`` the whole download: a
        server that trickles bytes can't hold the loop. Network errors and
        the limit raise."""
        deadline = time.monotonic() + self.max_sec
        with self.session.get(url, timeout=self.timeout, stream=True) as r:
            if r.status_code != 200:
                return r.status_code, b""
            buf = bytearray()
            for chunk in r.iter_content(1 << 16):
                buf += chunk
                if time.monotonic() > deadline:
                    raise TimeoutError(f"download over {self.max_sec:g} s: {url}")
            return 200, bytes(buf)

    def _file_under_slots(self, wave: int, t: dt.datetime, tmp: Path) -> bool:
        """Copies ``tmp`` (an image taken at ``t``) to every slot it stands
        for that has none yet. True if any was new."""
        new = False
        for slot in slots_for(t):
            dest = self.path_for(slot, wave)
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp, dest)   # a copy, not a hard link: tools that move files refuse links
            self._missed.pop(dest, None)
            new = True
        if new:
            self.last_download = dt.datetime.now(dt.timezone.utc)
        return new

    def refresh_mostrecent(self) -> int:
        """The newest image of each channel, filed under the slot(s) it stands for."""
        from astropy.io import fits

        n = 0
        for w in CHANNELS:
            url = f"{self.base}/mostrecent/AIAsynoptic{w:04d}.fits"
            got = None
            for attempt in (1, 2):  # JSOC drops a connection now and then: retry once
                try:
                    got = self.get(url)
                    break
                except Exception as e:
                    if attempt == 2:
                        log.warning("mostrecent %d failed twice: %s", w, e)
            if got is None:
                continue
            status, body = got
            if status != 200 or not body:
                log.warning("mostrecent %d: HTTP %s", w, status)
                continue
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.dir / f".mostrecent_{w}.part"
            tmp.write_bytes(body)
            try:
                with fits.open(tmp) as h:
                    hdr = next((x.header for x in h if x.data is not None), h[0].header)
                    t_obs = hdr.get("T_OBS") or hdr.get("DATE-OBS")
                t = parse_datetime(str(t_obs).replace("_TAI", ""))
            except Exception as e:
                log.warning("mostrecent %d unreadable: %s", w, e)
                tmp.unlink(missing_ok=True)
                continue
            if self._file_under_slots(w, t, tmp):
                n += 1
            tmp.unlink(missing_ok=True)
        return n

    def refresh(self, now: dt.datetime) -> int:
        """Newest images every poll; the archives only every
        ``archive_every`` and for at most ``archive_budget`` seconds, newest
        gap first. Network errors are logged and retried on a later poll,
        never raised."""
        n = self.refresh_mostrecent()
        if self._last_archive and now - self._last_archive < self.archive_every:
            return n
        self._last_archive = now
        return n + self.refresh_archive(now)

    # -- archives (1.3.1: nrt/ first, the dated archive for hours nrt/ lacks) --
    def _http_fail(self, status: int, url: str, now: dt.datetime) -> None:
        if not self._http_fails:
            self._http_since = now
        self._http_fails[status] += 1
        self._http_example = f"HTTP {status} {url}"

    def _report_http(self, now: dt.datetime) -> None:
        """Answers other than 200, counted: logged after the first pass that
        has any, then at most once an hour."""
        if not self._http_fails:
            return
        if self._http_reported and now - self._http_reported < HTTP_REPORT_EVERY:
            return
        total = sum(self._http_fails.values())
        by = ", ".join(f"{k} ×{v}" for k, v in sorted(self._http_fails.items()))
        log.warning("archive: %d answer(s) other than 200 since %s (%s), e.g. %s",
                    total, iso(self._http_since), by, self._http_example)
        self._http_fails.clear()
        self._http_reported = now

    def _listing(self, hour: dt.datetime, cache: dict, now: dt.datetime) -> dict | None:
        """{(slot, wave): (image time, URL)} for one hour of ``nrt/``, each
        slot's nearest image; None if ``nrt/`` has no such hour. Fetched
        once per pass (``cache``)."""
        if hour in cache:
            return cache[hour]
        url = nrt_dir_url(self.base, hour)
        status, body = self.get(url)
        idx = None
        if status == 200:
            idx = {}
            for t, w, name in parse_nrt_listing(body.decode("utf-8", "replace")):
                if w not in CHANNELS:
                    continue
                for s in slots_for(t):
                    old = idx.get((s, w))
                    if old is None or abs(t - s) < abs(old[0] - s):
                        idx[(s, w)] = (t, url + name)
        elif status != 404:          # 404: an hour nrt/ doesn't hold (old days) → dated archive
            self._http_fail(status, url, now)
        cache[hour] = idx
        return idx

    def _archive_source(self, slot: dt.datetime, wave: int, cache: dict, now: dt.datetime):
        """(URL, image time) to fill ``slot`` from, or None: ``nrt/`` if one
        of the hours an image within 90 s can be filed under has the slot, the
        dated archive if ``nrt/`` has none of those hours."""
        hours = sorted({(slot + d).replace(minute=0, second=0) for d in (-MAX_OFFSET, dt.timedelta(), MAX_OFFSET)})
        best, any_hour = None, False
        for h in hours:
            idx = self._listing(h, cache, now)
            if idx is None:
                continue
            any_hour = True
            hit = idx.get((slot, wave))
            if hit and (best is None or abs(hit[0] - slot) < abs(best[1] - slot)):
                best = (hit[1], hit[0])
        if best:
            return best
        if any_hour:
            return None            # nrt/ has the hour but not this image (yet): its listing is retried next pass
        return file_url(self.base, slot, wave), slot

    def refresh_archive(self, now: dt.datetime) -> int:
        n = 0
        t0 = time.monotonic()
        cache: dict = {}
        s, end = floor_slot(now - CADENCE), floor_slot(now - self.lookback)
        try:
            while s >= end:
                for w in CHANNELS:
                    if time.monotonic() - t0 > self.archive_budget:
                        return n
                    p = self.path_for(s, w)
                    if p.exists():
                        continue
                    miss = self._missed.get(p)
                    if miss and now - miss < dt.timedelta(minutes=30):
                        continue
                    try:
                        src = self._archive_source(s, w, cache, now)
                        if src is None:
                            continue
                        url, t = src
                        status, body = self.get(url)
                    except Exception as e:  # network down: try again next pass
                        log.warning("archive download failed %s: %s", p.name, e)
                        return n
                    if status == 200 and body:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        tmp = p.with_suffix(".part")
                        tmp.write_bytes(body)
                        if self._file_under_slots(w, t, tmp):
                            n += 1
                        tmp.unlink(missing_ok=True)
                    else:
                        self._http_fail(status, url, now)
                        self._missed[p] = now
                s -= CADENCE
            cutoff = now - 2 * self.lookback
            self._missed = {k: v for k, v in self._missed.items() if v > cutoff}
            return n
        finally:
            self._report_http(now)


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

    def missing_channels(self, slot: dt.datetime) -> list[int]:
        d = self.frames.get(slot) or {}
        return [w for w in CHANNELS if w not in d]
