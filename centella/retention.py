"""Bounded disk use (checklist F.26). Runs from the loop once an hour and
from the curation job.

Rules, in order:

1. ``staging/YYYYMMDD`` (raw downloads) is deleted once that day has been
   curated (``archive/…/day.json`` exists) and is older than
   ``staging_days``; an uncurated day is kept up to ``staging_days + 4``
   so a failed curation can be retried, then deleted anyway.
2. ``archive/YYYY/MM/DD`` is deleted when older than
   ``archive_local_days`` — the long-term archive is the off-VM copy
   (docs/archive-format.md, "Off-VM copy").
3. ``results/YYYYMMDD.jsonl`` older than ``results_days`` is deleted
   (~0.5 MB/day).
4. ``staging_hours`` > 0 (a small disk, no curation): raw files older than
   that many hours are deleted, whatever day they belong to.
5. If free space is below ``min_free_gb``: oldest archive days, then
   oldest staging days, until it isn't — never the last 24 h of staging,
   which the baselines and today's curation need.
"""

from __future__ import annotations

import datetime as dt
import logging
import shutil
import time
from pathlib import Path

from .grid import parse_file_name, utcnow

log = logging.getLogger(__name__)


def _day(name: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(name, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def archive_day_dir(archive: Path, day: dt.datetime) -> Path:
    return archive / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"


def is_curated(archive: Path, day: dt.datetime) -> bool:
    return (archive_day_dir(archive, day) / "day.json").exists()


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


class Retention:
    def __init__(self, cfg, every_sec: int = 3600):
        self.cfg = cfg
        self.every = every_sec
        self._last = 0.0

    def maybe_run(self) -> None:
        if time.time() - self._last >= self.every:
            self._last = time.time()
            try:
                self.run()
            except Exception:
                log.exception("retention failed")

    def _staging_days(self):
        out = []
        for p in sorted(self.cfg.staging_dir.glob("*")):
            d = _day(p.name)
            if p.is_dir() and d:
                out.append((d, p))
        return out

    def _archive_days(self):
        out = []
        for p in sorted(self.cfg.archive_dir.glob("*/*/*")):
            try:
                d = dt.datetime(int(p.parent.parent.name), int(p.parent.name), int(p.name), tzinfo=dt.timezone.utc)
            except ValueError:
                continue
            out.append((d, p))
        return out

    def run(self, now: dt.datetime | None = None) -> list[str]:
        cfg, now = self.cfg, now or utcnow()
        removed = []
        day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
        for d, p in self._staging_days():
            age = (day0 - d).total_seconds() / 86400
            curated = is_curated(cfg.archive_dir, d)
            if (curated and age > cfg.staging_days) or age > cfg.staging_days + 4:
                if not curated:
                    log.warning("deleting uncurated staging day %s", p.name)
                shutil.rmtree(p, ignore_errors=True)
                removed.append(str(p))
        if cfg.staging_hours > 0:
            cutoff = now - dt.timedelta(hours=cfg.staging_hours)
            for p in cfg.staging_dir.glob("*/AIA*.fits"):
                parsed = parse_file_name(p.name)
                if parsed and parsed[0] < cutoff:
                    p.unlink(missing_ok=True)
                    removed.append(str(p))
        for d, p in self._archive_days():
            if (day0 - d).total_seconds() / 86400 > cfg.archive_local_days:
                shutil.rmtree(p, ignore_errors=True)
                removed.append(str(p))
        for p in cfg.results_dir.glob("*.jsonl"):
            d = _day(p.stem)
            if d and (day0 - d).total_seconds() / 86400 > cfg.results_days:
                p.unlink(missing_ok=True)
                removed.append(str(p))
        # disk guard
        victims = [p for _, p in self._archive_days()] + \
                  [p for d, p in self._staging_days() if now - d > dt.timedelta(days=2)]
        while cfg.data_dir.exists() and free_gb(cfg.data_dir) < cfg.min_free_gb and victims:
            p = victims.pop(0)
            log.warning("low disk (%.1f GB free): deleting %s", free_gb(cfg.data_dir), p)
            shutil.rmtree(p, ignore_errors=True)
            removed.append(str(p))
        if removed:
            log.info("retention removed %d path(s), oldest %s", len(removed), min(removed))
        return removed
