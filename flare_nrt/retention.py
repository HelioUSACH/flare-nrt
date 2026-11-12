"""Bounded disk use (checklist F.26). Runs from the loop (hourly; every
10 min with ``staging_hours``) and from the curation job.

Rules:

1. Raw AIA FITS (``staging/YYYYMMDD/``) older than ``staging_days`` (4 ×
   24 h, by slot time) are deleted; curation (daily) has copied what is
   worth keeping to the archive long before.
2. ``staging_hours`` > 0 (a small disk) replaces that age with N hours.
3. ``tensors/YYYYMMDD`` (model inputs) older than ``tensors_days`` (2).
4. ``gradcam/YYYYMMDD/HHMM.png`` older than ``gradcam_days`` (2);
   ``HHMM_alert.png`` older than ``gradcam_alert_days`` (400).
   ``HHMM[_alert].cam.npz`` (the Grad-CAM map, 1.1.0) follow their PNG.
5. ``cache/*.png``, ``cache/*.webp`` (API previews) older than
   ``image_cache_days`` (1); the API prunes these itself too.
6. ``archive/YYYY/MM/DD`` older than ``archive_local_days`` (7): the
   long-term archive is the off-server copy (docs/archive-format.md).
7. ``results/YYYYMMDD.jsonl`` older than ``results_days`` (400, ~1 MB/day),
   and ``reports/*.json`` (weekly validation) older than that too.
8. ``render/YYYYMMDD/HHMM/`` (pre-rendered WebP, 1.1.0) with the raw FITS:
   older than ``staging_days`` (or ``staging_hours``), by slot time.
9. ``events/<event_id>/`` (event replays, 1.1.0) whose event ended more
   than ``replay_days`` (365) ago; leftovers of an interrupted build
   (``events/.<id>.tmp-*``) after a day.
10. If free space is below ``min_free_gb``: tensors, archive days, then
   staging days, oldest first, until it isn't — never the last 2 days of
   staging, which the baselines and the day's curation need.

Steady state with the defaults: see INTEGRATION.md "Resources".
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

    def _dated_dirs(self, root: Path):
        return [(d, p) for p in sorted(root.glob("*")) if p.is_dir() and (d := _day(p.name))]

    def _prune_replays(self, now: dt.datetime, rm) -> None:
        from .grid import parse_iso
        from .store import read_json

        cut = now - dt.timedelta(days=self.cfg.replay_days)
        for p in self.cfg.replays_dir.glob("*"):
            if not p.is_dir():
                continue
            if p.name.startswith("."):
                if time.time() - p.stat().st_mtime > 86400:
                    rm(p)
                continue
            doc = read_json(p / "replay.json") or {}
            try:
                t_end = parse_iso(doc["event"]["t_end"])
            except (KeyError, TypeError, ValueError):
                t_end = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc)
            if t_end < cut:
                rm(p)

    def run(self, now: dt.datetime | None = None) -> list[str]:
        cfg, now = self.cfg, now or utcnow()
        removed: list[str] = []
        day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)

        def age_days(d: dt.datetime) -> float:
            return (day0 - d).total_seconds() / 86400

        def rm(p: Path) -> None:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            removed.append(str(p))

        hours = cfg.staging_hours if cfg.staging_hours > 0 else cfg.staging_days * 24
        cutoff = now - dt.timedelta(hours=hours)
        for d, p in self._staging_days():
            if d + dt.timedelta(days=1) <= cutoff:          # the whole day is past the cutoff
                if not is_curated(cfg.archive_dir, d) and cfg.staging_hours <= 0:
                    log.warning("deleting uncurated staging day %s", p.name)
                rm(p)
            elif d < cutoff:                                # the day the cutoff falls in: per file
                for f in p.glob("AIA*.fits"):
                    parsed = parse_file_name(f.name)
                    if parsed and parsed[0] < cutoff:
                        rm(f)
        def prune_by_slot(root: Path, days: float, keep=lambda f: False) -> None:
            """root/YYYYMMDD/HHMM*.ext: delete files whose slot is older than ``days``."""
            cut = now - dt.timedelta(days=days)
            for d, p in self._dated_dirs(root):
                if d + dt.timedelta(days=1) <= cut and not any(keep(f) for f in p.iterdir()):
                    rm(p)
                elif d < cut:
                    for f in p.iterdir():
                        try:
                            t = d.replace(hour=int(f.name[:2]), minute=int(f.name[2:4]))
                        except ValueError:
                            continue
                        if t < cut and not keep(f):
                            rm(f)

        prune_by_slot(cfg.tensors_dir, cfg.tensors_days)
        prune_by_slot(cfg.gradcam_dir, cfg.gradcam_days, keep=lambda f: "_alert." in f.name)
        prune_by_slot(cfg.gradcam_dir, cfg.gradcam_alert_days)
        prune_by_slot(cfg.render_dir, hours / 24)
        for f in [*cfg.cache_dir.glob("*.png"), *cfg.cache_dir.glob("*.webp")]:
            if time.time() - f.stat().st_mtime > cfg.image_cache_days * 86400:
                rm(f)
        self._prune_replays(now, rm)
        for d, p in self._archive_days():
            if age_days(d) > cfg.archive_local_days:
                rm(p)
        for p in cfg.results_dir.glob("*.jsonl"):
            d = _day(p.stem)
            if d and age_days(d) > cfg.results_days:
                rm(p)
        for p in (cfg.data_dir / "reports").glob("*.json"):
            if time.time() - p.stat().st_mtime > cfg.results_days * 86400:
                rm(p)
        # disk guard
        victims = [p for _, p in self._dated_dirs(cfg.tensors_dir)] + \
                  [p for _, p in self._archive_days()] + \
                  [p for d, p in self._staging_days() if now - d > dt.timedelta(days=2)]
        while cfg.data_dir.exists() and free_gb(cfg.data_dir) < cfg.min_free_gb and victims:
            p = victims.pop(0)
            log.warning("low disk (%.1f GB free): deleting %s", free_gb(cfg.data_dir), p)
            rm(p)
        if removed:
            log.info("retention removed %d path(s), oldest %s", len(removed), min(removed))
        return removed
