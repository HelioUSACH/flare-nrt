"""Daily curation: turn one UTC day of raw staging files + loop results into
an archive day for retraining (format: docs/archive-format.md), then let
retention free the staging copy.

    python -m centella.curate                    # every uncurated day older than 1 day
    python -m centella.curate --day 2026-10-25   # one day
    python -m centella.curate --flares-csv F.csv # labels from a file instead of SWPC

What is kept (full 4-channel slots, raw FITS):

- **flare windows**: every GOES flare ≥ ``min_flare_class`` overlapping the
  day, [tstart − pre_flare_min, tend + post_flare_min];
- **alert windows**: every run of the loop's alerts, same padding — false
  alarms are the hardest negatives the model has;
- **negative blocks**: ``neg_blocks_per_day`` contiguous blocks of
  ``neg_block_min`` minutes, ≥60 min from any flare window, chosen with a
  seed fixed by the date (re-running gives the same blocks).

Windows are contiguous on purpose: a training pair can then be built with
any baseline age up to ``pre_flare_min`` for positives *and* negatives,
which is what fixes the baseline-age shortcut of the current model.

Every slot of the day (kept or not) gets a row in ``frames.csv`` with its
QC status, the loop's output and its GOES label, so the archive also
serves the operational validation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import random
import shutil
import sys
from pathlib import Path

from . import __version__
from .config import Settings
from .goes import Flare, class_flux, fetch_swpc, read_csv, write_csv
from .grid import CADENCE, CHANNELS, file_name, iso, parse_iso, utcnow
from .model import sha256
from .retention import Retention, archive_day_dir, is_curated
from .store import ResultStore, write_json_atomic

log = logging.getLogger("centella.curate")

SCHEMA = "centella-archive/1"
NEG_CLEARANCE = dt.timedelta(minutes=60)
FRAME_FIELDS = [
    "slot", "status", "qc", "probability", "alert", "event_id", "model_version", "lag_min",
    "threshold", "keep", "keep_reason", "files_complete", "flare_id", "flare_class",
    "flare_flux", "phase", "min_from_start", "min_from_peak",
]

Interval = tuple[dt.datetime, dt.datetime]


def day_slots(day: dt.datetime) -> list[dt.datetime]:
    return [day + i * CADENCE for i in range(720)]


def merge(intervals: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def inside(t: dt.datetime, intervals: list[Interval]) -> bool:
    return any(a <= t <= b for a, b in intervals)


def alert_runs(rows: list[dict], gap: dt.timedelta) -> list[Interval]:
    ts = sorted(parse_iso(r["slot"]) for r in rows if r.get("alert"))
    runs: list[Interval] = []
    for t in ts:
        if runs and t - runs[-1][1] <= gap:
            runs[-1] = (runs[-1][0], t)
        else:
            runs.append((t, t))
    return runs


def negative_blocks(day: dt.datetime, forbidden: list[Interval], available: set[dt.datetime],
                    n: int, block_min: int) -> list[Interval]:
    """Deterministic (seeded by the date) non-overlapping blocks that stay
    ``NEG_CLEARANCE`` away from every forbidden interval and have files for
    ≥ 80 % of their slots."""
    rng = random.Random(int(f"{day:%Y%m%d}"))
    block = dt.timedelta(minutes=block_min)
    bad = merge([(a - NEG_CLEARANCE, b + NEG_CLEARANCE) for a, b in forbidden])
    starts = [s for s in day_slots(day) if s + block <= day + dt.timedelta(days=1)]
    rng.shuffle(starts)
    chosen: list[Interval] = []
    need = 0.8 * block_min / 2
    for s in starts:
        if len(chosen) >= n:
            break
        iv = (s, s + block - CADENCE)
        if any(a <= iv[1] and b >= iv[0] for a, b in bad + chosen):
            continue
        n_av = sum(1 for k in range(block_min // 2) if s + k * CADENCE in available)
        if n_av >= need:
            chosen.append(iv)
    return sorted(chosen)


def label(t: dt.datetime, flares: list[Flare], pre: dt.timedelta, post: dt.timedelta) -> dict:
    """GOES label of one slot: the biggest flare whose padded window holds it."""
    best, best_flux = None, -1.0
    for f in flares:
        if f.tstart - pre <= t <= f.tend + post and f.flux > best_flux:
            best, best_flux = f, f.flux
    if best is None:
        return {"flare_id": "", "flare_class": "", "flare_flux": "", "phase": "none",
                "min_from_start": "", "min_from_peak": ""}
    if t < best.tstart:
        phase = "pre"
    elif t <= best.tpeak:
        phase = "rise"
    elif t <= best.tend:
        phase = "decay"
    else:
        phase = "post"
    return {"flare_id": best.flare_id, "flare_class": best.fclass, "flare_flux": f"{best.flux:.3e}",
            "phase": phase,
            "min_from_start": round((t - best.tstart).total_seconds() / 60, 1),
            "min_from_peak": round((t - best.tpeak).total_seconds() / 60, 1)}


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def curate_day(cfg: Settings, day: dt.datetime, flares: list[Flare], labels_source: str,
               labels_complete: bool) -> dict:
    day = day.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=dt.UTC)
    nxt = day + dt.timedelta(days=1)
    pre, post = dt.timedelta(minutes=cfg.pre_flare_min), dt.timedelta(minutes=cfg.post_flare_min)
    rows = {r["slot"]: r for r in ResultStore(cfg.results_dir).read_day(f"{day:%Y%m%d}")}
    staging = cfg.staging_dir / f"{day:%Y%m%d}"
    have = {s for s in day_slots(day)
            if all((staging / file_name(s, w)).exists() for w in CHANNELS)}

    min_flux = class_flux(cfg.min_flare_class)
    fl_day = [f for f in flares if f.tend + post >= day and f.tstart - pre < nxt]
    fl_keep = [f for f in fl_day if f.flux >= min_flux]
    flare_win = merge([(f.tstart - pre, f.tend + post) for f in fl_keep])
    alert_win = merge([(a - pre, b + post)
                       for a, b in alert_runs(list(rows.values()), dt.timedelta(minutes=cfg.merge_gap_min))])
    neg_win = negative_blocks(day, flare_win + alert_win, have, cfg.neg_blocks_per_day, cfg.neg_block_min)

    out_dir = archive_day_dir(cfg.archive_dir, day)
    fits_dir = out_dir / "fits"
    files: dict[str, str] = {}
    frames = []
    reasons: dict[str, int] = {}
    for s in day_slots(day):
        r = rows.get(iso(s), {})
        why = ("flare" if inside(s, flare_win) else "alert" if inside(s, alert_win)
               else "negative" if inside(s, neg_win) else "")
        keep = bool(why) and s in have
        if keep:
            for w in CHANNELS:
                name = file_name(s, w)
                _link_or_copy(staging / name, fits_dir / name)
                files[name] = sha256(fits_dir / name)
            reasons[why] = reasons.get(why, 0) + 1
        frames.append({
            "slot": iso(s), "status": r.get("status", "not_processed"),
            "qc": ";".join(r.get("qc", [])), "probability": r.get("probability", ""),
            "alert": int(bool(r.get("alert"))), "event_id": r.get("event_id") or "",
            "model_version": r.get("model_version", ""), "lag_min": r.get("lag_min", ""),
            "threshold": r.get("threshold", ""), "keep": int(keep), "keep_reason": why if keep else "",
            "files_complete": int(s in have), **label(s, fl_day, pre, post),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FRAME_FIELDS)
        w.writeheader()
        w.writerows(frames)
    write_csv(out_dir / "flares.csv", fl_day)
    summary = {
        "schema": SCHEMA, "day": f"{day:%Y-%m-%d}", "generated_at": iso(utcnow()),
        "curator_version": __version__,
        "model_versions": sorted({r.get("model_version") for r in rows.values() if r.get("model_version")}),
        "settings": {"pre_flare_min": cfg.pre_flare_min, "post_flare_min": cfg.post_flare_min,
                     "neg_blocks_per_day": cfg.neg_blocks_per_day, "neg_block_min": cfg.neg_block_min,
                     "min_flare_class": cfg.min_flare_class},
        "labels": {"source": labels_source, "complete": labels_complete, "n_flares": len(fl_day),
                   "by_letter": {k: sum(f.letter == k for f in fl_day) for k in "ABCMX"}},
        "slots": {"total": 720, "files_complete": len(have),
                  "processed": sum(1 for f in frames if f["status"] != "not_processed"),
                  "ok": sum(1 for f in frames if f["status"] == "ok"),
                  "qc_reject": sum(1 for f in frames if f["status"] == "qc_reject"),
                  "kept": sum(f["keep"] for f in frames), "kept_by_reason": reasons},
        "windows": {"flare": [[iso(a), iso(b)] for a, b in flare_win],
                    "alert": [[iso(a), iso(b)] for a, b in alert_win],
                    "negative": [[iso(a), iso(b)] for a, b in neg_win]},
        "files": files,
    }
    write_json_atomic(out_dir / "day.json", summary)  # written last: marks the day as curated
    return summary


def pending_days(cfg: Settings, now: dt.datetime) -> list[dt.datetime]:
    out = []
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for p in sorted(cfg.staging_dir.glob("*")):
        try:
            d = dt.datetime.strptime(p.name, "%Y%m%d").replace(tzinfo=dt.UTC)
        except ValueError:
            continue
        if d < today and not is_curated(cfg.archive_dir, d):
            out.append(d)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", help="YYYY-MM-DD (default: every uncurated day before today)")
    ap.add_argument("--flares-csv", help="labels from a CSV instead of SWPC")
    ap.add_argument("--force", action="store_true", help="re-curate a day that already has day.json")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    cfg = Settings()
    cfg.ensure_dirs()
    now = utcnow()
    days = [dt.datetime.strptime(a.day, "%Y-%m-%d").replace(tzinfo=dt.UTC)] if a.day else pending_days(cfg, now)
    if not a.force:
        days = [d for d in days if not is_curated(cfg.archive_dir, d)]
    if not days:
        log.info("nothing to curate")
        Retention(cfg).run()
        return 0
    if a.flares_csv:
        flares, source = read_csv(Path(a.flares_csv)), f"file:{Path(a.flares_csv).name}"
    else:
        try:
            flares, source = fetch_swpc(cfg.goes_url), cfg.goes_url
        except Exception as e:  # keep staging; the timer retries tomorrow
            log.error("GOES flare list unavailable (%s); nothing curated", e)
            return 1
    rc = 0
    for d in days:
        # SWPC's list spans 7 days: older days are curated with labels marked incomplete
        complete = bool(a.flares_csv) or (now - d) <= dt.timedelta(days=6)
        try:
            s = curate_day(cfg, d, flares, source, complete)
            log.info("curated %s: %d/%d slots kept %s, %d flares", s["day"], s["slots"]["kept"], 720,
                     s["slots"]["kept_by_reason"], s["labels"]["n_flares"])
        except Exception:
            log.exception("curation of %s failed", f"{d:%Y-%m-%d}")
            rc = 1
    Retention(cfg).run()
    return rc


if __name__ == "__main__":
    sys.exit(main())
