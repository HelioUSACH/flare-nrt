"""Event replays (1.1.0): a short movie of each event, kept for a year.

    python -m flare_nrt.replays                 # build every pending or incomplete replay
    python -m flare_nrt.replays --event 42      # one event (repeatable)
    python -m flare_nrt.replays --no-gradcam    # don't recompute missing Grad-CAM maps

Once an event has been closed for ``replay_post_min`` (60 min), so the
frames after it exist, its replay is built once: every 2-min slot from
``t_start − replay_pre_min`` (30) to ``t_end + replay_post_min`` (60), each
as ``171.webp`` and ``gradcam-heat.webp``:

- copied from ``render/`` when the live loop wrote it (or from this
  event's earlier, incomplete replay when rebuilding it);
- else 171 Å rendered from the FITS still on disk (staging, or
  ``FLARE_NRT_FITS_DIRS``), and the heat map from the stored Grad-CAM map
  (``gradcam/…cam.npz``);
- else, only in this command (not in the loop, where it would hold the
  stream back ~10 s per frame), the Grad-CAM is recomputed from the stored
  model input (``tensors/``) or rebuilt from the FITS with the baselines the
  result used. Same weights, same input: the same map the loop would have
  made.

Layout, under ``FLARE_NRT_DATA_DIR``::

    events/<event_id>/replay.json
    events/<event_id>/frames/YYYYMMDDTHHMM/{171,gradcam-heat}.webp

``replay.json`` (schema ``flare-nrt-replay/1``) has the event summary, the
window, one entry per frame (slot, t_obs, probability, state, status,
images) and says what is missing (``complete``, ``missing``, ``note``):
an event whose frames are gone (service down, retention) still gets a
replay with what exists. Replays are kept ``replay_days`` (365) after the
event's end and pruned by flare_nrt/retention.py.

A replay is written into a temporary directory and swapped in whole, so
the API never serves half of one and the loop and this command can run at
the same time. The live loop builds replays itself (without recomputing
Grad-CAM); this command is for events already in the archive at the
upgrade, and it also rebuilds replays marked incomplete."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from . import __version__, render
from .artifacts import _write, find_cam, tensor_path
from .display import FitsLocator, render_path
from .grid import CADENCE, CHANNELS, day_key, floor_slot, iso, parse_iso, utcnow
from .store import ResultStore, read_events, read_json, write_json_atomic

log = logging.getLogger("flare_nrt.replays")

SCHEMA = "flare-nrt-replay/1"
REPLAY_CHANNELS = ("171", "gradcam-heat")


def replay_dir(cfg, event_id: int) -> Path:
    return cfg.replays_dir / str(int(event_id))


def frame_name(slot: dt.datetime) -> str:
    return f"{slot:%Y%m%dT%H%M}"


def frame_path(cfg, event_id: int, slot: dt.datetime, channel: str) -> Path:
    return replay_dir(cfg, event_id) / "frames" / frame_name(slot) / f"{channel}.webp"


def read_replay(cfg, event_id: int) -> dict | None:
    return read_json(replay_dir(cfg, event_id) / "replay.json")


def window(cfg, ev: dict) -> tuple[dt.datetime, dt.datetime]:
    start = floor_slot(parse_iso(ev["t_start"]) - dt.timedelta(minutes=cfg.replay_pre_min))
    end = floor_slot(parse_iso(ev["t_end"]) + dt.timedelta(minutes=cfg.replay_post_min))
    return start, end


def ready(cfg, ev: dict, now: dt.datetime, last_slot: dt.datetime | None = None) -> bool:
    """Closed (no alert within ``merge_gap_min`` of its last one) for
    ``replay_post_min``, and the loop has processed up to t_end + post."""
    t_end = parse_iso(ev["t_end"])
    post = dt.timedelta(minutes=cfg.replay_post_min)
    if now < t_end + dt.timedelta(minutes=cfg.merge_gap_min) + post:
        return False
    return last_slot is None or last_slot >= t_end + post


class GradcamRecompute:
    """Grad-CAM of a past result, for the one-off command: the stored model
    input if still there (2 days), else rebuilt from the FITS with the
    baselines that result used."""

    def __init__(self, cfg, ensemble, locator: FitsLocator):
        from .preprocess import load_norm_stats

        self.cfg, self.ens, self.loc = cfg, ensemble, locator
        self.v_high = load_norm_stats(cfg.norm_stats)

    def _current(self, slot: dt.datetime):
        from .preprocess import build_current
        from .qc import read_frame

        paths = {w: self.loc.existing(slot, w) for w in CHANNELS}
        if any(p is None for p in paths.values()):
            return None, None
        frames = {w: read_frame(p) for w, p in paths.items()}
        return build_current({w: f.data for w, f in frames.items()}, self.v_high), frames[131].header

    def cam(self, slot: dt.datetime, row: dict) -> dict | None:
        import numpy as np
        import torch

        from .preprocess import build_input
        from .qc import read_frame

        try:
            header = None
            p131 = self.loc.existing(slot, 131)
            if p131 is not None:
                header = read_frame(p131).header
            tp = tensor_path(self.cfg.tensors_dir, slot)
            if tp.exists():
                with np.load(tp) as z:
                    x = torch.from_numpy(z["x"])
            else:
                cur, header = self._current(slot)
                bls = [self._current(parse_iso(b))[0] for b in row.get("baseline_slots") or []]
                if cur is None or not bls or any(b is None for b in bls):
                    return None
                x = build_input(cur, bls)
            cam = self.ens.gradcam(x).numpy()
        except Exception:
            log.exception("Grad-CAM recompute for %s failed", iso(slot))
            return None
        return {"cam": cam, "peak": float(cam.max()), "probability": row.get("probability"),
                "limb": render.limb_from_header(header or {})}


def _rows(cfg, start: dt.datetime, end: dt.datetime) -> dict[str, dict]:
    store = ResultStore(cfg.results_dir)
    days = sorted({day_key(start + dt.timedelta(days=i)) for i in range((end - start).days + 2)})
    return {r["slot"]: r for d in days for r in store.read_day(d) if start <= parse_iso(r["slot"]) <= end}


def build(cfg, ev: dict, now: dt.datetime | None = None, locator: FitsLocator | None = None,
          gradcam: GradcamRecompute | None = None) -> dict:
    """Build (or rebuild) one event's replay; returns its replay.json."""
    now = now or utcnow()
    loc = locator or FitsLocator(cfg)
    eid = int(ev["event_id"])
    start, end = window(cfg, ev)
    rows = _rows(cfg, start, end)
    final = replay_dir(cfg, eid)
    prev_doc = read_json(final / "replay.json") or {}
    # frames of this event's earlier (incomplete) replay are reused, not rendered or recomputed again
    prev = final / "frames" if (prev_doc.get("event") or {}).get("t_start") == ev["t_start"] else None
    tmp = cfg.replays_dir / f".{eid}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    q = cfg.webp_quality
    frames, missing = [], {c: [] for c in REPLAY_CHANNELS}
    n_slots = 0
    slot = start
    while slot <= end:
        n_slots += 1
        row = rows.get(iso(slot)) or {}
        fdir = tmp / "frames" / frame_name(slot)
        have = []
        src = render_path(cfg.render_dir, slot, "171")
        if not src.exists() and prev is not None:
            src = prev / frame_name(slot) / "171.webp"
        try:
            if src.exists():
                fdir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, fdir / "171.webp")
                have.append("171")
            elif (f := loc.existing(slot, 171)) is not None:
                _write(fdir / "171.webp", render.image(f, 171, fmt="webp", quality=q))
                have.append("171")
        except Exception:
            log.exception("replay %d: 171 Å of %s failed", eid, iso(slot))
        if "171" not in have:
            missing["171"].append(iso(slot))
        src = render_path(cfg.render_dir, slot, "gradcam-heat")
        if not src.exists() and prev is not None:
            src = prev / frame_name(slot) / "gradcam-heat.webp"
        try:
            if src.exists():
                fdir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, fdir / "gradcam-heat.webp")
                have.append("gradcam-heat")
            else:
                cam = find_cam(cfg.gradcam_dir, slot)
                if cam is None and gradcam is not None and row.get("status") == "ok":
                    cam = gradcam.cam(slot, row)
                if cam is not None:
                    _write(fdir / "gradcam-heat.webp", render.gradcam_heat(cam["cam"], cam["limb"], "webp", q))
                    have.append("gradcam-heat")
        except Exception:
            log.exception("replay %d: Grad-CAM of %s failed", eid, iso(slot))
        if "gradcam-heat" not in have and row.get("status") == "ok":
            missing["gradcam-heat"].append(iso(slot))
        if have:
            inputs = row.get("inputs") or {}
            frames.append({"slot": iso(slot), "t_obs": inputs.get("171"), "probability": row.get("probability"),
                           "state": row.get("state"), "status": row.get("status"), "images": have})
        slot += CADENCE
    complete = not any(missing.values())
    note = None
    if not complete:
        note = (f"{len(missing['171'])} of {n_slots} slots have no 171 Å image and "
                f"{len(missing['gradcam-heat'])} results have no Grad-CAM: their files were gone or never "
                "made (service down, input missing, backlog processed without Grad-CAM)")
    doc = {
        "schema": SCHEMA,
        "event": {"event_id": eid, "t_start": ev["t_start"], "t_peak": ev["t_pmax"], "t_end": ev["t_end"],
                  "p_max": ev["p_max"], "n_alert_frames": ev.get("n_alert_frames"), "location": ev.get("location")},
        "window": {"start": iso(start), "end": iso(end), "pre_min": cfg.replay_pre_min,
                   "post_min": cfg.replay_post_min},
        "n_slots": n_slots,
        "frames": frames,
        "complete": complete,
        "missing": missing,
        "note": note,
        "built_at": iso(now),
        "model_version": __version__,
    }
    write_json_atomic(tmp / "replay.json", doc)
    old = cfg.replays_dir / f".{eid}.old-{os.getpid()}"
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    log.info("replay of event %d: %d/%d frames%s", eid, len(frames), n_slots, "" if complete else " (incomplete)")
    return doc


def pending(cfg, now: dt.datetime, last_slot: dt.datetime | None = None, rebuild_incomplete: bool = False,
            only: set[int] | None = None, force: bool = False) -> list[dict]:
    """Closed events within ``replay_days`` whose replay is due: never built,
    built for another event with the same id, or (``rebuild_incomplete``)
    marked incomplete."""
    cutoff = now - dt.timedelta(days=cfg.replay_days)
    out = []
    for ev in read_events(cfg.events_path):
        if only and int(ev["event_id"]) not in only:
            continue
        if parse_iso(ev["t_end"]) < cutoff or not ready(cfg, ev, now, last_slot):
            continue
        cur = read_replay(cfg, ev["event_id"])
        same = cur is not None and (cur.get("event") or {}).get("t_start") == ev["t_start"]
        if force or not same or (rebuild_incomplete and not cur.get("complete")):
            out.append(ev)
    return out


class ReplayBuilder:
    """Called by the live loop after each poll; checks at most every
    ``every_sec`` and never raises."""

    def __init__(self, cfg, every_sec: int = 300):
        self.cfg = cfg
        self.every = every_sec
        self._last = 0.0
        self.locator = FitsLocator(cfg)

    def maybe_run(self, now: dt.datetime, last_slot: dt.datetime | None) -> list[int]:
        if time.time() - self._last < self.every:
            return []
        self._last = time.time()
        built = []
        try:
            for ev in pending(self.cfg, now, last_slot):
                build(self.cfg, ev, now, self.locator)
                built.append(int(ev["event_id"]))
        except Exception:
            log.exception("building event replays failed")
        return built


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", type=int, action="append", help="only this event id (repeatable)")
    ap.add_argument("--no-gradcam", action="store_true",
                    help="don't load the model to recompute Grad-CAM maps that are no longer stored")
    ap.add_argument("--force", action="store_true", help="rebuild even complete replays")
    ap.add_argument("--data-dir", help="override FLARE_NRT_DATA_DIR")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    from .config import Settings

    cfg = Settings()
    if a.data_dir:
        cfg.data_dir = Path(a.data_dir)
    cfg.ensure_dirs()
    now = utcnow()
    todo = pending(cfg, now, None, rebuild_incomplete=True, only=set(a.event or []), force=a.force)
    log.info("%d replay(s) to build", len(todo))
    if not todo:
        return 0
    loc = FitsLocator(cfg)
    gradcam = None
    if not a.no_gradcam:
        import torch

        from .model import Ensemble

        torch.set_num_threads(cfg.torch_threads)
        gradcam = GradcamRecompute(cfg, Ensemble.load(cfg.weights, cfg.weights_sha256), loc)
    rc = 0
    for ev in todo:
        try:
            doc = build(cfg, ev, now, loc, gradcam)
            print(f"event {doc['event']['event_id']}: {len(doc['frames'])}/{doc['n_slots']} frames, "
                  f"{'complete' if doc['complete'] else 'incomplete'}")
        except Exception:
            log.exception("replay of event %s failed", ev["event_id"])
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
