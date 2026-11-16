"""Event replays (1.1.0; 131 Å and the peak set since 1.2.0; a Grad-CAM map
on every frame with a result and the location since 1.3.0): a short movie
of each event, kept for a year.

    python -m flare_nrt.replays                 # build every pending, incomplete or pre-1.3.0 replay
    python -m flare_nrt.replays --event 42      # one event (repeatable)
    python -m flare_nrt.replays --no-gradcam    # don't recompute missing Grad-CAM maps
    python -m flare_nrt.replays --fits-root /srv/flare-nrt/keep   # more FITS, laid out like staging/ (repeatable)

Once an event has been closed for ``replay_post_min`` (60 min), so the
frames after it exist, its replay is built once: every 2-min slot from
``t_start − replay_pre_min`` (30) to ``t_end + replay_post_min`` (60), each
as ``171.webp``, ``gradcam-heat.webp`` and (1.2.0) ``131.webp``:

- copied from this event's earlier replay when rebuilding it, or from
  ``render/`` when the live loop wrote it (171 Å and the heat map);
- else 131/171 Å rendered from the FITS still on disk (staging, or
  ``FLARE_NRT_FITS_DIRS``), exactly as ``/api/image/{wave}.png?kind=img&format=webp``
  (512 px), and the heat map from the stored Grad-CAM map
  (``gradcam/…cam.npz``);
- else the Grad-CAM is recomputed from the stored model input
  (``tensors/``) or rebuilt from the FITS with the baselines the result
  used. Same weights, same input: the same map the loop would have made.
  Since 1.3.0 the loop does this too, with the model it already holds: it
  computes a Grad-CAM for its newest frame every slot, so only the slots of
  a backlog (after a restart) need one, ~10 s each; this command does it
  for every replay it rebuilds.

Each frame with a result (1.3.0) also says whether it alerted (``alert``),
the map's maximum before normalisation (``cam_peak``, as ``X-Gradcam-Peak``)
and, for alert frames, the experimental ``location`` (the brightest region
of the 131 Å DN/s difference against the baselines the result used,
flare_nrt/localize.py): the result's own, else computed here from the FITS.
``geometry`` places helioprojective arcsec on the 512-px north-up images
(``render.image_geometry``, from that frame's 131 Å header). An event
without a location gets its peak frame's, in ``replay.json`` and in
``events.csv`` (``/api/events``); a location already there is never
changed.

Once per event (1.2.0) it also keeps the peak frame (the slot of
``t_pmax``) as the portal shows a current one: ``peak/{131,171,304,1600}.webp``
(``kind=img``, 512 px) and its Grad-CAM in both styles,
``peak/gradcam-heat.webp`` and ``peak/gradcam-overlay.webp`` (the stored
overlay PNG as WebP, as ``/api/gradcam/{ISO}.png?format=webp``).

Layout, under ``FLARE_NRT_DATA_DIR``::

    events/<event_id>/replay.json
    events/<event_id>/frames/YYYYMMDDTHHMM/{131,171,gradcam-heat}.webp
    events/<event_id>/peak/{131,171,304,1600,gradcam-heat,gradcam-overlay}.webp

``replay.json`` (schema ``flare-nrt-replay/1``; 1.2.0 only adds to it) has
the event summary, the window, one entry per frame (slot, t_obs,
probability, state, status, images), the peak set (``peak``) and says what
is missing (``complete``, ``missing``, ``note``): an event whose frames are
gone (service down, retention) still gets a replay with what exists.
Replays are kept ``replay_days`` (365) after the event's end and pruned by
flare_nrt/retention.py.

A replay is written into a temporary directory and swapped in whole, so
the API never serves half of one and the loop and this command can run at
the same time. The live loop builds replays itself (without recomputing
Grad-CAM); this command is for events already in the archive at an
upgrade: it rebuilds replays marked incomplete and those built before
1.3.0, reusing the files they already have."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import resource
import shutil
import sys
import time
from pathlib import Path

from . import __version__, render
from .artifacts import _write, find_cam, gradcam_path, gradcam_png, tensor_path
from .display import FitsLocator, render_path
from .grid import CADENCE, CHANNELS, day_key, floor_slot, iso, parse_iso, utcnow
from .localize import locate, rate_from_fits
from .store import ResultStore, fill_event_location, read_events, read_json, write_json_atomic

log = logging.getLogger("flare_nrt.replays")

SCHEMA = "flare-nrt-replay/1"
REPLAY_CHANNELS = ("171", "gradcam-heat", "131")          # 131 since 1.2.0
IMAGE_CHANNELS = {"131": 131, "171": 171}                    # frames rendered from the FITS
PEAK_WAVES = ("131", "171", "304", "1600")
PEAK_FILES = (*PEAK_WAVES, "gradcam-heat", "gradcam-overlay")   # 1.2.0


def replay_dir(cfg, event_id: int) -> Path:
    return cfg.replays_dir / str(int(event_id))


def frame_name(slot: dt.datetime) -> str:
    return f"{slot:%Y%m%dT%H%M}"


def frame_path(cfg, event_id: int, slot: dt.datetime, channel: str) -> Path:
    return replay_dir(cfg, event_id) / "frames" / frame_name(slot) / f"{channel}.webp"


def peak_path(cfg, event_id: int, name: str) -> Path:
    return replay_dir(cfg, event_id) / "peak" / f"{name}.webp"


def current(doc: dict | None) -> bool:
    """Built by 1.3.0 or later: has the 131 Å frames, the peak set, the
    per-frame Grad-CAM fields, the locations and the geometry (or says what
    is missing)."""
    return (bool(doc) and "geometry" in doc and "location" in (doc.get("peak") or {})
            and all(c in (doc.get("missing") or {}) for c in (*REPLAY_CHANNELS, "peak", "location")))


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
        self.n = 0          # maps recomputed (reported by the command)

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
            self.n += 1
        except Exception:
            log.exception("Grad-CAM recompute for %s failed", iso(slot))
            return None
        # x[0] is the 131 Å channel the model saw: the overlay's underlay
        return {"cam": cam, "peak": float(cam.max()), "probability": row.get("probability"),
                "limb": render.limb_from_header(header or {}), "base131": x[0].float().numpy()}


class FitsInfo:
    """131 Å DN/s maps and headers of the window's slots, read once per
    build from the FITS (staging, ``--fits-root``, ``FLARE_NRT_FITS_DIRS``)."""

    def __init__(self, locator: FitsLocator):
        self.loc = locator
        self._rate: dict[str, tuple] = {}
        self._hdr: dict[str, dict | None] = {}

    def rate(self, slot: dt.datetime):
        k = iso(slot)
        if k not in self._rate:
            p = self.loc.existing(slot, 131)
            try:
                self._rate[k] = rate_from_fits(p) if p is not None else (None, None)
            except Exception:
                log.exception("131 Å of %s from %s failed", k, p)
                self._rate[k] = (None, None)
            if self._rate[k][1] is not None:
                self._hdr[k] = self._rate[k][1]
        return self._rate[k][0]

    def header(self, slot: dt.datetime) -> dict | None:
        k = iso(slot)
        if k not in self._hdr:
            self._hdr[k] = None
            p = self.loc.existing(slot, 131)
            if p is not None:
                from astropy.io import fits

                try:
                    with fits.open(p) as h:
                        hdu = next((h[i] for i in (1, 0) if i < len(h) and h[i].header.get("NAXIS")), None)
                        self._hdr[k] = dict(hdu.header) if hdu is not None else None
                except Exception:
                    log.exception("header of %s failed", p)
        return self._hdr[k]

    def geometry(self, slot: dt.datetime) -> dict | None:
        h = self.header(slot)
        return render.image_geometry(h) if h else None

    def location(self, slot: dt.datetime, row: dict) -> dict | None:
        """``localize`` on the 131 Å DN/s difference against the row's baselines."""
        bl = [parse_iso(b) for b in row.get("baseline_slots") or []]
        cur = self.rate(slot)
        if cur is None or not bl:
            return None
        return locate(cur, [self.rate(b) for b in bl], self.header(slot) or {})


def is_alert_row(row: dict, threshold: float) -> bool | None:
    """The row's alert (p ≥ threshold); None without a probability."""
    if row.get("probability") is None:
        return None
    if "alert" in row:
        return bool(row["alert"])
    return row["probability"] >= row.get("threshold", threshold)


def _rows(cfg, start: dt.datetime, end: dt.datetime) -> dict[str, dict]:
    store = ResultStore(cfg.results_dir)
    days = sorted({day_key(start + dt.timedelta(days=i)) for i in range((end - start).days + 2)})
    return {r["slot"]: r for d in days for r in store.read_day(d) if start <= parse_iso(r["slot"]) <= end}


def _copy(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def _note(missing: dict, n_slots: int) -> str | None:
    parts = [f"{len(missing[c])} of {n_slots} slots have no {c} Å image" for c in ("131", "171") if missing[c]]
    if missing["gradcam-heat"]:
        parts.append(f"{len(missing['gradcam-heat'])} results have no Grad-CAM")
    if missing["peak"]:
        parts.append(f"the peak frame lacks {', '.join(missing['peak'])}")
    out = []
    if parts:
        out.append("; ".join(parts) + ": their files were gone or never made (service down, input missing, "
                   "backlog processed without Grad-CAM, FITS past their 4 days)")
    if missing.get("location"):
        out.append(f"{len(missing['location'])} alert frames have no location: the 131 Å FITS of the frame or "
                   "of its baselines were gone, or 131 Å did not brighten anywhere")
    return ". ".join(out) or None


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
    # files of this event's earlier (incomplete or pre-1.3.0) replay are reused, not rendered or recomputed again
    same = (prev_doc.get("event") or {}).get("t_start") == ev["t_start"]
    prev = final / "frames" if same else None
    prev_frames = {f["slot"]: f for f in prev_doc.get("frames") or []} if same else {}
    tmp = cfg.replays_dir / f".{eid}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    q = cfg.webp_quality
    frames, missing = [], {c: [] for c in (*REPLAY_CHANNELS, "peak", "location")}
    recomputed: dict[str, dict | None] = {}
    info = FitsInfo(loc)
    locations: dict[str, dict | None] = {}

    def location_of(slot: dt.datetime, row: dict) -> dict | None:
        """The result's own location, else this replay's earlier one, else from the FITS."""
        k = iso(slot)
        if k not in locations:
            locations[k] = (row.get("location") or (prev_frames.get(k) or {}).get("location")
                            or info.location(slot, row))
        return locations[k]

    def cam_of(slot: dt.datetime, row: dict) -> dict | None:
        """Stored map, else (this command only) recomputed once per slot."""
        cam = find_cam(cfg.gradcam_dir, slot)
        if cam is None and gradcam is not None and row.get("status") == "ok":
            if iso(slot) not in recomputed:
                recomputed[iso(slot)] = gradcam.cam(slot, row)
            cam = recomputed[iso(slot)]
        return cam

    n_slots = 0
    slot = start
    while slot <= end:
        n_slots += 1
        row = rows.get(iso(slot)) or {}
        fdir = tmp / "frames" / frame_name(slot)
        have = []
        for ch in REPLAY_CHANNELS:
            dst = fdir / f"{ch}.webp"
            try:
                if _copy(prev / frame_name(slot) / f"{ch}.webp" if prev is not None else None, dst) \
                        or _copy(render_path(cfg.render_dir, slot, ch) if ch != "131" else None, dst):
                    have.append(ch)
                elif ch in IMAGE_CHANNELS:
                    if (f := loc.existing(slot, IMAGE_CHANNELS[ch])) is not None:
                        _write(dst, render.image(f, IMAGE_CHANNELS[ch], fmt="webp", quality=q))
                        have.append(ch)
                elif (cam := cam_of(slot, row)) is not None:
                    _write(dst, render.gradcam_heat(cam["cam"], cam["limb"], "webp", q))
                    have.append(ch)
            except Exception:
                log.exception("replay %d: %s of %s failed", eid, ch, iso(slot))
            # a slot without a result has no Grad-CAM to miss
            if ch not in have and (ch != "gradcam-heat" or row.get("status") == "ok"):
                missing[ch].append(iso(slot))
        if have:
            inputs = row.get("inputs") or {}
            pf = prev_frames.get(iso(slot)) or {}
            alert = is_alert_row(row, cfg.threshold)
            cam_peak = pf.get("cam_peak")
            if cam_peak is None and row.get("status") == "ok" and (cam := cam_of(slot, row)) is not None:
                cam_peak = round(float(cam["peak"]), 6)
            location = location_of(slot, row) if alert else None
            if alert and location is None:
                missing["location"].append(iso(slot))
            frames.append({"slot": iso(slot), "t_obs": inputs.get("171"), "probability": row.get("probability"),
                           "state": row.get("state"), "status": row.get("status"), "images": have,
                           "alert": alert, "cam_peak": cam_peak, "location": location,
                           "geometry": info.geometry(slot) or pf.get("geometry")})
        slot += CADENCE

    peak = _build_peak(cfg, ev, tmp, final if same and (prev_doc.get("peak") or {}).get("slot") else None,
                       prev_doc.get("peak") or {}, rows, loc, cam_of, eid)
    missing["peak"] = [n for n in PEAK_FILES if n not in peak["images"]]
    # 1.3.0: the peak frame's location and geometry; the event's location if it had none
    pslot = parse_iso(peak["slot"])
    prow = rows.get(peak["slot"]) or {}
    prev_peak = (prev_doc.get("peak") or {}) if same else {}
    peak["location"] = location_of(pslot, prow) or prev_peak.get("location")
    peak["geometry"] = info.geometry(pslot) or prev_peak.get("geometry")
    ev_location = ev.get("location") or peak["location"]
    if not ev.get("location") and ev_location and fill_event_location(cfg.events_path, eid, ev["t_start"],
                                                                      ev_location):
        log.info("event %d: location filled from its peak frame %s", eid, peak["slot"])
    # complete: every image there (a missing location is listed and explained, not counted)
    complete = not any(v for k, v in missing.items() if k != "location")
    geometry = peak["geometry"] or next((f["geometry"] for f in frames if f.get("geometry")), None)
    doc = {
        "schema": SCHEMA,
        "event": {"event_id": eid, "t_start": ev["t_start"], "t_peak": ev["t_pmax"], "t_end": ev["t_end"],
                  "p_max": ev["p_max"], "n_alert_frames": ev.get("n_alert_frames"), "location": ev_location},
        "window": {"start": iso(start), "end": iso(end), "pre_min": cfg.replay_pre_min,
                   "post_min": cfg.replay_post_min},
        "n_slots": n_slots,
        "threshold": cfg.threshold,
        "geometry": geometry,
        "frames": frames,
        "peak": peak,
        "complete": complete,
        "missing": missing,
        "note": _note(missing, n_slots),
        "built_at": iso(now),
        "model_version": __version__,
    }
    write_json_atomic(tmp / "replay.json", doc)
    old = cfg.replays_dir / f".{eid}.old-{os.getpid()}"
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    log.info("replay of event %d: %d/%d frames, peak %d/%d files%s", eid, len(frames), n_slots,
             len(peak["images"]), len(PEAK_FILES), "" if complete else " (incomplete)")
    return doc


def _build_peak(cfg, ev: dict, tmp: Path, prev: Path | None, prev_peak: dict, rows: dict, loc: FitsLocator,
                cam_of, eid: int) -> dict:
    """The peak set (1.2.0) into ``tmp/peak/``; returns replay.json's ``peak``."""
    q = cfg.webp_quality
    slot = floor_slot(parse_iso(ev["t_pmax"]))
    row = rows.get(iso(slot)) or {}
    reuse = prev is not None and prev_peak.get("slot") == iso(slot)
    gradcam_peak = prev_peak.get("gradcam_peak") if reuse else None
    have = []
    for name in PEAK_FILES:
        dst = tmp / "peak" / f"{name}.webp"
        try:
            if reuse and _copy(prev / "peak" / f"{name}.webp", dst):
                have.append(name)
            elif name in PEAK_WAVES:
                if (f := loc.existing(slot, int(name))) is not None:
                    _write(dst, render.image(f, int(name), fmt="webp", quality=q))
                    have.append(name)
            elif name == "gradcam-heat":
                # the peak frame's own heat frame (same map, same rendering), else from the map
                if _copy(tmp / "frames" / frame_name(slot) / "gradcam-heat.webp", dst):
                    have.append(name)
                elif (cam := cam_of(slot, row)) is not None:
                    _write(dst, render.gradcam_heat(cam["cam"], cam["limb"], "webp", q))
                    have.append(name)
            else:   # gradcam-overlay: the stored overlay PNG, else drawn from a recomputed map and its input
                png = next((p for p in (gradcam_path(cfg.gradcam_dir, slot, a) for a in (True, False))
                            if p.exists()), None)
                if png is not None:
                    _write(dst, render.reencode(png.read_bytes(), "webp", q))
                    have.append(name)
                elif (cam := cam_of(slot, row)) is not None and cam.get("base131") is not None:
                    strength = (row.get("probability") or 0.0) / cfg.threshold if cfg.threshold > 0 else 1.0
                    _write(dst, render.reencode(gradcam_png(cam["cam"], cam["base131"], strength), "webp", q))
                    have.append(name)
        except Exception:
            log.exception("replay %d: peak %s of %s failed", eid, name, iso(slot))
    if gradcam_peak is None and ("gradcam-heat" in have or "gradcam-overlay" in have):
        cam = find_cam(cfg.gradcam_dir, slot) or cam_of(slot, row)
        gradcam_peak = round(float(cam["peak"]), 6) if cam is not None else None
    inputs = row.get("inputs") or {}
    return {"slot": iso(slot), "t_obs": {w: inputs.get(w) for w in PEAK_WAVES} if inputs else None,
            "probability": row.get("probability"), "gradcam_peak": gradcam_peak, "images": have}


def pending(cfg, now: dt.datetime, last_slot: dt.datetime | None = None, rebuild_incomplete: bool = False,
            only: set[int] | None = None, force: bool = False) -> list[dict]:
    """Closed events within ``replay_days`` whose replay is due: never built,
    built for another event with the same id, or (``rebuild_incomplete``, the
    one-off command) marked incomplete or built before 1.3.0."""
    cutoff = now - dt.timedelta(days=cfg.replay_days)
    out = []
    for ev in read_events(cfg.events_path):
        if only and int(ev["event_id"]) not in only:
            continue
        if parse_iso(ev["t_end"]) < cutoff or not ready(cfg, ev, now, last_slot):
            continue
        cur = read_replay(cfg, ev["event_id"])
        same = cur is not None and (cur.get("event") or {}).get("t_start") == ev["t_start"]
        if force or not same or (rebuild_incomplete and not (cur.get("complete") and current(cur))):
            out.append(ev)
    return out


class ReplayBuilder:
    """Called by the live loop after each poll; checks at most every
    ``every_sec`` and never raises."""

    def __init__(self, cfg, ensemble=None, every_sec: int = 300):
        self.cfg = cfg
        self.every = every_sec
        self._last = 0.0
        self.locator = FitsLocator(cfg)
        # 1.3.0: with the loop's model, a frame without a stored map gets one (backlog slots)
        self.gradcam = GradcamRecompute(cfg, ensemble, self.locator) if ensemble is not None else None

    def maybe_run(self, now: dt.datetime, last_slot: dt.datetime | None) -> list[int]:
        if time.time() - self._last < self.every:
            return []
        self._last = time.time()
        built = []
        try:
            for ev in pending(self.cfg, now, last_slot):
                build(self.cfg, ev, now, self.locator, self.gradcam)
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
    ap.add_argument("--fits-root", action="append", default=[], metavar="PATH",
                    help="more FITS laid out like the staging dir (PATH/YYYYMMDD/AIA….fits), "
                         "searched after it (repeatable)")
    ap.add_argument("--data-dir", help="override FLARE_NRT_DATA_DIR")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    from .config import Settings

    cfg = Settings()
    if a.data_dir:
        cfg.data_dir = Path(a.data_dir)
    roots = [Path(r) for r in a.fits_root]
    for r in roots:
        if not r.is_dir():
            log.warning("--fits-root %s: not a directory", r)
    cfg.fits_roots = [*roots, *cfg.fits_roots]
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
            t0, n0 = time.perf_counter(), gradcam.n if gradcam else 0
            doc = build(cfg, ev, now, loc, gradcam)
            m = doc["missing"]
            lost = {k: len(v) for k, v in m.items() if v and k != "peak"}
            print(f"event {doc['event']['event_id']}: {len(doc['frames'])}/{doc['n_slots']} frames, "
                  f"{'complete' if doc['complete'] else 'incomplete'}, "
                  f"location {'yes' if doc['event']['location'] else 'none'}, "
                  f"{(gradcam.n if gradcam else 0) - n0} Grad-CAM maps recomputed, "
                  f"{time.perf_counter() - t0:.1f} s, peak RSS "
                  f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB"
                  + (f"; missing: {lost}" if lost else "") + (f", peak {m['peak']}" if m["peak"] else ""))
        except Exception:
            log.exception("replay of event %s failed", ev["event_id"])
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
