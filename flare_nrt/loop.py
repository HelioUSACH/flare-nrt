"""The detection loop: every poll, download what JSOC has published, then
process every slot in order, exactly once.

    python -m flare_nrt.loop                 # live (systemd: deploy/flare-nrt-loop.service)
    python -m flare_nrt.loop --replay DIR    # existing FITS, same code path

Per slot: QC → preprocess → baseline from t−L…(n_baseline frames) → ensemble
→ alert rule → event merging → results/YYYYMMDD.jsonl (+ status.json for
the API). A slot whose files haven't arrived is waited for up to
``give_up_min``, then recorded as ``missing``.

Since 1.3.1 the loop says when it stops producing results
(``flare_nrt/streak.py``: a WARNING after 30 min without one, then hourly,
an INFO when results resume) and tells systemd it is alive
(``flare_nrt/sdnotify.py``, ``Type=notify`` + ``WatchdogSec``)."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import signal
import statistics
import sys
import time

import torch

from . import __version__
from .artifacts import Artifacts
from .config import Settings
from .detector import EventTracker, FrameBuffer, baseline_slots, is_alert
from .display import write_result_renders
from .grid import CADENCE, CHANNELS, day_key, floor_slot, iso, parse_datetime, parse_iso, utcnow
from .localize import METHOD_ROBUST, locate, rate_from_fits, rate_map
from .model import Ensemble
from .preprocess import build_current, build_input, load_norm_stats
from .qc import QualityControl
from .replays import ReplayBuilder
from .retention import Retention
from .sdnotify import notify
from .source import LiveSource, ReplaySource
from .store import ResultStore, append_event, claim_event_id, read_json, write_json_atomic
from .streak import NoResultWatch

log = logging.getLogger("flare_nrt.loop")

MISSING = "missing"


def input_age_s(inputs: dict | None, now: dt.datetime) -> float | None:
    """now − newest provider T_OBS among the channels, seconds."""
    times = [parse_datetime(v) for v in (inputs or {}).values() if v]
    return round((now - max(times)).total_seconds(), 1) if times else None


def aia131_excess(cur, bls) -> float | None:
    """Relative rise of the disk-integrated 131 Å count rate over the same
    baseline frames the model uses: Σ(cur)/mean(Σ(bl)) − 1. The input of the
    non-neural baseline rule in ``flare_nrt.validate`` (docs/model-card.md)."""
    import numpy as np

    if cur is None or not bls or any(b is None for b in bls):
        return None
    ref = float(np.mean([np.nansum(b) for b in bls]))
    return round(float(np.nansum(cur)) / ref - 1.0, 6) if ref > 0 else None


def event_location(loc: dict | None, tracker: EventTracker, slot: dt.datetime) -> dict | None:
    """What the event tracker may keep of an alert frame's location (1.3.0).
    The tracker keeps the location of the event's highest-probability frame.
    A location found by the robust fallback (``METHOD_ROBUST``, after the
    131 Å maximum: post-flare loops, not the kernel) is offered only while
    the event has none, so every event 1.2.0 located keeps that location
    and an event it could not locate gets one."""
    if not loc or loc.get("method") != METHOD_ROBUST:
        return loc
    ev = tracker.current
    if ev is None or slot - parse_iso(ev.t_end) > tracker.gap:    # this slot opens a new event
        return loc
    return None if ev.location else loc


def signal_state(opened: bool, alert: bool, is_open: bool) -> str:
    """What the public signal is doing at this slot:

    - ``quiet``: no event open;
    - ``onset``: this slot opened an event (first alert);
    - ``active``: an event is open and this slot alerts too;
    - ``decay``: an event is open but this slot does not alert; it closes
      when no alert follows within ``merge_gap_min``.

    ``flare_detected`` is true in onset, active and decay."""
    if opened:
        return "onset"
    if not is_open:
        return "quiet"
    return "active" if alert else "decay"


class Processor:
    """Everything that turns one slot into one result row. No timing, no
    network: the loop (or a test) feeds it slots in order.

    ``keep_state=False`` (the reprocess command, 1.3.1): rows and events
    only; the live loop's state files (``loop_state.json``,
    ``last_frame.json``, ``status.json``, ``no_result.json``) are left alone."""

    def __init__(self, cfg: Settings, ensemble: Ensemble, source, live: bool, keep_state: bool = True):
        self.cfg = cfg
        self.ens = ensemble
        self.src = source
        self.live = live
        self.keep_state = keep_state
        self.supersede = False       # reprocess: a new row replaces the slot's old one
        self.v_high = load_norm_stats(cfg.norm_stats)
        self.qc = QualityControl(ignore_mask=cfg.qc_ignore_quality_mask)
        keep = dt.timedelta(minutes=cfg.lag_min) + (cfg.n_baseline + 2) * CADENCE
        self.buf = FrameBuffer(keep)
        self.qc_reasons: dict = {}
        self.headers: dict = {}
        self.inputs: dict = {}
        self.rate131: dict = {}
        self.quality: dict = {}      # slot → (raw QUALITY per channel, ignored flags present)
        st = read_json(cfg.loop_state_path, {}) or {}
        self.tracker = EventTracker(cfg.merge_gap_min, st.get("last_id", 0), st.get("current"))
        self.results = ResultStore(cfg.results_dir)
        self.frame_count = 0
        self.artifacts = Artifacts(cfg, ensemble, enabled=live)
        self.gradcam_wanted = True
        # 1.1.0: render/<slot>/ WebP after each live result (display only)
        self.render = live
        self.errors = 0
        self.last_error: tuple[str, str] | None = None
        # 1.3.1: says when results stop (live loop only)
        self.watch = NoResultWatch(cfg.state_dir / "no_result.json", cfg.no_result_warn_slots) \
            if live and keep_state else None

    # -- frames -------------------------------------------------------------
    def current(self, slot: dt.datetime):
        """Preprocessed (4,512,512) tensor, None if QC rejected it, or MISSING."""
        if slot in self.buf:
            return self.buf.get(slot)
        paths = self.src.paths(slot)
        if paths is None:
            return MISSING
        q = self.qc.check_slot(slot, paths)
        self.quality[slot] = ({str(w): v for w, v in sorted(q.quality.items())}, q.quality_soft)
        if not q.ok:
            self.qc_reasons[slot] = q.reasons
            self.buf.put(slot, None)
            return None
        t = build_current({w: q.frames[w].data for w in CHANNELS}, self.v_high)
        self.headers[slot] = q.frames[131].header
        self.inputs[slot] = q.t_obs
        self.rate131[slot] = rate_map(q.frames[131].data, q.frames[131].header)
        self.buf.put(slot, t)
        for d in (self.headers, self.inputs, self.qc_reasons, self.rate131, self.quality):
            for s in [s for s in d if s not in self.buf]:
                del d[s]
        return t

    # -- one slot -----------------------------------------------------------
    def process(self, slot: dt.datetime, now: dt.datetime) -> dict:
        cfg = self.cfg
        row = {"slot": iso(slot), "made_at": iso(now), "model_version": __version__,
               "lag_min": cfg.lag_min, "n_baseline": cfg.n_baseline, "threshold": cfg.threshold}
        cur = self.current(slot)
        alert = False
        prob = None
        if cur is not MISSING and slot in self.quality:
            row["quality"], row["quality_soft"] = self.quality[slot]
        if cur is MISSING:
            row.update(status="missing", reason="channels_missing",
                       channels_missing=self.src.missing_channels(slot))
        elif cur is None:
            row.update(status="qc_reject", reason="qc", qc=self.qc_reasons.get(slot, []))
        else:
            inputs = self.inputs.get(slot) or {}
            row["inputs"] = inputs
            age = input_age_s(inputs, now) if self.live else None   # a replay's inputs are old by design
            if age is not None:
                row["input_age_s"] = age
            bl_slots = baseline_slots(slot, cfg.lag_min, cfg.n_baseline)
            bls = [self.current(s) for s in bl_slots]
            have = [(s, b) for s, b in zip(bl_slots, bls, strict=True) if b is not None and b is not MISSING]
            missing_bl = [iso(s) for s, b in zip(bl_slots, bls, strict=True) if b is None or b is MISSING]
            if self.live and age is not None and age > cfg.max_input_age_sec:
                row.update(status="inputs_old", reason="inputs_old")
            elif len(have) < cfg.min_baseline:
                row.update(status="no_baseline", reason="baseline_missing", baseline_missing=missing_bl)
            else:
                used = [s for s, _ in have]
                x = build_input(cur, [b for _, b in have])
                t0 = time.perf_counter()
                prob, scores = self.ens.predict(x)
                latency_ms = round(1000 * (time.perf_counter() - t0))
                alert = is_alert(prob, cfg.threshold)
                loc = self._locate(slot, used) if alert else None
                row["aia131_excess"] = aia131_excess(self.rate131.get(slot), [self.rate131.get(s) for s in used])
                row.update(status="ok", probability=round(prob, 6), raw_score=round(prob, 6), alert=alert,
                           ensemble_scores=[round(v, 6) for v in scores],
                           ensemble_std=round(statistics.pstdev(scores), 6),
                           location=loc, baseline_slots=[iso(s) for s in used],
                           baseline_missing=missing_bl, n_baseline_used=len(used),
                           latency_ms=latency_ms)
                if self.live:
                    row["data_latency_s"] = round((now - slot).total_seconds())
                self.frame_count += 1
                cam = self.artifacts.save(slot, x, cur, alert, gradcam=alert or self.gradcam_wanted,
                                          probability=prob, header=self.headers.get(slot))
                if self.render:
                    p171 = (self.src.paths(slot) or {}).get(171)
                    write_result_renders(cfg, slot, p171, [(self.src.paths(s) or {}).get(171) for s in used], cam)
        ev_id, opened, closed = self.tracker.update(slot, alert, prob or 0.0,
                                                    event_location(row.get("location"), self.tracker, slot))
        if opened:
            ev_id = self._claim_event_id(ev_id)
        if closed:
            append_event(cfg.events_path, closed)
            log.info("event %d closed: %s → %s, p_max %.3f", closed.event_id, closed.t_start,
                     closed.t_end, closed.p_max)
        if opened:
            log.info("event %d opened at %s (p=%.3f)", ev_id, iso(slot), prob)
        row["state"] = signal_state(opened, alert, self.tracker.current is not None)
        row["event_id"] = ev_id
        if self.supersede:
            row["reprocessed"] = True
        self.results.append(row, supersede=self.supersede)
        if not self.keep_state:
            return row
        write_json_atomic(cfg.loop_state_path, {"last_slot": iso(slot), **self.tracker.state()})
        write_json_atomic(cfg.state_dir / "last_frame.json",
                          {k: row.get(k) for k in ("slot", "made_at", "status", "reason", "qc",
                                                   "channels_missing", "baseline_missing",
                                                   "input_age_s", "inputs", "quality", "quality_soft")})
        if row["status"] == "ok":
            self._write_status(row)
        if self.watch:
            self.watch.update(row)
        return row

    def _claim_event_id(self, proposed: int) -> int:
        """1.3.1: the loop and the reprocess command each have a tracker;
        a new event takes the next id neither has handed out."""
        new = claim_event_id(self.cfg.events_path, self.cfg.state_dir / "event_ids.json", proposed)
        if new != proposed:
            self.tracker.current.event_id = new
            self.tracker.last_id = new
        return new

    def _rate(self, slot: dt.datetime):
        """131 Å DN/s of a slot: kept from preprocessing, else read from its
        FITS (1.3.0), so a location never depends on what is in memory."""
        if slot in self.rate131:
            return self.rate131[slot]
        p = (self.src.paths(slot) or {}).get(131)
        if p is None:
            return None
        try:
            rate, header = rate_from_fits(p)
        except Exception:
            log.exception("131 Å rate of %s from %s failed", iso(slot), p)
            return None
        self.headers.setdefault(slot, header)
        return rate

    def _locate(self, slot: dt.datetime, used: list[dt.datetime]) -> dict | None:
        """The alert frame's location: 131 Å DN/s difference against the
        baselines the result used (flare_nrt/localize.py)."""
        cur, bls = self._rate(slot), [self._rate(s) for s in used]
        if cur is None or any(b is None for b in bls):
            log.warning("alert at %s: no location, 131 Å FITS missing for %s", iso(slot),
                        ", ".join(iso(s) for s, r in zip([slot, *used], [cur, *bls], strict=True) if r is None))
            return None
        loc = locate(cur, bls, self.headers.get(slot, {}))
        if loc is None:
            log.warning("alert at %s: no location, no significant 131 Å brightening", iso(slot))
        return loc

    def _write_status(self, row: dict) -> None:
        open_ev = self.tracker.current
        status = {
            "status": "running",
            "model": "flare-nrt",
            "model_version": __version__,
            "weights_sha256": self.ens.meta.get("sha256"),
            "timestamp": row["made_at"],          # when the result was made (portal: issued)
            "last_update": row["made_at"],
            "observation_time": row["slot"],      # the AIA slot the result is about
            "state": row["state"],
            "probability": row["probability"],
            "raw_score": row["raw_score"],
            "alert": row["alert"],
            "flare_detected": open_ev is not None,
            "event_id": open_ev.event_id if open_ev else None,
            "event_t_start": open_ev.t_start if open_ev else None,
            "threshold": self.cfg.threshold,
            "location": row.get("location") or (open_ev.location if open_ev else None),
            "ensemble_scores": row["ensemble_scores"],
            "ensemble_std": row["ensemble_std"],
            "lag_min": self.cfg.lag_min,
            "baseline_slots": row["baseline_slots"],
            "baseline_missing": row["baseline_missing"],
            "n_baseline_used": row["n_baseline_used"],
            "inputs": row.get("inputs"),
            "data_latency_s": row.get("data_latency_s"),
            "latency_ms": row["latency_ms"],
            "frame_count": self.frame_count,
            "loop_errors": self.errors,
            "loop_last_error": self.last_error[1] if self.last_error else None,
            "loop_last_error_ts": self.last_error[0] if self.last_error else None,
        }
        write_json_atomic(self.cfg.status_path, status)

    def flush(self) -> None:
        """Replay end: close the open event so events.csv is complete."""
        ev = self.tracker.close()
        if ev:
            append_event(self.cfg.events_path, ev)
        if self.keep_state:
            write_json_atomic(self.cfg.loop_state_path, {"last_slot": None, **self.tracker.state()})


def should_skip(src, slot: dt.datetime, now: dt.datetime, cfg: Settings,
                started: dt.datetime | None = None) -> bool:
    """A missing slot is given up after ``give_up_min``, or after
    ``skip_after_min`` if any later slot (up to ``give_up_min`` later) has
    already arrived — one lost file, or a cold start whose first minutes the
    archive doesn't have, must not hold the whole stream back. A slot from
    before the loop started (``started``) is not waited for at all once a
    later slot is complete: only the archive could still fill it, and it
    lags."""
    age = now - slot
    n_later = int(cfg.give_up_min // 2)
    if age >= dt.timedelta(minutes=cfg.give_up_min):
        return True
    cold = started is not None and slot < floor_slot(started) - dt.timedelta(minutes=cfg.lag_min)
    if not cold and age < dt.timedelta(minutes=cfg.skip_after_min):
        return False
    return any(src.paths(slot + k * CADENCE) is not None for k in range(1, n_later + 1))


def run(cfg: Settings, replay_dir=None, max_iterations: int | None = None,
        replay_gradcam: bool = True, resume: bool = False) -> Processor:
    cfg.ensure_dirs()
    torch.set_num_threads(cfg.torch_threads)
    ens = Ensemble.load(cfg.weights, cfg.weights_sha256)
    live = replay_dir is None
    src = LiveSource(cfg.staging_dir, cfg.jsoc_base, cfg.lookback_min, cfg.http_timeout,
                     cfg.archive_every_sec, cfg.archive_budget_sec, cfg.download_max_sec) if live \
        else ReplaySource(replay_dir)
    proc = Processor(cfg, ens, src, live)
    log.info("flare-nrt %s | L=%d min, n_bl=%d, thr=%.2f, merge=%d min | %s | weights %s",
             __version__, cfg.lag_min, cfg.n_baseline, cfg.threshold, cfg.merge_gap_min,
             "live" if live else f"replay {replay_dir}", ens.meta["sha256"][:12])

    if not live:
        # Grad-CAM of the alerting frames only (the retrospective viewer); no tensors
        proc.artifacts = Artifacts(cfg, ens, enabled=replay_gradcam, tensors=False)
        proc.gradcam_wanted = False
        done = set()
        if resume:
            for d in sorted({day_key(s) for s in src.slots}):
                done |= {r["slot"] for r in proc.results.read_day(d)}
        todo = [s for s in src.slots if iso(s) not in done]
        if done:
            log.info("resume: %d slots already in results, %d to go", len(src.slots) - len(todo), len(todo))
        t0 = time.monotonic()
        for i, slot in enumerate(todo, 1):
            proc.process(slot, utcnow())
            if i % 360 == 0 or i == len(todo):   # every 12 h of data
                left = (time.monotonic() - t0) / i * (len(todo) - i) / 60
                log.info("replay at %s: %d/%d slots, %d ok, %d events, ~%.0f min left",
                         iso(slot), i, len(todo), proc.frame_count, proc.tracker.last_id, left)
        proc.flush()
        return proc

    retention = Retention(cfg, every_sec=600 if cfg.staging_hours > 0 else 3600)
    def alive() -> None:                   # progress → systemd's watchdog (1.3.1)
        notify("WATCHDOG=1")

    replays = ReplayBuilder(cfg, ens, tick=alive)   # 1.3.0: the loaded model fills missing Grad-CAM maps
    st = read_json(cfg.loop_state_path, {}) or {}
    now = started = utcnow()
    hb = read_json(cfg.state_dir / "heartbeat.json", {}) or {}
    if hb.get("last_download"):          # 1.3.1: survives a restart
        src.last_download = parse_iso(hb["last_download"])
    proc.watch.bootstrap(proc.results, (read_json(cfg.status_path, {}) or {}).get("observation_time"), now)
    proc.watch.announce()
    notify("READY=1")
    nxt = floor_slot(now - dt.timedelta(minutes=20))
    if st.get("last_slot"):
        prev = parse_iso(st["last_slot"])
        if now - prev < dt.timedelta(minutes=cfg.lookback_min):
            nxt = prev + CADENCE
    it = 0
    while max_iterations is None or it < max_iterations:
        it += 1
        now = utcnow()
        src.refresh(now)
        alive()
        last_fetch = time.monotonic()
        while nxt <= floor_slot(now):
            if src.paths(nxt) is None and not should_skip(src, nxt, now, cfg, started):
                break  # may still arrive
            if time.monotonic() - last_fetch > cfg.poll_sec:
                # working through a backlog (~20 s/slot on 2 vCPU): keep catching
                # mostrecent/, which holds only the newest image of each channel
                src.refresh_mostrecent()
                last_fetch = time.monotonic()
                now = utcnow()
            try:
                # Grad-CAM costs as much as the inference: only for the newest
                # frame and for alerts, not for each slot of a backlog
                proc.gradcam_wanted = src.paths(nxt + CADENCE) is None
                proc.process(nxt, utcnow())
            except Exception as e:  # one bad slot must not stop the service
                proc.errors += 1
                proc.last_error = (iso(utcnow()), f"{type(e).__name__}: {e}")
                log.exception("slot %s failed", iso(nxt))
                err = {"slot": iso(nxt), "made_at": iso(utcnow()), "status": "error",
                       "error": proc.last_error[1], "model_version": __version__}
                proc.results.append(err)
                proc.watch.update(err)
            alive()
            nxt += CADENCE
        write_json_atomic(cfg.state_dir / "heartbeat.json",
                          {"heartbeat": iso(utcnow()), "next_slot": iso(nxt),
                           "last_download": iso(src.last_download)})
        replays.maybe_run(utcnow(), nxt - CADENCE)
        retention.maybe_run()
        alive()
        if max_iterations is None or it < max_iterations:
            time.sleep(cfg.poll_sec)
    return proc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", help="process existing FITS in this directory, then exit")
    ap.add_argument("--data-dir", help="override FLARE_NRT_DATA_DIR")
    ap.add_argument("--resume", action="store_true", help="replay: skip slots already in results/")
    ap.add_argument("--no-gradcam", action="store_true", help="replay: don't save Grad-CAM of alerting frames")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    cfg = Settings()
    if a.data_dir:
        from pathlib import Path
        cfg.data_dir = Path(a.data_dir)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    run(cfg, replay_dir=a.replay, replay_gradcam=not a.no_gradcam, resume=a.resume)


if __name__ == "__main__":
    main()
