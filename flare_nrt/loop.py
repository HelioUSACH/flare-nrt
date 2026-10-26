"""The detection loop: every poll, download what JSOC has published, then
process every slot in order, exactly once.

    python -m flare_nrt.loop                 # live (systemd: deploy/flare-loop.service)
    python -m flare_nrt.loop --replay DIR    # existing FITS, same code path

Per slot: QC → preprocess → baseline from t−L…(n_baseline frames) → ensemble
→ alert rule → event merging → results/YYYYMMDD.jsonl (+ status.json for
the API). A slot whose files haven't arrived is waited for up to
``give_up_min``, then recorded as ``missing``."""

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
from .config import Settings
from .detector import EventTracker, FrameBuffer, baseline_slots, is_alert
from .grid import CADENCE, CHANNELS, floor_slot, iso, parse_iso, utcnow
from .localize import localize
from .model import Ensemble
from .preprocess import build_current, build_input, load_norm_stats
from .qc import QualityControl
from .retention import Retention
from .source import LiveSource, ReplaySource
from .store import ResultStore, append_event, read_json, write_json_atomic

log = logging.getLogger("flare_nrt.loop")

MISSING = "missing"


class Processor:
    """Everything that turns one slot into one result row. No timing, no
    network: the loop (or a test) feeds it slots in order."""

    def __init__(self, cfg: Settings, ensemble: Ensemble, source, live: bool):
        self.cfg = cfg
        self.ens = ensemble
        self.src = source
        self.live = live
        self.v_high = load_norm_stats(cfg.norm_stats)
        self.qc = QualityControl()
        keep = dt.timedelta(minutes=cfg.lag_min) + (cfg.n_baseline + 2) * CADENCE
        self.buf = FrameBuffer(keep)
        self.qc_reasons: dict = {}
        self.headers: dict = {}
        self.inputs: dict = {}
        st = read_json(cfg.loop_state_path, {}) or {}
        self.tracker = EventTracker(cfg.merge_gap_min, st.get("last_id", 0), st.get("current"))
        self.results = ResultStore(cfg.results_dir)
        self.frame_count = 0
        self.errors = 0
        self.last_error: tuple[str, str] | None = None

    # -- frames -------------------------------------------------------------
    def current(self, slot: dt.datetime):
        """Preprocessed (4,512,512) tensor, None if QC rejected it, or MISSING."""
        if slot in self.buf:
            return self.buf.get(slot)
        paths = self.src.paths(slot)
        if paths is None:
            return MISSING
        q = self.qc.check_slot(slot, paths)
        if not q.ok:
            self.qc_reasons[slot] = q.reasons
            self.buf.put(slot, None)
            return None
        t = build_current({w: q.frames[w].data for w in CHANNELS}, self.v_high)
        self.headers[slot] = q.frames[131].header
        self.inputs[slot] = q.t_obs
        self.buf.put(slot, t)
        for d in (self.headers, self.inputs, self.qc_reasons):
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
        if cur is MISSING:
            row["status"] = "missing"
        elif cur is None:
            row.update(status="qc_reject", qc=self.qc_reasons.get(slot, []))
        else:
            bl_slots = baseline_slots(slot, cfg.lag_min, cfg.n_baseline)
            bls = [self.current(s) for s in bl_slots]
            if any(b is None or b is MISSING for b in bls):
                row.update(status="no_baseline",
                           baseline_missing=[iso(s) for s, b in zip(bl_slots, bls, strict=True)
                                             if b is None or b is MISSING])
            else:
                x = build_input(cur, bls)
                t0 = time.perf_counter()
                prob, scores = self.ens.predict(x)
                alert = is_alert(prob, cfg.threshold)
                loc = localize(x, self.headers.get(slot, {})) if alert else None
                row.update(status="ok", probability=round(prob, 6), alert=alert,
                           ensemble_scores=[round(s, 6) for s in scores],
                           ensemble_std=round(statistics.pstdev(scores), 6),
                           location=loc, inputs=self.inputs.get(slot),
                           baseline_slots=[iso(s) for s in bl_slots],
                           latency_ms=round(1000 * (time.perf_counter() - t0)))
                if self.live:
                    row["data_latency_s"] = round((now - slot).total_seconds())
                self.frame_count += 1
        ev_id, opened, closed = self.tracker.update(slot, alert, prob or 0.0, row.get("location"))
        if closed:
            append_event(cfg.events_path, closed)
            log.info("event %d closed: %s → %s, p_max %.3f", closed.event_id, closed.t_start,
                     closed.t_end, closed.p_max)
        if opened:
            log.info("event %d opened at %s (p=%.3f)", ev_id, iso(slot), prob)
        row["event_id"] = ev_id
        self.results.append(row)
        write_json_atomic(cfg.loop_state_path, {"last_slot": iso(slot), **self.tracker.state()})
        if row["status"] == "ok":
            self._write_status(row)
        return row

    def _write_status(self, row: dict) -> None:
        open_ev = self.tracker.current
        status = {
            "status": "running",
            "model_version": __version__,
            "weights_sha256": self.ens.meta.get("sha256"),
            "timestamp": row["made_at"],          # when the result was made (portal: issued)
            "last_update": row["made_at"],
            "observation_time": row["slot"],      # the AIA slot the result is about
            "probability": row["probability"],
            "alert": row["alert"],
            "flare_detected": open_ev is not None,
            "event_id": open_ev.event_id if open_ev else None,
            "threshold": self.cfg.threshold,
            "location": row.get("location") or (open_ev.location if open_ev else None),
            "ensemble_scores": row["ensemble_scores"],
            "ensemble_std": row["ensemble_std"],
            "lag_min": self.cfg.lag_min,
            "baseline_slots": row["baseline_slots"],
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
        write_json_atomic(self.cfg.loop_state_path, {"last_slot": None, **self.tracker.state()})


def should_skip(src, slot: dt.datetime, now: dt.datetime, cfg: Settings) -> bool:
    """A missing slot is given up after ``give_up_min``, or after
    ``skip_after_min`` if a later slot (within 10 min) has already arrived —
    one lost file must not hold the whole stream back."""
    age = now - slot
    if age >= dt.timedelta(minutes=cfg.give_up_min):
        return True
    if age < dt.timedelta(minutes=cfg.skip_after_min):
        return False
    return any(src.paths(slot + k * CADENCE) is not None for k in range(1, 6))


def run(cfg: Settings, replay_dir=None, max_iterations: int | None = None) -> Processor:
    cfg.ensure_dirs()
    torch.set_num_threads(cfg.torch_threads)
    ens = Ensemble.load(cfg.weights, cfg.weights_sha256)
    live = replay_dir is None
    src = LiveSource(cfg.staging_dir, cfg.jsoc_base, cfg.lookback_min, cfg.http_timeout) if live \
        else ReplaySource(replay_dir)
    proc = Processor(cfg, ens, src, live)
    log.info("flare-nrt %s | L=%d min, n_bl=%d, thr=%.2f, merge=%d min | %s | weights %s",
             __version__, cfg.lag_min, cfg.n_baseline, cfg.threshold, cfg.merge_gap_min,
             "live" if live else f"replay {replay_dir}", ens.meta["sha256"][:12])

    if not live:
        for slot in src.slots:
            proc.process(slot, utcnow())
        proc.flush()
        return proc

    retention = Retention(cfg)
    st = read_json(cfg.loop_state_path, {}) or {}
    now = utcnow()
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
        while nxt <= floor_slot(now):
            if src.paths(nxt) is None and not should_skip(src, nxt, now, cfg):
                break  # may still arrive
            try:
                proc.process(nxt, utcnow())
            except Exception as e:  # one bad slot must not stop the service
                proc.errors += 1
                proc.last_error = (iso(utcnow()), f"{type(e).__name__}: {e}")
                log.exception("slot %s failed", iso(nxt))
                proc.results.append({"slot": iso(nxt), "made_at": iso(utcnow()), "status": "error",
                                     "error": proc.last_error[1], "model_version": __version__})
            nxt += CADENCE
        write_json_atomic(cfg.state_dir / "heartbeat.json",
                          {"heartbeat": iso(utcnow()), "next_slot": iso(nxt),
                           "last_download": iso(src.last_download)})
        retention.maybe_run()
        if max_iterations is None or it < max_iterations:
            time.sleep(cfg.poll_sec)
    return proc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", help="process existing FITS in this directory, then exit")
    ap.add_argument("--data-dir", help="override FLARE_DATA_DIR")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    cfg = Settings()
    if a.data_dir:
        from pathlib import Path
        cfg.data_dir = Path(a.data_dir)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    run(cfg, replay_dir=a.replay)


if __name__ == "__main__":
    main()
