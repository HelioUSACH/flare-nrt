"""HTTP contract read by the portal (clima-espacial ``modules/flare/``).
Listens on 127.0.0.1 only; no authentication from localhost. The full field
tables are in INTEGRATION.md.

    uvicorn centella.server:app --host 127.0.0.1 --port 5050

``/api/status`` keeps the fields the portal's adapter reads (``live_status``,
``timestamp``, ``seconds_since_update``, ``probability``, ``flare_detected``,
``location``, ``ensemble_std``). ``live_status`` is computed on every
request:

- ``no_data``: no result yet;
- ``stale``: the loop heartbeat or the last result is older than
  ``stale_after_sec`` (``stale_reason`` ``loop_not_running`` /
  ``no_new_result``), or the newest input of the last result is older than
  ``max_input_age_sec`` (``inputs_old``);
- ``live`` otherwise.

``last_frame`` says what happened to the newest slot the loop looked at, so a
skipped frame explains itself (``missing`` / ``qc_reject`` /
``no_baseline`` / ``inputs_old``).

The API only reads the data directory, except ``cache/`` where it keeps the
PNG previews it renders."""

from __future__ import annotations

import datetime as dt
import shutil
import time

from fastapi import FastAPI, HTTPException, Query, Response

from . import __version__
from .artifacts import gradcam_path
from .config import Settings
from .detector import baseline_slots
from .grid import CADENCE, CHANNELS, day_key, file_name, iso, parse_datetime, parse_file_name, parse_iso, utcnow
from .store import ResultStore, read_events, read_json


def input_age(status: dict | None, now: dt.datetime) -> float | None:
    """now − newest provider T_OBS of the last result (falls back to its slot)."""
    if not status:
        return None
    times = [parse_datetime(v) for v in (status.get("inputs") or {}).values() if v]
    if not times and status.get("observation_time"):
        times = [parse_iso(status["observation_time"])]
    return round((now - max(times)).total_seconds(), 1) if times else None


def live_status(status: dict | None, heartbeat: dict | None, now: dt.datetime, stale_after: int,
                max_input_age: int = 900):
    """→ (live_status, stale_reason, seconds_since_update)."""
    if not status or not status.get("timestamp"):
        return "no_data", None, None
    since = (now - parse_iso(status["last_update"])).total_seconds()
    hb = heartbeat.get("heartbeat") if heartbeat else None
    if hb is None or (now - parse_iso(hb)).total_seconds() > stale_after:
        return "stale", "loop_not_running", since
    if since > stale_after:
        return "stale", "no_new_result", since
    age = input_age(status, now)
    if age is not None and age > max_input_age:
        return "stale", "inputs_old", since
    return "live", None, since


def _png(data: bytes, slot: dt.datetime, max_age: int = 60) -> Response:
    return Response(data, media_type="image/png",
                    headers={"Cache-Control": f"max-age={max_age}", "X-Observation-Time": iso(slot)})


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or Settings()
    app = FastAPI(title="Centella", version=__version__)

    def _rows(since: dt.datetime, now: dt.datetime) -> list[dict]:
        store = ResultStore(cfg.results_dir)
        days = sorted({day_key(since + dt.timedelta(days=i)) for i in range((now - since).days + 2)})
        return [r for d in days for r in store.read_day(d) if parse_iso(r["slot"]) >= since]

    def _slot_param(slot: str | None) -> dt.datetime | None:
        if slot is None:
            return None
        try:
            return parse_iso(slot)
        except ValueError:
            raise HTTPException(422, "slot must be ISO 8601, e.g. 2026-10-28T15:30:00Z") from None

    # -- the latest result -------------------------------------------------
    @app.get("/api/status")
    def status():
        now = utcnow()
        st = read_json(cfg.status_path)
        hb = read_json(cfg.state_dir / "heartbeat.json")
        ls, reason, since = live_status(st, hb, now, cfg.stale_after_sec, cfg.max_input_age_sec)
        out = dict(st or {})
        out.update({
            "live_status": ls,
            "stale_reason": reason,
            "seconds_since_update": None if since is None else round(since, 1),
            "input_age_s": input_age(st, now),
            "stale_after_sec": cfg.stale_after_sec,
            "max_input_age_sec": cfg.max_input_age_sec,
            "last_frame": read_json(cfg.state_dir / "last_frame.json"),
            "server_time": iso(now),
            "model_version": __version__,
        })
        out.setdefault("status", "no_data")
        out.setdefault("state", None)
        return out

    # -- history -----------------------------------------------------------
    @app.get("/api/events")
    def events(days: int = Query(30, ge=1, le=400)):
        """Alerts merged into events: start (first alert), peak (highest
        probability), end (last alert), location at the peak, Grad-CAM of
        the peak frame when it is still stored."""
        since = utcnow() - dt.timedelta(days=days)
        evs = [e for e in read_events(cfg.events_path) if parse_iso(e["t_start"]) >= since]
        st = read_json(cfg.loop_state_path, {}) or {}
        if st.get("current"):
            evs.append({**st["current"], "closed": False})
        for e in evs:
            peak = parse_iso(e["t_pmax"])
            has = gradcam_path(cfg.gradcam_dir, peak, True).exists()
            e["gradcam"] = f"/api/gradcam/{iso(peak)}.png" if has else None
        return {"events": evs}

    @app.get("/api/series")
    def series(days: float = Query(3, gt=0, le=14)):
        """Probability per slot, compact: ``points`` = [[slot, probability
        or null, state, status], …], oldest first."""
        now = utcnow()
        rows = _rows(now - dt.timedelta(days=days), now)
        return {"threshold": cfg.threshold, "lag_min": cfg.lag_min,
                "fields": ["slot", "probability", "state", "status"],
                "points": [[r["slot"], r.get("probability"), r.get("state"), r.get("status")] for r in rows]}

    @app.get("/api/results")
    def results(hours: int = Query(24, ge=1, le=168)):
        """As in 2.0: ``points`` = [{t, p, status, event_id}], plus ``state``."""
        now = utcnow()
        return {"threshold": cfg.threshold, "lag_min": cfg.lag_min,
                "points": [{"t": r["slot"], "p": r.get("probability"), "status": r.get("status"),
                            "state": r.get("state"), "event_id": r.get("event_id")}
                           for r in _rows(now - dt.timedelta(hours=hours), now)]}

    # -- images ------------------------------------------------------------
    def _slot_files(slot: dt.datetime) -> dict:
        d = cfg.staging_dir / day_key(slot)
        return {w: d / file_name(slot, w) for w in CHANNELS}

    def _latest_slot(wave: int):
        best = None
        for p in cfg.staging_dir.glob(f"*/AIA*_{wave:04d}.fits"):
            parsed = parse_file_name(p.name)
            if parsed and (best is None or parsed[0] > best):
                best = parsed[0]
        return best

    def _pick(wave: int, slot: dt.datetime | None):
        """(slot, image path, baseline paths) for one channel. ``slot`` given:
        that slot, with the baselines its result used. Otherwise the latest
        result's slot, or the newest staged image before the first result."""
        if slot is not None:
            row = next((r for r in ResultStore(cfg.results_dir).read_day(day_key(slot))
                        if r["slot"] == iso(slot)), {})
            bls = row.get("baseline_slots") or [iso(b) for b in baseline_slots(slot, cfg.lag_min, cfg.n_baseline)]
            return slot, _slot_files(slot)[wave], [_slot_files(parse_iso(b))[wave] for b in bls]
        st = read_json(cfg.status_path) or {}
        if st.get("observation_time"):
            s = parse_iso(st["observation_time"])
            if _slot_files(s)[wave].exists():
                return s, _slot_files(s)[wave], [_slot_files(parse_iso(b))[wave]
                                                 for b in st.get("baseline_slots") or []]
        s = _latest_slot(wave)
        if s is None:
            return None, None, []
        return s, _slot_files(s)[wave], [_slot_files(b)[wave] for b in baseline_slots(s, cfg.lag_min, cfg.n_baseline)]

    @app.get("/api/frames")
    def frames(hours: float = Query(6, gt=0, le=168)):
        """Recent slots with what can be shown for each: AIA images
        (``/api/image/{wave}.png?slot=``) while the FITS are kept (7 days),
        Grad-CAM (``/api/gradcam/{slot}.png``) while it is kept (2 days;
        alerting frames 400 days)."""
        now = utcnow()
        out = []
        for r in _rows(now - dt.timedelta(hours=hours), now):
            s = parse_iso(r["slot"])
            files = _slot_files(s)
            out.append({"slot": r["slot"], "status": r.get("status"), "state": r.get("state"),
                        "probability": r.get("probability"), "alert": r.get("alert", False),
                        "inputs": r.get("inputs"),
                        "images": [w for w in CHANNELS if files[w].exists()],
                        "gradcam": gradcam_path(cfg.gradcam_dir, s, False).exists()
                        or gradcam_path(cfg.gradcam_dir, s, True).exists()})
        return {"frames": out}

    @app.get("/api/channels")
    def channels():
        """The 4 channels of the image shown by default: its slot, provider
        time, and whether the difference against the baselines exists."""
        st = read_json(cfg.status_path) or {}
        inputs = st.get("inputs") or {}
        out = []
        for w in CHANNELS:
            s, path, bls = _pick(w, None)
            out.append({"wave": w, "slot": iso(s), "t_obs": inputs.get(str(w)) or inputs.get(w),
                        "image": bool(path and path.exists()),
                        "difference": bool(path and path.exists() and bls and all(b.exists() for b in bls))})
        return {"slot": st.get("observation_time"), "baseline_slots": st.get("baseline_slots") or [],
                "channels": out}

    @app.get("/api/image/{wave}.png")
    def image(wave: int, kind: str = Query("img", pattern="^(img|diff)$"), slot: str | None = None):
        """One AIA channel, 512 px PNG, north up: ``kind=img`` (DN/s, asinh,
        SDO-style colours) or ``kind=diff`` (current − mean of its baselines,
        grey ±). Without ``slot``: the latest result's slot."""
        from . import render

        if wave not in CHANNELS:
            raise HTTPException(404, "unknown channel")
        s, path, bls = _pick(wave, _slot_param(slot))
        if s is None or not path.exists():
            raise HTTPException(404, "image not available")
        bls = [b for b in bls if b.exists()]
        if kind == "diff" and len(bls) < cfg.min_baseline:
            raise HTTPException(404, "difference not available")
        cached = cfg.cache_dir / f"{s:%Y%m%dT%H%M}_{wave}_{kind}.png"
        if not cached.exists():
            png = render.image_png(path, wave) if kind == "img" else render.diff_png(path, bls)
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".tmp")
            tmp.write_bytes(png)
            tmp.replace(cached)
            for old in cfg.cache_dir.glob("*.png"):
                if time.time() - old.stat().st_mtime > cfg.image_cache_days * 86400:
                    old.unlink(missing_ok=True)
        return _png(cached.read_bytes(), s, 60 if slot is None else 3600)

    @app.get("/api/gradcam/{which}.png")
    def gradcam(which: str):
        """Grad-CAM of the ensemble over the 131 Å input, 512 px, north up:
        ``latest`` (the latest result) or a slot (ISO 8601)."""
        if which == "latest":
            st = read_json(cfg.status_path) or {}
            if not st.get("observation_time"):
                raise HTTPException(404, "no result yet")
            s = parse_iso(st["observation_time"])
        else:
            s = _slot_param(which)
        for alert in (False, True):
            p = gradcam_path(cfg.gradcam_dir, s, alert)
            if p.exists():
                return _png(p.read_bytes(), s, 60 if which == "latest" else 86400)
        raise HTTPException(404, "Grad-CAM not available for this slot")

    # -- operations --------------------------------------------------------
    @app.get("/api/health")
    def health():
        now = utcnow()
        rows = _rows(now - dt.timedelta(hours=24), now)
        counts: dict[str, int] = {}
        for r in rows:
            counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
        lat = sorted(r["data_latency_s"] for r in rows if r.get("data_latency_s") is not None)
        hb = read_json(cfg.state_dir / "heartbeat.json") or {}
        return {
            "slots_24h": len(rows),
            "coverage_24h": round(counts.get("ok", 0) / (86400 / CADENCE.total_seconds()), 3),
            "status_counts_24h": counts,
            "data_latency_s_median": lat[len(lat) // 2] if lat else None,
            "heartbeat": hb.get("heartbeat"),
            "last_download": hb.get("last_download"),
            "disk_free_gb": round(shutil.disk_usage(cfg.data_dir).free / 1e9, 1) if cfg.data_dir.exists() else None,
        }

    return app


app = create_app()
