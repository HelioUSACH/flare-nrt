"""HTTP contract read by the portal (clima-espacial ``modules/flare/``).
Listens on 127.0.0.1 only; no authentication from localhost.

    uvicorn centella.server:app --host 127.0.0.1 --port 5050

``/api/status`` keeps the field names the portal's adapter reads
(``live_status``, ``timestamp``, ``seconds_since_update``, ``probability``,
``flare_detected``, ``location``, ``ensemble_std``); see INTEGRATION.md for
the full table. ``live_status`` is computed on every request:

- ``no_data``: no result yet;
- ``stale``: loop heartbeat or last result older than ``stale_after_sec``,
  or the observation behind the result older than ``max_input_age_sec``
  (``stale_reason``: ``loop_not_running`` / ``no_new_result`` / ``inputs_old``);
- ``live`` otherwise.
"""

from __future__ import annotations

import datetime as dt
import time

from fastapi import FastAPI, HTTPException, Query, Response

from . import __version__
from .config import Settings
from .detector import baseline_slots
from .grid import CHANNELS, day_key, file_name, iso, parse_file_name, parse_iso, utcnow
from .store import ResultStore, read_events, read_json


def live_status(status: dict | None, heartbeat: dict | None, now: dt.datetime, stale_after: int,
                max_input_age: int = 3600):
    if not status or not status.get("timestamp"):
        return "no_data", None, None
    since = (now - parse_iso(status["last_update"])).total_seconds()
    hb = heartbeat.get("heartbeat") if heartbeat else None
    if hb is None or (now - parse_iso(hb)).total_seconds() > stale_after:
        return "stale", "loop_not_running", since
    if since > stale_after:
        return "stale", "no_new_result", since
    obs = status.get("observation_time")
    if obs and (now - parse_iso(obs)).total_seconds() > max_input_age:
        return "stale", "inputs_old", since
    return "live", None, since


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or Settings()
    app = FastAPI(title="Centella", version=__version__)

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
            "stale_after_sec": cfg.stale_after_sec,
            "server_time": iso(now),
            "model_version": __version__,
        })
        out.setdefault("status", "no_data")
        return out

    @app.get("/api/events")
    def events(days: int = Query(30, ge=1, le=400)):
        since = utcnow() - dt.timedelta(days=days)
        evs = [e for e in read_events(cfg.events_path) if parse_iso(e["t_start"]) >= since]
        st = read_json(cfg.loop_state_path, {}) or {}
        if st.get("current"):
            evs.append({**st["current"], "closed": False})
        return {"events": evs}

    @app.get("/api/results")
    def results(hours: int = Query(24, ge=1, le=168)):
        now = utcnow()
        since = now - dt.timedelta(hours=hours)
        store = ResultStore(cfg.results_dir)
        days = sorted({day_key(since + dt.timedelta(days=i)) for i in range((now - since).days + 2)})
        pts = []
        for d in days:
            for r in store.read_day(d):
                if parse_iso(r["slot"]) >= since:
                    pts.append({"t": r["slot"], "p": r.get("probability"), "status": r.get("status"),
                                "event_id": r.get("event_id")})
        return {"threshold": cfg.threshold, "lag_min": cfg.lag_min, "points": pts}

    def _slot_files(slot_iso: str) -> dict:
        slot = parse_iso(slot_iso)
        d = cfg.staging_dir / day_key(slot)
        return {w: d / file_name(slot, w) for w in CHANNELS}

    def _latest_slot(wave: int):
        """Newest staging slot holding this channel (before the first full
        result, or when the result's files were already cleaned up)."""
        best = None
        for p in cfg.staging_dir.glob(f"*/AIA*_{wave:04d}.fits"):
            parsed = parse_file_name(p.name)
            if parsed and (best is None or parsed[0] > best):
                best = parsed[0]
        return best

    def _pick(wave: int):
        """(slot, image path, baseline paths) for one channel: the latest
        result's slot when its file exists, else the newest file staged."""
        st = read_json(cfg.status_path) or {}
        slot = st.get("observation_time")
        if slot:
            path = _slot_files(slot)[wave]
            if path.exists():
                return parse_iso(slot), path, [_slot_files(b)[wave] for b in st.get("baseline_slots") or []]
        s = _latest_slot(wave)
        if s is None:
            return None, None, []
        bls = [_slot_files(iso(b))[wave] for b in baseline_slots(s, cfg.lag_min, cfg.n_baseline)]
        return s, _slot_files(iso(s))[wave], bls

    @app.get("/api/channels")
    def channels():
        """The 4 channels: slot and provider time of the image shown, and
        whether its difference against the baselines is available."""
        st = read_json(cfg.status_path) or {}
        inputs = st.get("inputs") or {}
        out = []
        for w in CHANNELS:
            s, path, bls = _pick(w)
            out.append({"wave": w, "slot": iso(s), "t_obs": inputs.get(str(w)) or inputs.get(w),
                        "image": bool(path and path.exists()),
                        "difference": bool(path and path.exists() and bls and all(b.exists() for b in bls))})
        return {"slot": st.get("observation_time"), "baseline_slots": st.get("baseline_slots") or [],
                "channels": out}

    @app.get("/api/image/{wave}.png")
    def image(wave: int, kind: str = Query("img", pattern="^(img|diff)$")):
        """PNG of one channel (``kind=img``) or its difference against the
        baselines (``kind=diff``), 512 px, north up. The latest result's slot,
        or the newest staged image until there is one."""
        from . import render

        if wave not in CHANNELS:
            raise HTTPException(404, "unknown channel")
        slot, path, bls = _pick(wave)
        if slot is None or not path.exists():
            raise HTTPException(404, "no image yet")
        if kind == "diff" and (not bls or not all(b.exists() for b in bls)):
            raise HTTPException(404, "difference not available yet")
        cache_dir = cfg.state_dir / "images"
        cached = cache_dir / f"{slot:%Y%m%dT%H%M}_{wave}_{kind}.png"
        if not cached.exists():
            png = render.image_png(path, wave) if kind == "img" else render.diff_png(path, bls)
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".tmp")
            tmp.write_bytes(png)
            tmp.replace(cached)
            for old in cache_dir.glob("*.png"):  # previews live one day
                if time.time() - old.stat().st_mtime > 86400:
                    old.unlink(missing_ok=True)
        return Response(cached.read_bytes(), media_type="image/png",
                        headers={"Cache-Control": "max-age=60", "X-Observation-Time": iso(slot)})

    @app.get("/api/health")
    def health():
        now = utcnow()
        store = ResultStore(cfg.results_dir)
        rows = store.read_day(day_key(now)) + store.read_day(day_key(now - dt.timedelta(days=1)))
        rows = [r for r in rows if now - parse_iso(r["slot"]) <= dt.timedelta(hours=24)]
        counts: dict[str, int] = {}
        for r in rows:
            counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
        lat = sorted(r["data_latency_s"] for r in rows if r.get("data_latency_s") is not None)
        hb = read_json(cfg.state_dir / "heartbeat.json") or {}
        import shutil
        return {
            "slots_24h": len(rows),
            "coverage_24h": round(counts.get("ok", 0) / 720, 3),
            "status_counts_24h": counts,
            "data_latency_s_median": lat[len(lat) // 2] if lat else None,
            "heartbeat": hb.get("heartbeat"),
            "last_download": hb.get("last_download"),
            "disk_free_gb": round(shutil.disk_usage(cfg.data_dir).free / 1e9, 1) if cfg.data_dir.exists() else None,
        }

    return app


app = create_app()
