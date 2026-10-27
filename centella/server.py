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

from fastapi import FastAPI, Query

from . import __version__
from .config import Settings
from .grid import day_key, iso, parse_iso, utcnow
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
