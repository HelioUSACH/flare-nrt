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
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse

from . import __version__
from .artifacts import gradcam_path
from .config import Settings
from .detector import baseline_slots
from .goes import class_flux, read_csv
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

    def _rows(since: dt.datetime, until: dt.datetime) -> list[dict]:
        store = ResultStore(cfg.results_dir)
        days = sorted({day_key(since + dt.timedelta(days=i)) for i in range((until - since).days + 2)})
        return [r for d in days for r in store.read_day(d) if since <= parse_iso(r["slot"]) <= until]

    def _window(start: str | None, end: str | None, default_days: float, max_days: float):
        """(since, until) from optional ISO ``start``/``end``; without them the
        last ``default_days`` up to now. Longer than ``max_days`` → 422."""
        until = _slot_param(end) if end else (
            _slot_param(start) + dt.timedelta(days=default_days) if start else utcnow())
        since = _slot_param(start) if start else until - dt.timedelta(days=default_days)
        if until <= since or until - since > dt.timedelta(days=max_days):
            raise HTTPException(422, f"start < end, at most {max_days:g} days apart")
        return since, until

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
    def events(days: int = Query(30, ge=1, le=400), start: str | None = None, end: str | None = None):
        """Alerts merged into events: start (first alert), peak (highest
        probability), end (last alert), location at the peak, Grad-CAM of
        the peak frame when it is still stored. ``start``/``end`` (ISO)
        select any window instead of the last ``days``."""
        since, until = _window(start, end, days, 400)
        evs = [e for e in read_events(cfg.events_path) if since <= parse_iso(e["t_start"]) <= until]
        st = read_json(cfg.loop_state_path, {}) or {}
        cur = st.get("current")
        if cur and since <= parse_iso(cur["t_start"]) <= until:
            evs.append({**cur, "closed": False})
        for e in evs:
            peak = parse_iso(e["t_pmax"])
            has = gradcam_path(cfg.gradcam_dir, peak, True).exists()
            e["gradcam"] = f"/api/gradcam/{iso(peak)}.png" if has else None
        return {"events": evs}

    @app.get("/api/series")
    def series(days: float = Query(3, gt=0, le=40), start: str | None = None, end: str | None = None):
        """Probability per slot, compact: ``points`` = [[slot, probability
        or null, state, status], …], oldest first. Last ``days`` (≤ 40), or
        ``start``/``end`` (ISO, ≤ 40 days apart)."""
        rows = _rows(*_window(start, end, days, 40))
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
    fits_index: dict = {"built": 0.0, "paths": {}}

    def _indexed(name: str) -> Path | None:
        """FITS in ``CENTELLA_FITS_DIRS`` (e.g. the month a replay ran on),
        by file name; the index is rebuilt at most every 10 min."""
        if not cfg.fits_dirs:
            return None
        if time.time() - fits_index["built"] > 600:
            paths = {}
            for root in cfg.fits_dirs:
                for p in Path(root).rglob("AIA*.fits"):
                    paths.setdefault(p.name, p)
            fits_index.update(built=time.time(), paths=paths)
        return fits_index["paths"].get(name)

    def _slot_files(slot: dt.datetime) -> dict:
        d = cfg.staging_dir / day_key(slot)
        out = {}
        for w in CHANNELS:
            p = d / file_name(slot, w)
            out[w] = p if p.exists() else (_indexed(p.name) or p)
        return out

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
    def frames(hours: float = Query(6, gt=0, le=168), start: str | None = None, end: str | None = None):
        """Recent slots with what can be shown for each: AIA images
        (``/api/image/{wave}.png?slot=``) while the FITS are kept (7 days),
        Grad-CAM (``/api/gradcam/{slot}.png``) while it is kept (2 days;
        alerting frames 400 days). ``start``/``end`` (ISO, ≤ 7 days apart)
        select any window."""
        out = []
        for r in _rows(*_window(start, end, hours / 24, 7)):
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

    # -- retrospective viewer ----------------------------------------------
    @app.get("/api/range")
    def data_range():
        """First and last slot with a result: the span the viewer can browse."""
        files = sorted(cfg.results_dir.glob("*.jsonl"))
        if not files:
            return {"first": None, "last": None, "days": 0}
        store = ResultStore(cfg.results_dir)
        first = next((r["slot"] for f in files for r in store.read_day(f.stem)[:1]), None)
        last = next((store.read_day(f.stem)[-1]["slot"] for f in reversed(files) if store.read_day(f.stem)), None)
        return {"first": first, "last": last, "days": len(files)}

    @app.get("/api/frame/{slot}")
    def frame(slot: str):
        """Everything about one slot: its result row (all fields the loop
        wrote, or null), which channel images and differences can be shown,
        and whether there is a Grad-CAM."""
        s = _slot_param(slot)
        row = next((r for r in ResultStore(cfg.results_dir).read_day(day_key(s)) if r["slot"] == iso(s)), None)
        images, diffs = [], []
        for w in CHANNELS:
            _, path, bls = _pick(w, s)
            if path.exists():
                images.append(w)
                if sum(b.exists() for b in bls) >= cfg.min_baseline:
                    diffs.append(w)
        cam = gradcam_path(cfg.gradcam_dir, s, False).exists() or gradcam_path(cfg.gradcam_dir, s, True).exists()
        return {"slot": iso(s), "row": row, "images": images, "differences": diffs, "gradcam": cam}

    flare_cache: dict = {"key": None, "flares": []}

    def _flares():
        srcs = [Path(cfg.flares_csv)] if cfg.flares_csv else sorted(cfg.archive_dir.glob("*/*/*/flares.csv"))
        key = tuple((str(p), p.stat().st_mtime) for p in srcs if p.exists())
        if key != flare_cache["key"]:
            out = {}
            for p, _ in key:
                for f in read_csv(Path(p)):
                    out[f.flare_id] = f
            flare_cache.update(key=key, flares=sorted(out.values(), key=lambda f: f.tstart))
        return flare_cache["flares"]

    @app.get("/api/flares")
    def flares(start: str | None = None, end: str | None = None, days: float = Query(3, gt=0, le=40)):
        """GOES flares (C and above) in a window, for comparison:
        ``CENTELLA_FLARES_CSV`` if set, else the curated archive's lists."""
        since, until = _window(start, end, days, 40)
        return {"source": cfg.flares_csv or "archive", "flares": [
            {"id": f.flare_id, "class": f.fclass, "tstart": iso(f.tstart), "tpeak": iso(f.tpeak), "tend": iso(f.tend)}
            for f in _flares() if f.tend >= since and f.tstart <= until and f.flux >= class_flux("C1.0")]}

    @app.get("/viewer", response_class=HTMLResponse)
    def viewer():
        """Retrospective viewer (internal): scrub through the results, AIA
        channels, differences, Grad-CAM and GOES flares of any period."""
        return HTMLResponse((resources.files("centella") / "static" / "viewer.html").read_text(encoding="utf-8"))

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
