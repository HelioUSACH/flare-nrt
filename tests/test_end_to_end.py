"""Stored synthetic input → full inference path → results, events, API,
curated archive. Uses the tiny stand-in ensemble (conftest); the real-weights
check is tests/test_real_weights.py (skipped unless FLARE_NRT_REAL_WEIGHTS is set)."""

import datetime as dt
import json

from conftest import T0, write_slot
from fastapi.testclient import TestClient

from flare_nrt import curate
from flare_nrt.goes import Flare
from flare_nrt.grid import CADENCE, iso, parse_iso
from flare_nrt.loop import run
from flare_nrt.server import create_app, live_status
from flare_nrt.store import ResultStore, read_events, read_json, write_json_atomic


def _make_day(cfg, n=16, bad=(5,), flare_from=12):
    slots = [T0 + i * CADENCE for i in range(n)]
    for i, s in enumerate(slots):
        write_slot(cfg.staging_dir, s, quality=65536 if i in bad else 0,
                   flare=400.0 if i >= flare_from else 0.0)
    return slots


def test_replay_produces_consistent_outputs(cfg):
    slots = _make_day(cfg)
    cfg.min_baseline = 3  # the validated rule: all 3 baseline frames
    cfg.threshold = 0.0  # stand-in weights: force alerts to exercise events
    proc = run(cfg, replay_dir=cfg.staging_dir)
    rows = ResultStore(cfg.results_dir).read_day(f"{T0:%Y%m%d}")
    by = {r["slot"]: r for r in rows}
    assert len(rows) == len(slots)
    # t−8, −10, −12 needed: the first 6 slots have no baseline
    assert all(by[iso(s)]["status"] == "no_baseline" for s in slots[:5])
    assert by[iso(slots[5])]["status"] == "qc_reject"
    assert by[iso(slots[5])]["qc"][0].startswith("quality:")
    # slots whose baseline includes the rejected slot 5 (t−8/−10/−12 → i = 9, 10, 11)
    assert all(by[iso(slots[i])]["status"] == "no_baseline" for i in (9, 10, 11))
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) == proc.frame_count == 7  # i = 6, 7, 8, 12…15
    for r in ok:
        assert 0 <= r["probability"] <= 1 and len(r["ensemble_scores"]) == 2
        assert r["baseline_slots"][0] == iso(parse_iso(r["slot"])
                                             - dt.timedelta(minutes=8))
    evs = read_events(cfg.events_path)
    assert len(evs) >= 1 and evs[-1]["n_alert_frames"] >= 1
    st = read_json(cfg.status_path)
    assert st["observation_time"] == ok[-1]["slot"] and st["threshold"] == 0.0
    assert {r["state"] for r in rows} <= {"quiet", "onset", "active", "decay"}


def test_two_of_three_baselines_keep_the_slot(cfg):
    slots = _make_day(cfg)          # slot 5 is rejected by QC
    run(cfg, replay_dir=cfg.staging_dir)   # default min_baseline = 2
    by = {r["slot"]: r for r in ResultStore(cfg.results_dir).read_day(f"{T0:%Y%m%d}")}
    for i in (9, 10, 11):
        r = by[iso(slots[i])]
        assert r["status"] == "ok" and r["n_baseline_used"] == 2
        assert r["baseline_missing"] == [iso(slots[5])]
    assert by[iso(slots[12])]["n_baseline_used"] == 3
    assert by[iso(slots[4])]["status"] == "no_baseline"    # only 1 of 3


def test_replay_is_deterministic(cfg, tmp_path):
    _make_day(cfg, n=10, bad=())
    run(cfg, replay_dir=cfg.staging_dir)
    a = [r.get("probability") for r in ResultStore(cfg.results_dir).read_day(f"{T0:%Y%m%d}")]
    cfg.data_dir = tmp_path / "var2"
    cfg.ensure_dirs()
    run(cfg, replay_dir=tmp_path / "var" / "staging")
    b = [r.get("probability") for r in ResultStore(cfg.results_dir).read_day(f"{T0:%Y%m%d}")]
    assert a == b and any(p is not None for p in a)


def test_api_contract_and_freshness(cfg):
    client = TestClient(create_app(cfg))
    assert client.get("/api/status").json()["live_status"] == "no_data"
    _make_day(cfg, n=10, bad=())
    run(cfg, replay_dir=cfg.staging_dir)
    # replay writes no heartbeat → a dead loop must read as stale
    s = client.get("/api/status").json()
    assert s["live_status"] == "stale" and s["stale_reason"] == "loop_not_running"
    now = iso(dt.datetime.now(dt.timezone.utc))
    write_json_atomic(cfg.state_dir / "heartbeat.json", {"heartbeat": now})
    s = client.get("/api/status").json()
    # the synthetic slots are days old: a running loop on old inputs is stale too
    assert s["live_status"] == "stale" and s["stale_reason"] == "inputs_old"
    st = read_json(cfg.status_path)
    assert all(v.endswith("Z") for v in st["inputs"].values())        # provider T_OBS, UTC Z
    write_json_atomic(cfg.status_path, dict(st, observation_time=now, inputs={w: now for w in st["inputs"]}))
    s = client.get("/api/status").json()
    assert s["live_status"] == "live" and s["input_age_s"] < 60
    # the portal's adapter reads these (D-096) …
    for k in ("live_status", "timestamp", "probability", "flare_detected", "location", "ensemble_std",
              "seconds_since_update"):
        assert k in s
    # … and these are the 2.1 contract (INTEGRATION.md)
    for k in ("state", "raw_score", "alert", "threshold", "observation_time", "inputs", "input_age_s",
              "max_input_age_sec", "last_frame", "baseline_slots", "n_baseline_used", "model",
              "model_version", "weights_sha256", "server_time"):
        assert k in s, k
    assert s["state"] in {"quiet", "onset", "active", "decay"}
    assert s["timestamp"].endswith("Z") and s["server_time"].endswith("Z") and s["model"] == "flare-nrt"
    assert {"slot", "status", "reason", "inputs"} <= set(s["last_frame"])
    assert "severity" not in s            # never sent (exploratory in early internal versions, removed)
    assert client.get("/api/events").status_code == 200
    assert "points" in client.get("/api/results?hours=168").json()
    assert "coverage_24h" in client.get("/api/health").json()


def test_live_status_rules():
    now = dt.datetime(2026, 10, 20, 12, 0, tzinfo=dt.timezone.utc)
    st = {"timestamp": "2026-10-20T11:58:00Z", "last_update": "2026-10-20T11:58:00Z"}
    hb = {"heartbeat": "2026-10-20T11:59:30Z"}
    assert live_status(st, hb, now, 600)[0] == "live"
    assert live_status(st, {"heartbeat": "2026-10-20T11:40:00Z"}, now, 600)[:2] == ("stale", "loop_not_running")
    old = {"timestamp": "2026-10-20T11:30:00Z", "last_update": "2026-10-20T11:30:00Z"}
    assert live_status(old, hb, now, 600)[:2] == ("stale", "no_new_result")
    assert live_status(None, hb, now, 600)[0] == "no_data"
    late = dict(st, observation_time="2026-10-20T10:30:00Z")
    assert live_status(late, hb, now, 600)[:2] == ("stale", "inputs_old")
    # the newest provider T_OBS decides, 900 s by default (15 min)
    fresh = dict(st, inputs={"131": "2026-10-20T11:46:00.000Z", "171": "2026-10-20T11:44:00.000Z"})
    assert live_status(fresh, hb, now, 600)[0] == "live"                # 14 min
    old_in = dict(st, inputs={"131": "2026-10-20T11:44:00.000Z", "171": "2026-10-20T11:44:30.000Z"})
    assert live_status(old_in, hb, now, 600)[:2] == ("stale", "inputs_old")   # 15.5 min


def test_missing_slot_is_skipped_once_later_slots_arrive(cfg):
    from flare_nrt.loop import should_skip
    from flare_nrt.source import ReplaySource

    slots = [T0 + i * CADENCE for i in range(4)]
    for s in slots[1:]:
        write_slot(cfg.staging_dir, s)
    src = ReplaySource(cfg.staging_dir)
    assert not should_skip(src, slots[0], slots[0] + dt.timedelta(minutes=10), cfg)
    assert should_skip(src, slots[0], slots[0] + dt.timedelta(minutes=25), cfg)
    far = T0 - 30 * CADENCE    # cold start: next file arrives an hour later
    assert should_skip(src, far, far + dt.timedelta(minutes=25), cfg)
    lone = T0 + 100 * CADENCE  # nothing after it: wait up to give_up_min
    assert not should_skip(src, lone, lone + dt.timedelta(minutes=60), cfg)
    assert should_skip(src, lone, lone + dt.timedelta(minutes=121), cfg)


def test_curation_builds_archive_day(cfg):
    slots = _make_day(cfg, n=16, bad=(5,))
    run(cfg, replay_dir=cfg.staging_dir)
    day = T0.replace(hour=0)
    f = Flare(slots[10], slots[12], slots[14], "M1.0")
    s = curate.curate_day(cfg, day, [f], "test", True)
    out = cfg.archive_dir / "2026" / "10" / "20"
    assert (out / "day.json").exists() and (out / "frames.csv").exists() and (out / "flares.csv").exists()
    assert s["schema"] == "flare-nrt-archive/1"
    assert s["slots"]["kept_by_reason"].get("flare", 0) == 16  # raw kept even for the QC-rejected slot
    assert len(s["files"]) == 4 * s["slots"]["kept"]
    assert all(len(h) == 64 for h in s["files"].values())
    lines = (out / "frames.csv").read_text().splitlines()
    assert len(lines) == 721
    assert json.loads((out / "day.json").read_text())["labels"]["by_letter"]["M"] == 1


def test_channel_previews(cfg):
    _make_day(cfg, n=10, bad=())
    run(cfg, replay_dir=cfg.staging_dir)  # replay reads the same staging tree the API renders from
    client = TestClient(create_app(cfg))
    ch = client.get("/api/channels").json()
    assert [c["wave"] for c in ch["channels"]] == [131, 171, 304, 1600]
    assert all(c["image"] and c["difference"] for c in ch["channels"])
    for kind in ("img", "diff"):
        r = client.get(f"/api/image/131.png?kind={kind}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
        from io import BytesIO

        from PIL import Image
        assert Image.open(BytesIO(r.content)).size == (512, 512)
    assert client.get("/api/image/94.png").status_code == 404
    assert client.get("/api/image/131.png?kind=raw").status_code == 422


def test_mostrecent_is_filed_under_its_own_slot(cfg, tmp_path):
    from flare_nrt.source import LiveSource

    slot = T0 + 3 * CADENCE
    src_dir = tmp_path / "jsoc"
    write_slot(src_dir, slot)
    blobs = {w: (src_dir / f"{slot:%Y%m%d}" / f"AIA{slot:%Y%m%d}_{slot:%H%M}_{w:04d}.fits").read_bytes()
             for w in (131, 171, 304, 1600)}

    class Resp:
        def __init__(self, code, content=b""):
            self.status_code, self.content = code, content

    class Session:
        headers = {}

        def get(self, url, timeout=None):
            for w, b in blobs.items():
                if url.endswith(f"/mostrecent/AIAsynoptic{w:04d}.fits"):
                    return Resp(200, b)
            return Resp(404)

    live = LiveSource(cfg.staging_dir, "https://example.invalid/synoptic", 30, 5)
    live.session = Session()
    assert live.refresh(slot + dt.timedelta(minutes=1)) == 4
    assert live.paths(slot) is not None and live.last_download is not None
    assert live.refresh(slot + dt.timedelta(minutes=2)) == 0     # same images: nothing new


def test_staging_hours_keeps_only_recent_files(cfg):
    from flare_nrt.retention import Retention

    cfg.staging_hours = 1
    old, new = T0, T0 + dt.timedelta(hours=3)
    write_slot(cfg.staging_dir, old)
    write_slot(cfg.staging_dir, new)
    Retention(cfg).run(now=new + dt.timedelta(minutes=5))
    left = sorted(p.name for p in cfg.staging_dir.glob("*/AIA*.fits"))
    assert len(left) == 4 and all(f"_{new:%H%M}_" in n for n in left)


def test_mostrecent_stands_for_every_slot_within_90s():
    from flare_nrt.source import slots_for

    s = T0
    assert slots_for(s + dt.timedelta(seconds=10)) == [s]                     # 110 s from the next
    assert slots_for(s + dt.timedelta(seconds=40)) == [s, s + CADENCE]        # 80 s from the next
    assert slots_for(s + dt.timedelta(seconds=100)) == [s + CADENCE]          # nearest is the next
    assert slots_for(s + dt.timedelta(seconds=70)) == [s + CADENCE, s]


def test_mostrecent_off_slot_fills_both_slots(cfg, tmp_path):
    from conftest import write_fits

    from flare_nrt.source import LiveSource

    t = T0 + dt.timedelta(seconds=50)       # stands for T0 (50 s) and T0+2 min (70 s)
    blobs = {}
    for w in (131, 171, 304, 1600):
        p = tmp_path / "jsoc" / f"{w}.fits"
        write_fits(p, w, t)
        blobs[w] = p.read_bytes()

    class Resp:
        def __init__(self, code, content=b""):
            self.status_code, self.content = code, content

    class Session:
        headers = {}

        def get(self, url, timeout=None):
            for w, b in blobs.items():
                if url.endswith(f"/mostrecent/AIAsynoptic{w:04d}.fits"):
                    return Resp(200, b)
            return Resp(404)

    live = LiveSource(cfg.staging_dir, "https://example.invalid/synoptic", 10, 5)
    live.session = Session()
    assert live.refresh(T0 + dt.timedelta(minutes=5)) == 4
    assert live.paths(T0) is not None and live.paths(T0 + CADENCE) is not None


def test_cold_start_does_not_wait_for_slots_before_start(cfg):
    from flare_nrt.loop import should_skip
    from flare_nrt.source import ReplaySource

    start = T0 + 10 * CADENCE
    write_slot(cfg.staging_dir, start)
    src = ReplaySource(cfg.staging_dir)
    early = start - 5 * CADENCE      # 10 min before start, nothing will fill it
    assert should_skip(src, early, start + dt.timedelta(minutes=1), cfg, started=start)
    assert not should_skip(src, early, start + dt.timedelta(minutes=1), cfg)


def test_previews_before_the_first_result(cfg):
    # only an image, no result yet: the image shows, its difference doesn't
    write_slot(cfg.staging_dir, T0)
    client = TestClient(create_app(cfg))
    r = client.get("/api/image/171.png?kind=img")
    assert r.status_code == 200 and r.headers["X-Observation-Time"] == iso(T0)
    assert client.get("/api/image/171.png?kind=diff").status_code == 404
    ch = client.get("/api/channels").json()["channels"]
    assert all(c["image"] and not c["difference"] for c in ch)
    # with t−8, −10, −12 staged the difference shows too
    later = T0 + 6 * CADENCE
    write_slot(cfg.staging_dir, T0 + CADENCE)
    write_slot(cfg.staging_dir, T0 + 2 * CADENCE)
    write_slot(cfg.staging_dir, later)
    r = client.get("/api/image/171.png?kind=diff")
    assert r.status_code == 200 and r.headers["X-Observation-Time"] == iso(later)


def test_retention_keeps_4_days_of_fits_2_of_tensors(cfg):
    from flare_nrt.retention import Retention

    now = dt.datetime(2026, 10, 28, 12, 0, tzinfo=dt.timezone.utc)

    def touch(p):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p

    keep, drop = [], []
    for days in (0, 4, 5):                      # FLARE_NRT_STAGING_DAYS default: 4
        d = now - dt.timedelta(days=days)
        (keep if days <= 4 else drop).append(touch(cfg.staging_dir / f"{d:%Y%m%d}" / f"AIA{d:%Y%m%d}_1200_0131.fits"))
    for days in (2, 3):
        d = now - dt.timedelta(days=days)
        (keep if days <= 2 else drop).append(touch(cfg.tensors_dir / f"{d:%Y%m%d}" / "1200.npz"))
        (keep if days <= 2 else drop).append(touch(cfg.gradcam_dir / f"{d:%Y%m%d}" / "1200.png"))
        keep.append(touch(cfg.gradcam_dir / f"{d:%Y%m%d}" / "1202_alert.png"))   # alerts: 400 days
    Retention(cfg).run(now=now)
    assert all(p.exists() for p in keep) and not any(p.exists() for p in drop)


def test_display_endpoints(cfg):
    """What the portal will show: recent frames, AIA images per slot,
    Grad-CAM (latest and per slot), the probability series, events."""
    from io import BytesIO

    from PIL import Image

    from flare_nrt.loop import Processor
    from flare_nrt.model import Ensemble
    from flare_nrt.source import ReplaySource

    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    t0 = now - dt.timedelta(minutes=now.minute % 2) - 9 * CADENCE
    slots = [t0 + i * CADENCE for i in range(10)]
    for s in slots:
        write_slot(cfg.staging_dir, s)
    cfg.threshold = 0.0   # every ok frame alerts: exercises the alert Grad-CAM too
    proc = Processor(cfg, Ensemble.load(cfg.weights, cfg.weights_sha256), ReplaySource(cfg.staging_dir), live=True)
    for s in slots:
        proc.process(s, s + dt.timedelta(minutes=1))
    write_json_atomic(cfg.state_dir / "heartbeat.json", {"heartbeat": iso(now)})
    c = TestClient(create_app(cfg))

    fr = c.get("/api/frames?hours=1").json()["frames"]
    assert [f["slot"] for f in fr] == [iso(s) for s in slots]
    ok = [f for f in fr if f["status"] == "ok"]
    assert ok and all(f["images"] == [131, 171, 304, 1600] and f["gradcam"] for f in ok)
    assert (cfg.tensors_dir / f"{slots[-1]:%Y%m%d}" / f"{slots[-1]:%H%M}.npz").exists()

    for url in (f"/api/image/171.png?slot={ok[0]['slot']}", f"/api/image/171.png?slot={ok[0]['slot']}&kind=diff",
                "/api/gradcam/latest.png", f"/api/gradcam/{ok[0]['slot']}.png"):
        r = c.get(url)
        assert r.status_code == 200, url
        assert Image.open(BytesIO(r.content)).size == (512, 512)
    assert c.get("/api/gradcam/latest.png").headers["X-Observation-Time"] == iso(slots[-1])
    assert c.get(f"/api/gradcam/{iso(slots[0])}.png").status_code == 404     # no baseline → no result
    assert c.get("/api/image/171.png?slot=yesterday").status_code == 422

    se = c.get("/api/series?days=1").json()
    assert se["fields"] == ["slot", "probability", "state", "status"] and len(se["points"]) == 10
    ev = c.get("/api/events").json()["events"]
    assert ev and ev[-1]["gradcam"] and c.get(ev[-1]["gradcam"]).status_code == 200


def test_retrospective_viewer(cfg, tmp_path):
    """A past period replayed from FITS kept elsewhere (FLARE_NRT_FITS_DIRS),
    browsed by date: range, series/events/flares windows, one frame's
    detail, its images, the alert Grad-CAM, the viewer page."""
    fits = tmp_path / "month"
    slots = [T0 + i * CADENCE for i in range(12)]
    for i, s in enumerate(slots):
        write_slot(fits, s, flare=400.0 if i >= 8 else 0.0)
    cfg.fits_dirs = [fits]
    flares = tmp_path / "flares.csv"
    flares.write_text("tstart,tpeak,tend,fclass\n2026-10-20T12:14:00Z,2026-10-20T12:20:00Z,2026-10-20T12:30:00Z,M1.5\n"
                      "2026-10-19T03:00:00Z,2026-10-19T03:05:00Z,2026-10-19T03:10:00Z,B5.0\n")
    cfg.flares_csv = str(flares)
    cfg.threshold = 0.0                      # stand-in weights: every ok frame alerts
    run(cfg, replay_dir=fits)
    rows = ResultStore(cfg.results_dir).read_day(f"{T0:%Y%m%d}")
    assert all("input_age_s" not in r for r in rows)          # meaningless in a replay
    ok = [r for r in rows if r["status"] == "ok"]
    assert not any(cfg.tensors_dir.rglob("*.npz"))              # replay keeps no tensors …
    assert all((cfg.gradcam_dir / f"{T0:%Y%m%d}" / f"{parse_iso(r['slot']):%H%M}_alert.png").exists() for r in ok)

    c = TestClient(create_app(cfg))
    rg = c.get("/api/range").json()
    assert (rg["first"], rg["last"]) == (iso(slots[0]), iso(slots[-1]))
    win = f"start={iso(T0)}&end={iso(slots[-1])}"
    assert len(c.get("/api/series?" + win).json()["points"]) == 12
    assert c.get("/api/events?" + win).json()["events"]
    fl = c.get("/api/flares?" + win).json()["flares"]
    assert [f["class"] for f in fl] == ["M1.5"]                 # B flares and other days left out
    assert c.get(f"/api/series?start={iso(T0)}&end={iso(T0 + dt.timedelta(days=41))}").status_code == 422
    assert c.get(f"/api/series?start={iso(slots[3])}&end={iso(slots[2])}").status_code == 422

    d = c.get(f"/api/frame/{ok[-1]['slot']}").json()
    assert d["row"]["probability"] == ok[-1]["probability"]
    assert d["images"] == [131, 171, 304, 1600] and d["differences"] == [131, 171, 304, 1600] and d["gradcam"]
    assert c.get(f"/api/frame/{iso(slots[0])}").json()["differences"] == []   # no baselines before it
    for kind in ("img", "diff"):                                 # rendered from the FITS in fits_dirs
        assert c.get(f"/api/image/304.png?slot={ok[-1]['slot']}&kind={kind}").status_code == 200
    assert c.get(f"/api/gradcam/{ok[-1]['slot']}.png").status_code == 200
    page = c.get("/viewer")
    assert page.status_code == 200 and "visor retrospectivo" in page.text and 'fetch(path)' in page.text


def test_replay_resume_skips_done_slots(cfg, tmp_path):
    fits = tmp_path / "f"
    for i in range(8):
        write_slot(fits, T0 + i * CADENCE)
    run(cfg, replay_dir=fits, replay_gradcam=False)
    n = len(ResultStore(cfg.results_dir).read_day(f"{T0:%Y%m%d}"))
    proc = run(cfg, replay_dir=fits, replay_gradcam=False, resume=True)
    assert proc.frame_count == 0 and len(ResultStore(cfg.results_dir).read_day(f"{T0:%Y%m%d}")) == n == 8


def test_live_flares_add_swpc_recent_list(cfg, monkeypatch):
    """Live: the curated archive covers up to yesterday; SWPC's 7-day list
    fills the recent days (and its failure is not the API's)."""
    import flare_nrt.server as srv
    from flare_nrt.goes import Flare

    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    recent = Flare(now - dt.timedelta(hours=3), now - dt.timedelta(hours=2, minutes=50),
                   now - dt.timedelta(hours=2, minutes=40), "M1.4")
    calls = []
    monkeypatch.setattr(srv, "fetch_swpc", lambda url, timeout=30: calls.append(url) or [recent])
    c = TestClient(create_app(cfg))
    r = c.get("/api/flares?days=1").json()
    assert r["source"] == "archive+swpc" and [f["class"] for f in r["flares"]] == ["M1.4"]
    c.get("/api/flares?days=1")
    assert len(calls) == 1                                   # cached 10 min
    old = c.get("/api/flares?start=2026-06-19T00:00:00Z&end=2026-06-20T00:00:00Z").json()
    assert old["source"] == "archive" and old["flares"] == []

    def down(url, timeout=30):
        raise OSError("no network")

    monkeypatch.setattr(srv, "fetch_swpc", down)
    assert TestClient(create_app(cfg)).get("/api/flares?days=1").status_code == 200
