"""Stored synthetic input → full inference path → results, events, API,
curated archive. Uses the tiny stand-in ensemble (conftest); the real-weights
check is tests/test_real_weights.py (skipped unless CENTELLA_REAL_WEIGHTS is set)."""

import datetime as dt
import json

from conftest import T0, write_slot
from fastapi.testclient import TestClient

from centella import curate
from centella.goes import Flare
from centella.grid import CADENCE, iso
from centella.loop import run
from centella.server import create_app, live_status
from centella.store import ResultStore, read_events, read_json, write_json_atomic


def _make_day(cfg, n=16, bad=(5,), flare_from=12):
    slots = [T0 + i * CADENCE for i in range(n)]
    for i, s in enumerate(slots):
        write_slot(cfg.staging_dir, s, quality=65536 if i in bad else 0,
                   flare=400.0 if i >= flare_from else 0.0)
    return slots


def test_replay_produces_consistent_outputs(cfg):
    slots = _make_day(cfg)
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
        assert r["baseline_slots"][0] == iso(dt.datetime.fromisoformat(r["slot"].replace("Z", "+00:00"))
                                             - dt.timedelta(minutes=8))
    evs = read_events(cfg.events_path)
    assert len(evs) >= 1 and evs[-1]["n_alert_frames"] >= 1
    st = read_json(cfg.status_path)
    assert st["observation_time"] == ok[-1]["slot"] and st["threshold"] == 0.0


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
    now = iso(dt.datetime.now(dt.UTC))
    write_json_atomic(cfg.state_dir / "heartbeat.json", {"heartbeat": now})
    s = client.get("/api/status").json()
    # the synthetic slots are days old: a running loop on old inputs is stale too
    assert s["live_status"] == "stale" and s["stale_reason"] == "inputs_old"
    st = read_json(cfg.status_path)
    write_json_atomic(cfg.status_path, dict(st, observation_time=now))
    s = client.get("/api/status").json()
    assert s["live_status"] == "live"
    for k in ("timestamp", "probability", "flare_detected", "location", "ensemble_std",
              "seconds_since_update", "threshold", "observation_time", "model_version", "model"):
        assert k in s
    assert s["timestamp"].endswith("Z") and s["model"] == "Centella"
    assert client.get("/api/events").status_code == 200
    assert "points" in client.get("/api/results?hours=168").json()
    assert "coverage_24h" in client.get("/api/health").json()


def test_live_status_rules():
    now = dt.datetime(2026, 10, 20, 12, 0, tzinfo=dt.UTC)
    st = {"timestamp": "2026-10-20T11:58:00Z", "last_update": "2026-10-20T11:58:00Z"}
    hb = {"heartbeat": "2026-10-20T11:59:30Z"}
    assert live_status(st, hb, now, 600)[0] == "live"
    assert live_status(st, {"heartbeat": "2026-10-20T11:40:00Z"}, now, 600)[:2] == ("stale", "loop_not_running")
    old = {"timestamp": "2026-10-20T11:30:00Z", "last_update": "2026-10-20T11:30:00Z"}
    assert live_status(old, hb, now, 600)[:2] == ("stale", "no_new_result")
    assert live_status(None, hb, now, 600)[0] == "no_data"
    late = dict(st, observation_time="2026-10-20T10:30:00Z")
    assert live_status(late, hb, now, 600)[:2] == ("stale", "inputs_old")


def test_missing_slot_is_skipped_once_later_slots_arrive(cfg):
    from centella.loop import should_skip
    from centella.source import ReplaySource

    slots = [T0 + i * CADENCE for i in range(4)]
    for s in slots[1:]:
        write_slot(cfg.staging_dir, s)
    src = ReplaySource(cfg.staging_dir)
    assert not should_skip(src, slots[0], slots[0] + dt.timedelta(minutes=10), cfg)
    assert should_skip(src, slots[0], slots[0] + dt.timedelta(minutes=25), cfg)
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
    assert s["schema"] == "centella-archive/1"
    assert s["slots"]["kept_by_reason"].get("flare", 0) == 16  # raw kept even for the QC-rejected slot
    assert len(s["files"]) == 4 * s["slots"]["kept"]
    assert all(len(h) == 64 for h in s["files"].values())
    lines = (out / "frames.csv").read_text().splitlines()
    assert len(lines) == 721
    assert json.loads((out / "day.json").read_text())["labels"]["by_letter"]["M"] == 1
