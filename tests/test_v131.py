"""1.3.1: frames flagged Q_NOASD are processed; the loop says when it
produces no result; the archive fallback reads JSOC's near-real-time
archive; downloads have a total time limit and the loop feeds systemd's
watchdog; the outage's slots can be reprocessed."""

import datetime as dt
import json
import logging
import os
import socket
from pathlib import Path

import pytest
from conftest import T0, FakeSession, write_fits, write_slot
from fastapi.testclient import TestClient

from flare_nrt.grid import CADENCE, CHANNELS, file_url, iso, parse_iso
from flare_nrt.loop import Processor, run
from flare_nrt.model import Ensemble
from flare_nrt.reprocess import plan, reprocess
from flare_nrt.server import create_app
from flare_nrt.source import LiveSource, ReplaySource, StagingSource
from flare_nrt.store import ResultStore, append_event, claim_event_id, read_events, read_json, write_json_atomic
from flare_nrt.streak import NoResultWatch, duration

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]
NRT, NOASD = 0x40000000, 0x40000004          # QUALITY before / since 2026-11-24 23:00 UT
BASE = "https://jsoc1.stanford.edu/data/aia/synoptic"
UTC = dt.timezone.utc


def _rows(cfg, day=T0):
    return ResultStore(cfg.results_dir).read_day(f"{day:%Y%m%d}")


# -- 1. soft quality bits ------------------------------------------------------
def test_noasd_frames_give_results_and_rows_keep_quality(cfg):
    slots = [T0 + i * CADENCE for i in range(8)]
    for s in slots:
        write_slot(cfg.staging_dir, s, quality=NOASD)
    run(cfg, replay_dir=cfg.staging_dir)
    ok = [r for r in _rows(cfg) if r["status"] == "ok"]
    assert ok
    for r in _rows(cfg):
        assert r["quality"] == {str(w): NOASD for w in CHANNELS}
        assert r["quality_soft"] == ["noasd"]


def test_with_the_old_mask_noasd_frames_are_rejected(cfg):
    cfg.qc_ignore_quality_mask = NRT                  # what 1.0.0–1.3.0 did
    for i in range(3):
        write_slot(cfg.staging_dir, T0 + i * CADENCE, quality=NOASD)
    run(cfg, replay_dir=cfg.staging_dir)
    rows = _rows(cfg)
    assert {r["status"] for r in rows} == {"qc_reject"}
    assert "quality:131=1073741828" in rows[0]["qc"] and rows[0]["quality"]["131"] == NOASD


# -- 2. say when every slot is rejected -----------------------------------------
def _live_processor(cfg):
    return Processor(cfg, Ensemble.load(cfg.weights, cfg.weights_sha256), ReplaySource(cfg.staging_dir), live=True)


def test_rejected_streak_warns_once_then_resumes(cfg, caplog):
    slots = [T0 + i * CADENCE for i in range(22)]
    for s in slots:
        write_slot(cfg.staging_dir, s, quality=NOASD)
    cfg.qc_ignore_quality_mask = NRT                  # the outage: 1.3.0 rejects every frame
    proc = _live_processor(cfg)
    client = TestClient(create_app(cfg))
    caplog.set_level(logging.INFO, logger="flare_nrt.loop")
    for s in slots[:15]:
        proc.process(s, s + dt.timedelta(minutes=1))
    warns = [r for r in caplog.records if r.levelno == logging.WARNING and "without a result" in r.message]
    assert len(warns) == 1
    assert warns[0].message.startswith("30 min without a result (since 2026-10-20T12:00:00Z")
    assert "quality:131=1073741828 ×15" in warns[0].message
    st = client.get("/api/status").json()
    assert st["rejected_streak"] == 15 and st["rejected_since"] == iso(slots[0])
    assert "quality:131=1073741828" in st["last_reject_reasons"]
    assert st["rejected_streak_reasons"]["quality:1600=1073741828"] == 15
    assert st["live_status"] == "no_data" and st["last_ok_slot"] is None

    proc.qc.ignore_mask = cfg.qc_ignore_quality_mask = NOASD   # 1.3.1
    caplog.clear()
    for s in slots[15:]:
        proc.process(s, s + dt.timedelta(minutes=1))
    rows = {r["slot"]: r for r in _rows(cfg)}
    # slots 15–19: their baselines (t−8…−12) were rejected; 20 is the first result again
    assert [rows[iso(s)]["status"] for s in slots[15:21]] == ["no_baseline"] * 5 + ["ok"]
    infos = [r for r in caplog.records if "results resumed" in r.message]
    assert len(infos) == 1 and infos[0].levelno == logging.INFO
    assert infos[0].message.startswith(f"results resumed at {iso(slots[20])} after 40 min without a result (20 slots")
    assert not [r for r in caplog.records if r.levelno == logging.WARNING and "without a result" in r.message]
    write_json_atomic(cfg.state_dir / "heartbeat.json", {"heartbeat": iso(slots[-1]), "next_slot": None,
                                                         "last_download": iso(slots[-1])})
    st = client.get("/api/status").json()
    assert st["rejected_streak"] == 0 and st["last_ok_slot"] == iso(slots[-1])
    assert st["heartbeat"] == st["last_download"] == iso(slots[-1])
    assert st["last_frame"]["quality_soft"] == ["noasd"]


def test_streak_warns_every_hour_while_it_lasts(tmp_path, caplog):
    w = NoResultWatch(tmp_path / "no_result.json", warn_slots=15)
    caplog.set_level(logging.INFO, logger="flare_nrt.loop")
    t = T0
    for _ in range(15 + 30 + 30 + 10):           # 2 h 50 min of rejects, then missing files
        w.update({"slot": iso(t), "status": "qc_reject", "qc": ["quality:131=1073741828", "quality:171=1073741828"]})
        t += CADENCE
    for _ in range(5):
        w.update({"slot": iso(t), "status": "missing", "channels_missing": [131]})
        t += CADENCE
    warns = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert [m.split(" without")[0] for m in warns] == ["30 min", "1 h 30 min", "2 h 30 min"]
    assert "quality:131=1073741828 ×75, quality:171=1073741828 ×75" in warns[-1]
    w.update({"slot": iso(t), "status": "ok"})
    assert "after 3 h without a result (90 slots" in caplog.records[-1].message
    assert "missing:131 ×5" in caplog.records[-1].message
    assert read_json(tmp_path / "no_result.json") == {
        "last_ok_slot": iso(t), "rejected_streak": 0, "rejected_since": None,
        "rejected_streak_reasons": {}, "last_reject_reasons": ["missing:131"]}
    assert [duration(n) for n in (1, 15, 30, 720, 750)] == ["2 min", "30 min", "1 h", "1 d", "1 d 1 h"]


def test_streak_is_rebuilt_from_results_on_upgrade(cfg, caplog):
    store = ResultStore(cfg.results_dir)
    store.append({"slot": iso(T0), "status": "ok"})
    for i in range(1, 21):
        store.append({"slot": iso(T0 + i * CADENCE), "status": "qc_reject", "qc": ["quality:304=1073741828"]})
    write_json_atomic(cfg.status_path, {"observation_time": iso(T0)})
    w = NoResultWatch(cfg.state_dir / "no_result.json")
    caplog.set_level(logging.INFO, logger="flare_nrt.loop")
    w.bootstrap(store, iso(T0), T0 + dt.timedelta(hours=1))
    assert not caplog.records                    # quiet …
    w.announce()                                 # … then said once at start
    assert caplog.records[-1].message.startswith("starting with 40 min without a result (since 2026-10-20T12:02:00Z")
    st = TestClient(create_app(cfg)).get("/api/status").json()
    assert st["rejected_streak"] == 20 and st["last_ok_slot"] == iso(T0)


# -- 3. the archive fallback reads nrt/ ----------------------------------------
def _blobs(tmp_path):
    out = {}
    for w in CHANNELS:
        p = tmp_path / f"{w}.fits"
        write_fits(p, w, T0)
        out[w] = p.read_bytes()
    return out


def _nrt_answers(tmp_path, hours=("H2200", "H2300")):
    """The recorded listings of 2026-11-24 H2200 and H2300, and every image
    they list in our channels."""
    from flare_nrt.grid import parse_nrt_listing

    blobs, answers = _blobs(tmp_path), {}
    for h in hours:
        url = f"{BASE}/nrt/2026/11/24/{h}/"
        html = (FIX / f"jsoc_nrt_20261124_{h}.html").read_bytes()
        answers[url] = html
        for _, w, name in parse_nrt_listing(html.decode()):
            if w in CHANNELS:
                answers[url + name] = blobs[w]
    return answers


def test_archive_fills_gaps_from_nrt_listing(cfg, tmp_path):
    now = dt.datetime(2026, 11, 24, 23, 20, 30, tzinfo=UTC)
    live = LiveSource(cfg.staging_dir, BASE, 30, 5, archive_budget_sec=600)
    live.session = FakeSession(_nrt_answers(tmp_path))
    n = live.refresh_archive(now)
    slots = [dt.datetime(2026, 11, 24, 22, 50, tzinfo=UTC) + i * CADENCE for i in range(15)]   # 22:50 … 23:18
    assert all(live.paths(s) is not None for s in slots)
    files = [u for u in live.session.calls if u.endswith(".fits")]
    assert all("/nrt/" in u for u in live.session.calls)               # the dated archive (404 today) is not asked
    assert len(files) == len(set(files)) == n == 10 * 4                # 3-min images: 10 fill 15 slots
    # 23:03 stands for 23:02 and 23:04 (60 s each); 23:00 for itself
    assert f"{BASE}/nrt/2026/11/24/H2300/AIA20261124_230300_0131.fits" in files
    assert f"{BASE}/nrt/2026/11/24/H2200/AIA20261124_225100_1600.fits" in files
    calls = len(live.session.calls)
    assert live.refresh_archive(now) == 0
    assert len(live.session.calls) == calls                            # nothing missing: not even a listing


def test_archive_uses_the_dated_archive_for_hours_nrt_lacks(cfg, tmp_path):
    now = dt.datetime(2026, 10, 1, 12, 9, tzinfo=UTC)
    slots = [dt.datetime(2026, 10, 1, 12, 2, tzinfo=UTC) + i * CADENCE for i in range(3)]
    blobs = _blobs(tmp_path)
    live = LiveSource(cfg.staging_dir, BASE, 6, 5, archive_budget_sec=600)
    live.session = FakeSession({file_url(BASE, s, w): blobs[w] for s in slots for w in CHANNELS})
    assert live.refresh_archive(now) == 12
    assert all(live.paths(s) is not None for s in slots)
    assert f"{BASE}/nrt/2026/10/01/H1200/" in live.session.calls


def test_archive_answers_other_than_200_are_logged_hourly(cfg, tmp_path, caplog):
    answers = {u: (v if u.endswith("/") else 503) for u, v in _nrt_answers(tmp_path).items()}
    live = LiveSource(cfg.staging_dir, BASE, 10, 5, archive_budget_sec=600)
    live.session = FakeSession(answers)
    caplog.set_level(logging.WARNING, logger="flare_nrt.source")
    t = dt.datetime(2026, 11, 24, 23, 20, tzinfo=UTC)
    for minutes in (0, 10, 40, 70):          # 10: within the 30-min retry pause; 40: < 1 h since the report
        live.refresh_archive(t + dt.timedelta(minutes=minutes))
    logged = [r.message for r in caplog.records if "other than 200" in r.message]
    assert len(logged) == 2
    # 23:10 … 23:18, 4 channels each, newest first: the example is the last one asked
    assert logged[0] == ("archive: 20 answer(s) other than 200 since 2026-11-24T23:20:00Z (503 ×20), "
                         f"e.g. HTTP 503 {BASE}/nrt/2026/11/24/H2300/AIA20261124_230900_1600.fits")
    # 23:30 and 00:00: new slots fail too (the 00:00 hour is in neither archive here: 404), one line for all
    assert logged[1].startswith("archive: 60 answer(s) other than 200 since 2026-11-24T23:30:00Z (404 ×20, 503 ×40)")


# -- 4. hangs ------------------------------------------------------------------
def test_a_download_has_a_total_time_limit(cfg, caplog):
    url = "https://example.invalid/synoptic/mostrecent/AIAsynoptic0131.fits"
    live = LiveSource(cfg.staging_dir, "https://example.invalid/synoptic", 30, 5, max_sec=0.2)
    live.session = FakeSession({url: b"x" * (1 << 16) * 10}, chunk_delay=0.05)    # 10 reads, none slow
    with pytest.raises(TimeoutError):
        live.get(url)
    assert live.refresh_mostrecent() == 0
    assert "mostrecent 131 failed twice: download over 0.2 s" in caplog.text


@pytest.fixture
def notify_socket(monkeypatch):
    name = f"flare-nrt-test-{os.getpid()}-{id(monkeypatch)}"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.bind("\0" + name)
    s.setblocking(False)
    monkeypatch.setenv("NOTIFY_SOCKET", "@" + name)
    yield s
    s.close()


def _received(sock) -> list[str]:
    out = []
    while True:
        try:
            out.append(sock.recv(256).decode())
        except BlockingIOError:
            return out


def test_sd_notify(notify_socket, monkeypatch):
    from flare_nrt.sdnotify import notify

    assert notify("WATCHDOG=1") and _received(notify_socket) == ["WATCHDOG=1"]
    monkeypatch.delenv("NOTIFY_SOCKET")
    assert notify("WATCHDOG=1") is False


def test_live_loop_says_ready_and_feeds_the_watchdog(cfg, notify_socket, monkeypatch):
    import flare_nrt.loop as loop

    class Offline(StagingSource):
        def __init__(self, staging_dir, *a, **k):
            super().__init__(staging_dir)
            self.last_download = None

        def refresh_mostrecent(self):
            return 0

    monkeypatch.setattr(loop, "LiveSource", Offline)
    write_json_atomic(cfg.state_dir / "heartbeat.json", {"heartbeat": "2026-11-24T22:58:30Z",
                                                         "last_download": "2026-11-24T22:58:10Z"})
    run(cfg, max_iterations=1)
    msgs = _received(notify_socket)
    assert msgs[0] == "READY=1" and msgs.count("WATCHDOG=1") >= 2
    hb = read_json(cfg.state_dir / "heartbeat.json")
    assert hb["last_download"] == "2026-11-24T22:58:10Z"          # kept across a restart, not None


def test_loop_unit_has_the_watchdog():
    unit = (ROOT / "deploy" / "flare-nrt-loop.service").read_text()
    for line in ("Type=notify", "WatchdogSec=600", "Restart=always"):
        assert f"\n{line}\n" in unit, line


# -- 5. reprocess the gap ------------------------------------------------------
def test_reprocess_the_gap(cfg):
    slots = [T0 + i * CADENCE for i in range(16)]
    for s in slots:
        write_slot(cfg.staging_dir, s, quality=NOASD)
    write_json_atomic(cfg.loop_state_path, {"last_slot": None, "last_id": 3, "current": None})
    cfg.qc_ignore_quality_mask = NRT                         # the outage, as 1.3.0 ran it
    live = _live_processor(cfg)
    for s in slots:
        live.process(s, s + dt.timedelta(minutes=1))
    assert {r["status"] for r in _rows(cfg)} == {"qc_reject"}
    state = {p.name: p.read_bytes() for p in cfg.state_dir.glob("*.json")}

    cfg.qc_ignore_quality_mask = NOASD                       # 1.3.1
    cfg.threshold = 0.0                                      # stand-in weights: every result alerts
    todo, seen = plan(cfg, slots[0], slots[-1])
    assert todo == slots and seen == {"qc_reject": 16}
    out = reprocess(cfg, Ensemble.load(cfg.weights, cfg.weights_sha256), slots[0], slots[-1])
    assert out == {"no_baseline": 5, "ok": 11, "events": 1}   # 0–4 have no t−8…−12 frames
    rows = _rows(cfg)
    assert len(rows) == 16 and all(r["reprocessed"] for r in rows)
    assert [r["status"] for r in rows] == ["no_baseline"] * 5 + ["ok"] * 11
    assert rows[-1]["quality_soft"] == ["noasd"] and rows[-1]["event_id"] == 4
    every = ResultStore(cfg.results_dir).read_day(f"{T0:%Y%m%d}", every_row=True)
    assert [r["status"] for r in every[:16]] == ["qc_reject"] * 16 and len(every) == 32
    # the live loop's state is untouched
    assert {p.name: p.read_bytes() for p in cfg.state_dir.glob("*.json") if p.name in state} == state
    # readers see the new rows
    client = TestClient(create_app(cfg))
    win = f"start={iso(slots[0])}&end={iso(slots[-1])}"
    pts = client.get(f"/api/series?{win}").json()["points"]
    assert [p[3] for p in pts] == ["no_baseline"] * 5 + ["ok"] * 11 and pts[-1][1] is not None
    frames = client.get(f"/api/frames?{win}").json()["frames"]
    assert [f["status"] for f in frames][-1] == "ok" and frames[-1]["probability"] is not None
    assert client.get(f"/api/frame/{iso(slots[-1])}").json()["row"]["reprocessed"] is True
    # the gap's event, with an id the loop hasn't used; the loop's next one comes after it
    evs = read_events(cfg.events_path)
    assert [(e["event_id"], e["t_start"], e["n_alert_frames"]) for e in evs] == [(4, iso(slots[5]), 11)]
    assert claim_event_id(cfg.events_path, cfg.state_dir / "event_ids.json", 4) == 5
    # run again: only the slots that still have no baseline
    assert plan(cfg, slots[0], slots[-1])[0] == slots[:5]
    # slots after the loop's newest one are the loop's, not the command's
    write_slot(cfg.staging_dir, slots[-1] + CADENCE, quality=NOASD)
    assert plan(cfg, slots[0], slots[-1] + 5 * CADENCE)[0] == slots[:5]


def test_reprocess_skips_slots_whose_fits_are_gone(cfg):
    for i in range(3):
        write_slot(cfg.staging_dir, T0 + i * CADENCE)
    store = ResultStore(cfg.results_dir)
    for i in range(5):
        store.append({"slot": iso(T0 + i * CADENCE), "status": "ok" if i == 1 else "qc_reject"})
    todo, seen = plan(cfg, T0, T0 + 5 * CADENCE)
    assert todo == [T0, T0 + 2 * CADENCE]
    assert seen == {"qc_reject": 2, "ok": 1, "qc_reject, FITS gone": 2, "none, FITS gone": 1}


def test_reprocess_command_dry_run(cfg, capsys):
    from flare_nrt.reprocess import main

    write_slot(cfg.staging_dir, T0)
    ResultStore(cfg.results_dir).append({"slot": iso(T0), "status": "qc_reject"})
    assert main(["--since", iso(T0), "--until", iso(T0 + CADENCE), "--dry-run", "--data-dir", str(cfg.data_dir)]) == 0
    assert "1 to reprocess, 2026-10-20T12:00:00Z → 2026-10-20T12:00:00Z" in capsys.readouterr().out


def test_last_row_of_a_slot_is_its_result(tmp_path):
    store = ResultStore(tmp_path)
    a, b = iso(T0), iso(T0 + CADENCE)
    assert store.append({"slot": a, "status": "qc_reject"}) and store.append({"slot": b, "status": "ok"})
    assert not store.append({"slot": a, "status": "ok"})                  # a restart doesn't write twice
    assert store.append({"slot": a, "status": "ok", "reprocessed": True}, supersede=True)
    assert store.read_day(f"{T0:%Y%m%d}") == [{"slot": a, "status": "ok", "reprocessed": True},
                                              {"slot": b, "status": "ok"}]
    assert [r["status"] for r in ResultStore(tmp_path).read_day(f"{T0:%Y%m%d}", every_row=True)] == \
        ["qc_reject", "ok", "ok"]


def test_event_ids_are_shared_and_events_listed_in_start_order(cfg):
    from flare_nrt.detector import Event

    ids = cfg.state_dir / "event_ids.json"
    assert claim_event_id(cfg.events_path, ids, 4) == 4       # the loop opens event 4 …
    assert claim_event_id(cfg.events_path, ids, 4) == 5       # … the reprocess, starting from the same state, 5
    assert claim_event_id(cfg.events_path, ids, 9) == 9
    later = Event(4, iso(T0 + 60 * CADENCE), iso(T0 + 61 * CADENCE), iso(T0 + 61 * CADENCE), 0.9, 2, None, True)
    gap = Event(5, iso(T0), iso(T0 + CADENCE), iso(T0), 0.95, 2, None, True)
    append_event(cfg.events_path, later)
    append_event(cfg.events_path, gap)                        # the gap's, closed later
    write_json_atomic(cfg.loop_state_path, {"last_slot": None, "last_id": 4, "current": None})
    evs = TestClient(create_app(cfg)).get(f"/api/events?start={iso(T0)}&end={iso(T0 + 70 * CADENCE)}").json()
    assert [e["event_id"] for e in evs["events"]] == [5, 4]
    assert parse_iso(evs["events"][0]["t_start"]) == T0
    assert json.loads(ids.read_text()) == {"last_id": 9}
