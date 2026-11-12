"""1.2.0: event replays keep 131 Å per frame and, once per event, the peak
frame's 4 channels and Grad-CAM in both styles; the one-off command adds
them to replays built by 1.1.0. Display only: detection is untouched
(tests/test_snapshot.py)."""

import datetime as dt
import shutil
from io import BytesIO

import numpy as np
from conftest import T0
from fastapi.testclient import TestClient
from PIL import Image
from test_display import _event, _live, _synthetic_frames

from flare_nrt import replays
from flare_nrt.artifacts import cam_path, gradcam_path
from flare_nrt.grid import CADENCE, iso, parse_iso
from flare_nrt.retention import Retention
from flare_nrt.server import create_app
from flare_nrt.store import write_json_atomic

IMMUTABLE = "public, max-age=31536000, immutable"


def _alert_event(cfg, eid=11):
    """Live slots whose last 3 alert (threshold 0.5): an event at the end."""
    _, slots, rows = _live(cfg, n=12, threshold=0.5)
    alerts = [r for r in rows if r.get("alert")]
    assert alerts, "the stand-in ensemble must alert on the synthetic flare"
    top = max(alerts, key=lambda r: r["probability"])
    ev = {"event_id": eid, "t_start": alerts[0]["slot"], "t_end": alerts[-1]["slot"], "t_pmax": top["slot"],
          "p_max": top["probability"], "n_alert_frames": len(alerts), "location": None}
    return ev, slots, rows, parse_iso(top["slot"])


def _as_110(cfg, eid):
    """Turn a replay into what 1.1.0 left: no 131 Å frames, no peak set."""
    d = replays.replay_dir(cfg, eid)
    for p in d.rglob("131.webp"):
        p.unlink()
    shutil.rmtree(d / "peak", ignore_errors=True)
    doc = replays.read_replay(cfg, eid)
    for f in doc["frames"]:
        f["images"] = [c for c in f["images"] if c != "131"]
    doc.pop("peak")
    doc["missing"] = {k: v for k, v in doc["missing"].items() if k in ("171", "gradcam-heat")}
    doc["model_version"] = "1.1.0"
    write_json_atomic(d / "replay.json", doc)
    return doc


def _decoded(content: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(content)).convert("RGBA"), dtype=float)


def test_replay_has_131_per_frame_rendered_like_the_api(cfg):
    ev, slots, rows, _ = _alert_event(cfg)
    doc = replays.build(cfg, ev, slots[-1] + dt.timedelta(hours=2))
    staged = [f for f in doc["frames"] if parse_iso(f["slot"]) in slots]
    assert len(staged) == len(slots) and all("131" in f["images"] for f in staged)
    # the window's slots outside the staged ones have no FITS: listed as missing, like 171 Å
    assert doc["missing"]["131"] == doc["missing"]["171"] == [
        f["slot"] for f in [{"slot": iso(parse_iso(doc["window"]["start"]) + i * CADENCE)}
                            for i in range(doc["n_slots"])] if parse_iso(f["slot"]) not in slots]
    c = TestClient(create_app(cfg))
    for s in (slots[0], slots[-1]):
        r = c.get(f"/api/events/11/frames/{iso(s)}/131.webp")
        assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
        assert r.headers["cache-control"] == IMMUTABLE and r.headers["x-observation-time"] == iso(s)
        assert Image.open(BytesIO(r.content)).size == (512, 512)
        # the same file /api/image/131.png?kind=img&format=webp gives (512 px)
        assert r.content == c.get(f"/api/image/131.png?kind=img&format=webp&slot={iso(s)}").content
    assert c.get(f"/api/events/11/frames/{iso(slots[0] - CADENCE * 40)}/131.webp").status_code == 404
    assert c.get(f"/api/events/11/frames/{iso(slots[0])}/304.webp").status_code == 404     # not a frame channel


def test_peak_set_matches_the_live_endpoints(cfg):
    ev, slots, rows, peak = _alert_event(cfg, eid=12)
    doc = replays.build(cfg, ev, slots[-1] + dt.timedelta(hours=2))
    pk = doc["peak"]
    row = next(r for r in rows if r["slot"] == iso(peak))
    assert pk["slot"] == iso(peak) and pk["images"] == list(replays.PEAK_FILES)
    assert pk["probability"] == row["probability"]
    assert pk["t_obs"] == {w: row["inputs"][int(w)] for w in replays.PEAK_WAVES}
    assert doc["missing"]["peak"] == [] and 0 < pk["gradcam_peak"] <= 1
    c = TestClient(create_app(cfg))
    live = {w: f"/api/image/{w}.png?kind=img&format=webp&slot={iso(peak)}" for w in replays.PEAK_WAVES}
    live["gradcam-heat"] = f"/api/gradcam/{iso(peak)}.png?style=heat&format=webp"
    live["gradcam-overlay"] = f"/api/gradcam/{iso(peak)}.png?style=overlay&format=webp"
    for name, url in live.items():
        r = c.get(f"/api/events/12/peak/{name}.webp")
        assert r.status_code == 200 and r.headers["content-type"] == "image/webp", name
        assert r.headers["cache-control"] == IMMUTABLE and r.headers["x-observation-time"] == iso(peak)
        ref = c.get(url)
        assert r.content == ref.content, name                          # byte-identical to a current frame
        if name.startswith("gradcam-"):
            assert r.headers["x-gradcam-slot"] == iso(peak)
            assert r.headers["x-gradcam-prob"] == f"{row['probability']:.6f}"
        if name == "gradcam-heat":
            assert r.headers["x-gradcam-peak"] == ref.headers["x-gradcam-peak"]
    # the peak's heat frame is also a replay frame
    assert (replays.peak_path(cfg, 12, "gradcam-heat").read_bytes()
            == replays.frame_path(cfg, 12, peak, "gradcam-heat").read_bytes())
    # 404s
    assert c.get("/api/events/12/peak/171-diff.webp").status_code == 404           # not a peak file
    assert c.get("/api/events/99/peak/171.webp").status_code == 404                # no replay
    assert c.get("/api/events/x/peak/171.webp").status_code == 422
    replays.peak_path(cfg, 12, "304").unlink()
    assert c.get("/api/events/12/peak/304.webp").status_code == 404                # file not there
    _as_110(cfg, 12)
    assert c.get("/api/events/12/peak/171.webp").status_code == 404                # a 1.1.0 replay


def test_peak_overlay_recomputed_when_its_png_is_gone(cfg):
    """The one-off draws the overlay from a recomputed map and its stored
    input when the PNG is gone; it matches the loop's own overlay."""
    from flare_nrt.display import FitsLocator
    from flare_nrt.model import Ensemble

    ev, slots, rows, peak = _alert_event(cfg, eid=13)
    now = slots[-1] + dt.timedelta(hours=2)
    replays.build(cfg, ev, now)
    ref = _decoded(replays.peak_path(cfg, 13, "gradcam-overlay").read_bytes())
    shutil.rmtree(replays.replay_dir(cfg, 13))
    for a in (False, True):
        gradcam_path(cfg.gradcam_dir, peak, a).unlink(missing_ok=True)
        cam_path(cfg.gradcam_dir, peak, a).unlink(missing_ok=True)
    shutil.rmtree(cfg.render_dir)
    doc = replays.build(cfg, ev, now)                                           # the loop: can't recompute
    assert {"gradcam-heat", "gradcam-overlay"} <= set(doc["missing"]["peak"]) and not doc["complete"]
    assert "the peak frame lacks gradcam-heat, gradcam-overlay" in doc["note"]
    loc = FitsLocator(cfg)
    gc = replays.GradcamRecompute(cfg, Ensemble.load(cfg.weights, cfg.weights_sha256), loc)
    calls = []
    orig = gc.cam
    gc.cam = lambda s, r: calls.append(s) or orig(s, r)
    doc = replays.build(cfg, ev, now, loc, gc)
    assert doc["missing"]["peak"] == [] and doc["peak"]["gradcam_peak"] > 0
    assert calls.count(peak) == 1                                               # once for the frame and the peak
    got = _decoded(replays.peak_path(cfg, 13, "gradcam-overlay").read_bytes())
    assert np.abs(got - ref).mean() < 1.0


def test_one_off_adds_the_new_files_to_110_replays(cfg, monkeypatch):
    monkeypatch.setenv("FLARE_NRT_DATA_DIR", str(cfg.data_dir))
    for k in ("WEIGHTS", "WEIGHTS_SHA256", "NORM_STATS"):
        monkeypatch.setenv(f"FLARE_NRT_{k}", str(getattr(cfg, k.lower())))
    monkeypatch.setenv("FLARE_NRT_REPLAY_DAYS", "100000")
    now = dt.datetime.now(dt.timezone.utc)
    # event 21: its FITS are still on disk; event 22: they are gone
    ev = _event(cfg, T0, T0 + 4 * CADENCE, eid=21)
    start, end = replays.window(cfg, ev)
    _synthetic_frames(cfg, start, end, peak=T0 + CADENCE)
    replays.build(cfg, ev, now)
    old = _as_110(cfg, 21)
    marker = replays.frame_path(cfg, 21, start, "171")
    marker.write_bytes(b"kept from the 1.1.0 replay")
    t1 = T0 + dt.timedelta(days=2)
    ev2 = _event(cfg, t1, t1 + 2 * CADENCE, eid=22)
    s2, e2 = replays.window(cfg, ev2)
    _synthetic_frames(cfg, s2, e2, peak=t1 + CADENCE)
    replays.build(cfg, ev2, now)
    _as_110(cfg, 22)
    shutil.rmtree(cfg.staging_dir)
    shutil.rmtree(cfg.render_dir)

    # the loop leaves 1.1.0 replays alone; the one-off picks them up
    assert old["complete"] and not replays.current(old)
    assert not replays.pending(cfg, now)
    assert {e["event_id"] for e in replays.pending(cfg, now, rebuild_incomplete=True)} == {21, 22}
    # FITS of event 21 back (staging was wiped for event 22's sake)
    _synthetic_frames(cfg, start, end, peak=T0 + CADENCE)
    shutil.rmtree(cfg.render_dir)
    assert replays.main(["--no-gradcam"]) == 0

    doc = replays.read_replay(cfg, 21)
    assert doc["complete"] and replays.current(doc) and doc["model_version"] == "1.2.0"
    assert all(f["images"] == ["171", "gradcam-heat", "131"] for f in doc["frames"])
    assert marker.read_bytes() == b"kept from the 1.1.0 replay"                # reused, not re-rendered
    assert sorted(p.stem for p in (replays.replay_dir(cfg, 21) / "peak").iterdir()) == sorted(replays.PEAK_FILES)

    doc2 = replays.read_replay(cfg, 22)
    assert not doc2["complete"] and replays.current(doc2)
    assert len(doc2["missing"]["131"]) == doc2["n_slots"] and doc2["missing"]["171"] == []
    assert doc2["missing"]["peak"] == list(replays.PEAK_WAVES)                # Grad-CAM survives (400 d)
    assert all(f["images"] == ["171", "gradcam-heat"] for f in doc2["frames"])
    assert f"{doc2['n_slots']} of {doc2['n_slots']} slots have no 131 Å image" in doc2["note"]
    assert "the peak frame lacks 131, 171, 304, 1600" in doc2["note"]
    # run again: event 22 is rebuilt (still incomplete), event 21 is left alone
    assert {e["event_id"] for e in replays.pending(cfg, now, rebuild_incomplete=True)} == {22}


def test_retention_unchanged_with_the_new_files(cfg):
    now = dt.datetime(2027, 6, 1, tzinfo=dt.timezone.utc)
    for eid, age in ((1, 366), (2, 364)):
        t = now - dt.timedelta(days=age)
        ev = _event(cfg, t, t + 2 * CADENCE, eid=eid)
        start, end = replays.window(cfg, ev)
        _synthetic_frames(cfg, start, end, peak=t + CADENCE)
        doc = replays.build(cfg, ev, t + dt.timedelta(hours=2))
        assert doc["complete"] and (replays.replay_dir(cfg, eid) / "peak").is_dir()
    shutil.rmtree(cfg.staging_dir)                                  # FITS rules are not what is tested here
    shutil.rmtree(cfg.render_dir)
    Retention(cfg).run(now=now)
    assert not replays.replay_dir(cfg, 1).exists()                  # whole directory, peak/ and 131 frames too
    d2 = replays.replay_dir(cfg, 2)
    assert len(list((d2 / "frames").rglob("131.webp"))) == replays.read_replay(cfg, 2)["n_slots"]
    assert len(list((d2 / "peak").iterdir())) == len(replays.PEAK_FILES)
