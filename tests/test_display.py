"""1.1.0 display: Grad-CAM heat style, 1600 Å scaling, PNG/WebP and
512/1024, pre-rendered render/ frames and /api/loop, event replays and
their retention. Detection is untouched (tests/test_snapshot.py)."""

import datetime as dt
import json
import shutil
from io import BytesIO

import numpy as np
import pytest
from conftest import T0, write_fits, write_slot
from fastapi.testclient import TestClient
from PIL import Image

from flare_nrt import render, replays
from flare_nrt.artifacts import cam_path, find_cam, gradcam_path, tensor_path
from flare_nrt.display import render_path
from flare_nrt.grid import CADENCE, CHANNELS, file_name, iso, parse_iso
from flare_nrt.retention import Retention
from flare_nrt.server import create_app
from flare_nrt.store import ResultStore, append_event, read_events, write_json_atomic


def _img(content: bytes) -> Image.Image:
    return Image.open(BytesIO(content))


def _live(cfg, n=10, threshold=0.0, t0=None):
    """Slots ending now through the live Processor (renders on), like the loop."""
    from flare_nrt.loop import Processor
    from flare_nrt.model import Ensemble
    from flare_nrt.source import ReplaySource

    if t0 is None:
        now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
        t0 = now - dt.timedelta(minutes=now.minute % 2) - (n - 1) * CADENCE
    slots = [t0 + i * CADENCE for i in range(n)]
    for i, s in enumerate(slots):
        write_slot(cfg.staging_dir, s, flare=400.0 if i >= n - 3 else 0.0)
    cfg.threshold = threshold
    proc = Processor(cfg, Ensemble.load(cfg.weights, cfg.weights_sha256), ReplaySource(cfg.staging_dir), live=True)
    rows = [proc.process(s, s + dt.timedelta(minutes=1)) for s in slots]
    write_json_atomic(cfg.state_dir / "heartbeat.json", {"heartbeat": iso(slots[-1] + CADENCE)})
    return proc, slots, rows


# -- Grad-CAM heat style ------------------------------------------------------

LIMB = [255.5, 255.5, 200.0]


def _blob(peak=0.4):
    yy, xx = np.mgrid[:512, :512]
    return (peak * np.exp(-((xx - 330) ** 2 + (yy - 300) ** 2) / (2 * 25**2))).astype(np.float32)


def test_heat_is_transparent_below_the_floor_and_has_the_limb():
    cam = _blob()
    rgba = render.heat_rgba(cam, LIMB)[::-1]               # display rows (north up)
    assert rgba.shape == (512, 512, 4)
    c = (cam / cam.max())[::-1]
    yy, xx = np.mgrid[:512, :512]
    dist = np.abs(np.hypot(xx - 255.5, (511 - yy) - 255.5) - 200.0)
    off = (c < render.HEAT_FLOOR) & (dist > 3)
    assert off.sum() > 100_000 and (rgba[..., 3][off] == 0).all()      # transparent, no 131 Å underlay
    on_limb = (dist < 0.3) & (c < render.HEAT_FLOOR)
    assert on_limb.sum() > 500 and (rgba[..., 3][on_limb] > 150).all()  # the limb circle is drawn
    assert np.abs(rgba[..., :3][on_limb].astype(int) - 200).max() <= 2
    peak = np.unravel_index(np.argmax(c), c.shape)
    assert rgba[peak][3] == 255 and tuple(rgba[peak][:3]) == tuple(render.heat_lut()[255])
    no_limb = render.heat_rgba(cam, None)
    assert (no_limb[..., 3][(cam / cam.max()) < render.HEAT_FLOOR] == 0).all()


def test_heat_is_normalised_to_its_own_maximum():
    a = render.heat_rgba(_blob(0.4), LIMB)
    b = render.heat_rgba(_blob(0.9), LIMB)
    d = np.abs(a.astype(int) - b.astype(int))
    assert d.max() <= 4 and d.mean() < 0.05                            # a weak map looks as strong (± rounding) …
    u8 = np.round(_blob(0.4) / 0.4 * 255).astype(np.uint8)
    stored = render.heat_rgba(u8, LIMB).astype(int)                     # … and so does the stored uint8 form:
    opaque = (stored[..., 3] == 255) & (a[..., 3] == 255)               # same colours; the 8-bit steps only
    assert np.abs(stored - a.astype(int))[opaque].max() <= 6             # (one LUT step) and shift the ramp edge
    assert np.abs(stored[..., 3] - a[..., 3].astype(int)).max() <= 40


def test_inferno_stops_and_lut():
    lut = render.heat_lut()
    assert lut.shape == (256, 3) and tuple(lut[0]) == (0, 0, 4) and tuple(lut[255]) == (252, 255, 164)
    lum = lut @ np.array([0.2126, 0.7152, 0.0722])
    assert (np.diff(lum) >= -0.5).all()                               # monotonic lightness: perceptual


def test_limb_from_header():
    h = {"NAXIS1": 1024, "CRPIX1": 512.5, "CRPIX2": 512.5, "CDELT1": 2.4, "RSUN_OBS": 960.0}
    assert render.limb_from_header(h) == [255.5, 255.5, 200.0]
    assert render.limb_from_header({}) is None


def test_gradcam_heat_endpoint_and_headers(cfg):
    _, slots, rows = _live(cfg, threshold=0.5)
    c = TestClient(create_app(cfg))
    ok = [r for r in rows if r["status"] == "ok"]
    for which, slot in (("latest", slots[-1]), (ok[0]["slot"], parse_iso(ok[0]["slot"]))):
        r = c.get(f"/api/gradcam/{which}.png?style=heat")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        im = _img(r.content)
        assert im.mode == "RGBA" and im.size == (512, 512)
        rec = find_cam(cfg.gradcam_dir, slot)
        row = next(x for x in rows if x["slot"] == iso(slot))
        assert r.headers["X-Gradcam-Slot"] == iso(slot)
        assert abs(float(r.headers["X-Gradcam-Peak"]) - rec["peak"]) < 1e-5 and 0 < rec["peak"] <= 1
        assert abs(float(r.headers["X-Gradcam-Prob"]) - row["probability"]) < 1e-6
        a = np.asarray(im)[..., 3]
        assert (a == 0).any() and (a == 255).any()
        w = c.get(f"/api/gradcam/{which}.png?style=heat&format=webp")
        assert w.headers["content-type"] == "image/webp" and w.content[:4] == b"RIFF" and w.content[8:12] == b"WEBP"
        assert _img(w.content).mode == "RGBA"
    # the default is still the 1.0.0 overlay: the stored PNG, no heat headers
    r = c.get(f"/api/gradcam/{ok[0]['slot']}.png")
    assert r.content == gradcam_path(cfg.gradcam_dir, parse_iso(ok[0]["slot"]), False).read_bytes()
    assert "X-Gradcam-Peak" not in r.headers and _img(r.content).mode == "RGB"
    w = c.get(f"/api/gradcam/{ok[0]['slot']}.png?format=webp")
    assert w.headers["content-type"] == "image/webp" and _img(w.content).size == (512, 512)
    assert c.get("/api/gradcam/latest.png?style=hot").status_code == 422
    assert c.get("/api/gradcam/latest.png?format=gif").status_code == 422


def test_heat_404_without_a_stored_map(cfg):
    """A 1.0.0 Grad-CAM (PNG only, no map) keeps its overlay; heat says 404."""
    _, _, rows = _live(cfg, threshold=0.5)
    s = parse_iso([r for r in rows if r["status"] == "ok"][0]["slot"])
    for alert in (False, True):
        cam_path(cfg.gradcam_dir, s, alert).unlink(missing_ok=True)
    c = TestClient(create_app(cfg))
    assert c.get(f"/api/gradcam/{iso(s)}.png").status_code == 200
    assert c.get(f"/api/gradcam/{iso(s)}.png?style=heat").status_code == 404
    assert c.get("/api/gradcam/2026-01-01T00:00:00Z.png?style=heat").status_code == 404


# -- 1600 Å scaling -------------------------------------------------------------

def _uv_disk(n=512):
    """A 1600 Å-like frame: limb-darkened disk ~10–16 DN/s, plage ~30, sky ~1."""
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - n / 2 + 0.5, yy - n / 2 + 0.5) / (0.4 * n)
    disk = np.where(r < 1, 16.0 * (0.6 + 0.4 * np.sqrt(np.clip(1 - r**2, 0, 1))), 1.0)
    plage = 14.0 * np.exp(-((xx - 300) ** 2 + (yy - 220) ** 2) / (2 * 12**2))
    return disk + plage, r


def test_1600_scaling_bounds():
    v, r = _uv_disk()
    x = render.scale(v, 1600)
    centre, limb, plage, sky = (np.median(x[r < 0.2]), np.median(x[(r > 0.9) & (r < 0.98)]),
                                x[220, 300], np.median(x[r > 1.1]))
    assert 0.3 <= centre <= 0.65                 # mid-range disk, not saturated
    assert centre - limb >= 0.12                 # limb darkening visible
    assert plage >= 0.85 and plage - centre >= 0.3
    assert sky <= 0.05
    # 1.0.0's asinh stretch (still used for 131/171/304) washed this disk out
    old = render.scale(v, 171)
    assert np.median(old[r < 0.98]) > 0.75 and np.median(old[r < 0.2]) - np.median(old[(r > 0.9) & (r < 0.98)]) < 0.1


def test_other_channels_keep_their_scaling():
    assert {w: render.SCALING[w] for w in (131, 171, 304)} == {w: ("asinh", 99.7, 0.05) for w in (131, 171, 304)}
    assert render.SCALING[1600] == ("linear", 99.9, None)


# -- formats and sizes ------------------------------------------------------------

def test_image_formats_sizes_and_cache_headers(cfg):
    for i in range(7):                                   # t−8…−12 for the last one
        write_slot(cfg.staging_dir, T0 + i * CADENCE, size=1024)
    slot = T0 + 6 * CADENCE
    c = TestClient(create_app(cfg))
    png = c.get(f"/api/image/171.png?slot={iso(slot)}")
    webp = c.get(f"/api/image/171.png?slot={iso(slot)}&format=webp")
    big = c.get(f"/api/image/171.png?slot={iso(slot)}&size=1024")
    bigw = c.get(f"/api/image/171.png?slot={iso(slot)}&size=1024&format=webp")
    assert png.headers["content-type"] == "image/png" and png.content[:4] == b"\x89PNG"
    assert webp.headers["content-type"] == "image/webp" and webp.content[8:12] == b"WEBP"
    assert _img(png.content).size == _img(webp.content).size == (512, 512)
    assert _img(big.content).size == _img(bigw.content).size == (1024, 1024)
    assert bigw.headers["content-type"] == "image/webp"
    for r in (png, webp, big, bigw):
        assert r.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert r.headers["X-Observation-Time"] == iso(slot)
    # diff and Grad-CAM stay 512
    d = c.get(f"/api/image/171.png?slot={iso(slot)}&kind=diff&format=webp")
    assert d.status_code == 200 and _img(d.content).size == (512, 512)
    assert d.headers["cache-control"] == "max-age=3600"      # no result row yet: baselines may still change
    assert c.get(f"/api/image/171.png?slot={iso(slot)}&kind=diff&size=1024").status_code == 422
    assert c.get(f"/api/image/171.png?slot={iso(slot)}&size=256").status_code == 422
    assert c.get(f"/api/image/171.png?slot={iso(slot)}&format=jpg").status_code == 422
    # latest (no slot): short cache, as in 1.0.0
    assert c.get("/api/image/171.png?format=webp").headers["cache-control"] == "max-age=60"
    k = f"{slot:%Y%m%dT%H%M}_171"
    assert {p.name for p in cfg.cache_dir.iterdir()} == {f"{k}_img.png", f"{k}_img.webp", f"{k}_img_1024.png",
                                                         f"{k}_img_1024.webp", f"{k}_diff.webp"}


def test_webp_is_smaller_than_png(cfg):
    write_slot(cfg.staging_dir, T0, size=1024)
    c = TestClient(create_app(cfg))
    png = c.get(f"/api/image/171.png?slot={iso(T0)}").content
    webp = c.get(f"/api/image/171.png?slot={iso(T0)}&format=webp").content
    assert len(webp) < len(png)


# -- render/ (pre-rendered per result) -------------------------------------------

def test_loop_writes_render_frames(cfg):
    _, slots, rows = _live(cfg, threshold=0.5)
    for r in rows:
        s = parse_iso(r["slot"])
        have = {ch for ch in ("171", "171-diff", "gradcam-heat") if render_path(cfg.render_dir, s, ch).exists()}
        if r["status"] == "ok":
            assert have == {"171", "171-diff", "gradcam-heat"}
            im = _img(render_path(cfg.render_dir, s, "171").read_bytes())
            assert im.format == "WEBP" and im.size == (512, 512)
            assert _img(render_path(cfg.render_dir, s, "gradcam-heat").read_bytes()).mode == "RGBA"
        else:
            assert have == set()                          # no result, no render


def test_backlog_slot_without_gradcam_has_no_heat_frame(cfg):
    from flare_nrt.loop import Processor
    from flare_nrt.model import Ensemble
    from flare_nrt.source import ReplaySource

    slots = [T0 + i * CADENCE for i in range(8)]
    for s in slots:
        write_slot(cfg.staging_dir, s)
    cfg.threshold = 1.1                                    # no alert
    proc = Processor(cfg, Ensemble.load(cfg.weights, cfg.weights_sha256), ReplaySource(cfg.staging_dir), live=True)
    proc.gradcam_wanted = False                            # a backlog slot: no Grad-CAM
    r = proc.process(slots[7], slots[7] + dt.timedelta(minutes=1))
    assert r["status"] == "ok"
    assert render_path(cfg.render_dir, slots[7], "171").exists()
    assert not render_path(cfg.render_dir, slots[7], "gradcam-heat").exists()


def test_replay_mode_writes_no_render(cfg):
    from flare_nrt.loop import run

    for i in range(8):
        write_slot(cfg.staging_dir, T0 + i * CADENCE)
    run(cfg, replay_dir=cfg.staging_dir)
    assert not any(cfg.render_dir.rglob("*.webp"))


def _touch(p, data=b"RIFF0000WEBP"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_render_is_pruned_with_the_fits(cfg):
    now = dt.datetime(2026, 10, 28, 12, 0, tzinfo=dt.timezone.utc)
    keep = [_touch(render_path(cfg.render_dir, now - dt.timedelta(days=d), "171")) for d in (0, 3.9)]
    drop = [_touch(render_path(cfg.render_dir, now - dt.timedelta(days=d), "171")) for d in (4.1, 6)]
    keep.append(_touch(cfg.staging_dir / f"{now:%Y%m%d}" / file_name(now, 171)))
    Retention(cfg).run(now=now)
    assert all(p.exists() for p in keep) and not any(p.exists() for p in drop)
    cfg.staging_hours = 1                                   # small disk: render follows the FITS hours too
    Retention(cfg).run(now=now + dt.timedelta(hours=3))
    assert not any(cfg.render_dir.rglob("*.webp"))


def test_gradcam_maps_follow_their_png(cfg):
    now = dt.datetime(2026, 10, 28, 12, 0, tzinfo=dt.timezone.utc)
    d = now - dt.timedelta(days=3)
    gone = [_touch(gradcam_path(cfg.gradcam_dir, d, False)), _touch(cam_path(cfg.gradcam_dir, d, False))]
    kept = [_touch(gradcam_path(cfg.gradcam_dir, d, True)), _touch(cam_path(cfg.gradcam_dir, d, True))]
    Retention(cfg).run(now=now)
    assert all(p.exists() for p in kept) and not any(p.exists() for p in gone)


# -- /api/loop ---------------------------------------------------------------------

def test_loop_json_order_and_window(cfg):
    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    now -= dt.timedelta(minutes=now.minute % 2)
    ages = [7 * 60, 5 * 60, 30, 2, 50]                      # minutes, out of order on purpose
    store = ResultStore(cfg.results_dir)
    for m in sorted(ages, reverse=True):
        s = now - dt.timedelta(minutes=m)
        _touch(render_path(cfg.render_dir, s, "171"))
        store.append({"slot": iso(s), "status": "ok", "probability": m / 1000, "state": "quiet",
                      "inputs": {"171": iso(s)[:-1] + ".350Z"}})
    _touch(render_path(cfg.render_dir, now - dt.timedelta(minutes=4), "gradcam-heat"))
    c = TestClient(create_app(cfg))
    j = c.get("/api/loop").json()
    assert j["channel"] == "171" and j["hours"] == 6
    exp = [iso(now - dt.timedelta(minutes=m)) for m in (300, 50, 30, 2)]
    assert [f["slot"] for f in j["frames"]] == exp                          # time order, 7 h ago left out
    f = j["frames"][-1]
    assert f["url"] == f"/api/render/{exp[-1]}/171.webp" and f["probability"] == 0.002 and f["state"] == "quiet"
    assert f["t_obs"].endswith(".350Z")
    r = c.get(f["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    assert r.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert [x["slot"] for x in c.get("/api/loop?hours=1").json()["frames"]] == exp[1:]
    assert len(c.get("/api/loop?hours=24").json()["frames"]) == 5
    h = c.get("/api/loop?channel=gradcam-heat").json()["frames"]
    assert len(h) == 1 and h[0]["probability"] is None                       # a frame without a result row
    for bad in ("hours=0", "hours=25", "channel=304"):
        assert c.get("/api/loop?" + bad).status_code == 422
    assert c.get(f"/api/render/{exp[0]}/171-diff.webp").status_code == 404
    assert c.get(f"/api/render/{exp[0]}/304.webp").status_code == 404
    assert c.get("/api/render/yesterday/171.webp").status_code == 422


def test_loop_from_the_live_processor(cfg):
    _, slots, rows = _live(cfg, threshold=0.5)
    c = TestClient(create_app(cfg))
    ok = [r for r in rows if r["status"] == "ok"]
    for ch in ("171", "171-diff", "gradcam-heat"):
        fr = c.get(f"/api/loop?channel={ch}").json()["frames"]
        assert [f["slot"] for f in fr] == [r["slot"] for r in ok]
        assert [f["probability"] for f in fr] == [r["probability"] for r in ok]
        assert all(c.get(f["url"]).status_code == 200 for f in fr)


# -- event replays -------------------------------------------------------------------

def _event(cfg, t_start, t_end, eid=1, p=0.9):
    from flare_nrt.detector import Event

    append_event(cfg.events_path, Event(eid, iso(t_start), iso(t_end), iso(t_start + CADENCE), p, 3,
                                        {"centroid_arcsec": [100.0, -200.0]}, True))
    return read_events(cfg.events_path)[-1]


def _peak_inputs(cfg, slot):
    """What the loop leaves for an alerting slot: its 4 FITS, its Grad-CAM
    overlay (``_alert.png``) and map (``_alert.cam.npz``)."""
    from flare_nrt.artifacts import _write, cam_npz, gradcam_png

    for w in CHANNELS:
        write_fits(cfg.staging_dir / f"{slot:%Y%m%d}" / file_name(slot, w), w, slot, size=64)
    _write(gradcam_path(cfg.gradcam_dir, slot, True), gradcam_png(_blob(), np.full((512, 512), 0.3), 1.0))
    _write(cam_path(cfg.gradcam_dir, slot, True), cam_npz(_blob(), 0.9, LIMB))


def _synthetic_frames(cfg, start, end, skip=(), peak=None):
    """Rows and render files for every slot in [start, end], and a small
    131 Å FITS (1.2.0), except ``skip``; the peak set's inputs at ``peak``."""
    store = ResultStore(cfg.results_dir)
    s = start
    while s <= end:
        store.append({"slot": iso(s), "status": "ok", "probability": 0.5, "state": "quiet",
                      "inputs": {"171": iso(s)}})
        if s not in skip:
            for ch in ("171", "gradcam-heat"):
                _touch(render_path(cfg.render_dir, s, ch), f"{ch}{iso(s)}".encode())
            write_fits(cfg.staging_dir / f"{s:%Y%m%d}" / file_name(s, 131), 131, s, size=64)
        s += CADENCE
    if peak is not None:
        _peak_inputs(cfg, peak)


def test_replay_waits_until_closed_for_60_min(cfg):
    t_start, t_end = T0, T0 + 10 * CADENCE
    ev = _event(cfg, t_start, t_end)
    closed = t_end + dt.timedelta(minutes=cfg.merge_gap_min)
    assert not replays.pending(cfg, closed + dt.timedelta(minutes=59))
    assert [e["event_id"] for e in replays.pending(cfg, closed + dt.timedelta(minutes=60))] == [1]
    # the loop also needs to have processed up to t_end + 60 min
    assert not replays.pending(cfg, closed + dt.timedelta(minutes=60), last_slot=t_end + dt.timedelta(minutes=58))
    assert replays.pending(cfg, closed + dt.timedelta(minutes=60), last_slot=t_end + dt.timedelta(minutes=60))
    assert replays.window(cfg, ev) == (t_start - dt.timedelta(minutes=30), t_end + dt.timedelta(minutes=60))


def test_replay_build_complete(cfg):
    t_start, t_end = T0, T0 + 10 * CADENCE
    ev = _event(cfg, t_start, t_end, eid=7)
    start, end = replays.window(cfg, ev)
    _synthetic_frames(cfg, start, end, peak=t_start + CADENCE)
    now = end + dt.timedelta(hours=1)
    builder = replays.ReplayBuilder(cfg)
    assert builder.maybe_run(now, now) == [7]
    assert builder.maybe_run(now, now) == []                    # at most every 5 min …
    builder._last = 0
    assert builder.maybe_run(now, now) == []                    # … and only once
    doc = json.loads((cfg.replays_dir / "7" / "replay.json").read_text())
    assert doc["schema"] == "flare-nrt-replay/1" and doc["complete"] and doc["note"] is None
    assert doc["event"] == {"event_id": 7, "t_start": iso(t_start), "t_peak": iso(t_start + CADENCE),
                            "t_end": iso(t_end), "p_max": 0.9, "n_alert_frames": 3,
                            "location": {"centroid_arcsec": [100.0, -200.0]}}
    n = (end - start) // CADENCE + 1
    assert doc["n_slots"] == len(doc["frames"]) == n == 56
    assert [f["slot"] for f in doc["frames"]] == [iso(start + i * CADENCE) for i in range(n)]
    f = doc["frames"][0]
    assert {"slot", "t_obs", "probability", "state"} <= set(f) and f["images"] == ["171", "gradcam-heat", "131"]
    assert doc["peak"]["images"] == list(replays.PEAK_FILES) and doc["missing"]["peak"] == []
    assert (cfg.replays_dir / "7" / "frames" / f"{start:%Y%m%dT%H%M}" / "171.webp").read_bytes() == \
        render_path(cfg.render_dir, start, "171").read_bytes()
    assert not [p for p in cfg.replays_dir.iterdir() if p.name.startswith(".")]   # no temp left

    c = TestClient(create_app(cfg))
    assert c.get("/api/events/7/replay").json() == doc
    r = c.get(f"/api/events/7/frames/{iso(start)}/gradcam-heat.webp")
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    assert r.headers["cache-control"] == "public, max-age=31536000, immutable"
    ev_api = c.get(f"/api/events?start={iso(T0 - dt.timedelta(days=1))}&end={iso(T0 + dt.timedelta(days=1))}")
    assert ev_api.json()["events"][0]["replay"] is True
    # 404s
    assert c.get("/api/events/8/replay").status_code == 404
    assert c.get(f"/api/events/8/frames/{iso(start)}/171.webp").status_code == 404
    assert c.get(f"/api/events/7/frames/{iso(start - CADENCE)}/171.webp").status_code == 404
    assert c.get(f"/api/events/7/frames/{iso(start)}/171-diff.webp").status_code == 404
    assert c.get("/api/events/7/frames/notatime/171.webp").status_code == 422
    assert c.get("/api/events/x/replay").status_code == 422


def test_replay_partial_when_frames_are_gone(cfg):
    t_start, t_end = T0, T0 + 4 * CADENCE
    ev = _event(cfg, t_start, t_end, eid=3)
    start, end = replays.window(cfg, ev)
    gone = {start + i * CADENCE for i in range(10, 20)}           # service down for 20 min
    _synthetic_frames(cfg, start, end, skip=gone)
    doc = replays.build(cfg, ev, end + dt.timedelta(hours=2))
    assert not doc["complete"] and "10 of 50 slots have no 171" in doc["note"]
    assert sorted(doc["missing"]["171"]) == sorted(iso(s) for s in gone) == sorted(doc["missing"]["gradcam-heat"])
    assert len(doc["frames"]) == 40 and not set(f["slot"] for f in doc["frames"]) & {iso(s) for s in gone}
    # an event whose frames are all gone still gets a replay that says so
    ev2 = _event(cfg, T0 + dt.timedelta(days=1), T0 + dt.timedelta(days=1, minutes=4), eid=4)
    doc2 = replays.build(cfg, ev2, T0 + dt.timedelta(days=2))
    assert doc2["frames"] == [] and not doc2["complete"] and doc2["missing"]["171"]
    assert doc2["missing"]["gradcam-heat"] == []                # no result rows: nothing to explain
    c = TestClient(create_app(cfg))
    win = f"start={iso(T0 - dt.timedelta(days=1))}&end={iso(T0 + dt.timedelta(days=2))}"
    assert {e["event_id"]: e["replay"] for e in c.get("/api/events?" + win).json()["events"]} == {3: True, 4: False}
    assert c.get("/api/events/4/replay").json()["frames"] == []      # its JSON still says why


def test_replay_from_fits_and_stored_maps(cfg):
    """No render/ (e.g. frames from before the upgrade): 171 Å from the FITS,
    the heat map from the stored Grad-CAM map; the one-off recomputes maps
    that are gone, from the stored input or the FITS, identically."""
    _, slots, rows = _live(cfg, n=12, threshold=0.5)
    ok = [parse_iso(r["slot"]) for r in rows if r["status"] == "ok"]
    ev = {"event_id": 9, "t_start": iso(slots[-1]), "t_end": iso(slots[-1]), "t_pmax": iso(slots[-1]),
          "p_max": 0.9, "n_alert_frames": 1, "location": None}
    shutil.rmtree(cfg.render_dir)
    now = slots[-1] + dt.timedelta(hours=2)
    doc = replays.build(cfg, ev, now)
    by = {f["slot"]: f for f in doc["frames"]}
    assert all(by[iso(s)]["images"] == ["171", "gradcam-heat", "131"] for s in ok)
    assert all(by[iso(s)]["images"] == ["171", "131"] for s in slots if s not in ok)   # no result: no Grad-CAM
    assert doc["missing"]["gradcam-heat"] == []
    # Grad-CAM maps gone: --no-gradcam leaves them missing …
    peaks = {s: find_cam(cfg.gradcam_dir, s)["peak"] for s in ok}
    for p in cfg.gradcam_dir.rglob("*.cam.npz"):
        p.unlink()
    shutil.rmtree(replays.replay_dir(cfg, 9))                         # a first build, nothing to reuse
    doc = replays.build(cfg, ev, now)
    assert sorted(doc["missing"]["gradcam-heat"]) == [iso(s) for s in ok] and not doc["complete"]
    # … the one-off recomputes them: from the stored model input, then from the FITS alone
    from flare_nrt.display import FitsLocator
    from flare_nrt.model import Ensemble

    loc = FitsLocator(cfg)
    gc = replays.GradcamRecompute(cfg, Ensemble.load(cfg.weights, cfg.weights_sha256), loc)
    rows_by = {r["slot"]: r for r in rows}
    for s in ok[:2]:
        rec = gc.cam(s, rows_by[iso(s)])
        assert abs(rec["peak"] - peaks[s]) < 1e-5 and rec["limb"] is not None
    tensor_path(cfg.tensors_dir, ok[0]).unlink()
    assert abs(gc.cam(ok[0], rows_by[iso(ok[0])])["peak"] - peaks[ok[0]]) < 1e-3
    doc = replays.build(cfg, ev, now, loc, gc)
    assert doc["missing"]["gradcam-heat"] == []

    class Boom:
        def cam(self, slot, row):
            raise AssertionError("recomputed a frame the earlier replay already had")

    for p in cfg.tensors_dir.rglob("*.npz"):                          # a rebuild reuses its own frames
        p.unlink()
    assert replays.build(cfg, ev, now, loc, Boom())["missing"]["gradcam-heat"] == []
    assert not set(doc["missing"]["171"]) & {iso(s) for s in slots}    # only slots before the staged ones


def test_replays_command_rebuilds_incomplete(cfg, monkeypatch):
    t_start, t_end = T0, T0 + 4 * CADENCE
    ev = _event(cfg, t_start, t_end, eid=5)
    start, end = replays.window(cfg, ev)
    _synthetic_frames(cfg, start, end, skip={start}, peak=t_start + CADENCE)
    now = dt.datetime.now(dt.timezone.utc)
    monkeypatch.setenv("FLARE_NRT_DATA_DIR", str(cfg.data_dir))
    for k in ("WEIGHTS", "WEIGHTS_SHA256", "NORM_STATS"):
        monkeypatch.setenv(f"FLARE_NRT_{k}", str(getattr(cfg, k.lower())))
    monkeypatch.setenv("FLARE_NRT_REPLAY_DAYS", "100000")
    replays.build(cfg, ev, now)
    assert not replays.read_replay(cfg, 5)["complete"]
    assert not replays.pending(cfg, now)                                      # the loop leaves it
    assert replays.pending(cfg, now, rebuild_incomplete=True)                 # the one-off doesn't
    _touch(render_path(cfg.render_dir, start, "171"))                         # e.g. recomputed
    _touch(render_path(cfg.render_dir, start, "gradcam-heat"))
    write_fits(cfg.staging_dir / f"{start:%Y%m%d}" / file_name(start, 131), 131, start, size=64)
    assert replays.main([]) == 0
    assert replays.read_replay(cfg, 5)["complete"]
    assert replays.main(["--no-gradcam"]) == 0                                # nothing left to do


def test_replay_of_a_reused_event_id_is_rebuilt(cfg):
    ev = _event(cfg, T0, T0 + CADENCE, eid=2)
    replays.build(cfg, ev, T0 + dt.timedelta(hours=3))
    other = dict(ev, t_start=iso(T0 + dt.timedelta(days=1)), t_end=iso(T0 + dt.timedelta(days=1)))
    cfg.events_path.unlink()
    from flare_nrt.detector import Event
    append_event(cfg.events_path, Event(2, other["t_start"], other["t_end"], other["t_pmax"], 0.9, 1, None, True))
    assert replays.pending(cfg, T0 + dt.timedelta(days=2))


def test_replay_retention(cfg):
    now = dt.datetime(2027, 6, 1, tzinfo=dt.timezone.utc)
    for eid, age in ((1, 366), (2, 364)):
        t = now - dt.timedelta(days=age)
        replays.build(cfg, {"event_id": eid, "t_start": iso(t), "t_end": iso(t), "t_pmax": iso(t),
                            "p_max": 0.9, "n_alert_frames": 1, "location": None}, t)
    stale = cfg.replays_dir / ".3.tmp-99"
    stale.mkdir()
    import os
    os.utime(stale, (0, 0))
    Retention(cfg).run(now=now)
    assert not (cfg.replays_dir / "1").exists() and (cfg.replays_dir / "2" / "replay.json").exists()
    assert not stale.exists()
    # pruned replays are not rebuilt: events older than replay_days are out of scope
    from flare_nrt.detector import Event
    t = now - dt.timedelta(days=366)
    append_event(cfg.events_path, Event(1, iso(t), iso(t), iso(t), 0.9, 1, None, True))
    assert not replays.pending(cfg, now)


@pytest.mark.parametrize("wave", CHANNELS)
def test_cache_names_differ_only_for_1600(cfg, wave):
    """A 1.0.0 cache file of 1600 Å (washed out) is never served by 1.1.0."""
    write_fits(cfg.staging_dir / f"{T0:%Y%m%d}" / file_name(T0, wave), wave, T0)
    c = TestClient(create_app(cfg))
    c.get(f"/api/image/{wave}.png?slot={iso(T0)}")
    name = next(cfg.cache_dir.glob("*.png")).name
    assert name == f"{T0:%Y%m%dT%H%M}_{wave}_img{'-lin' if wave == 1600 else ''}.png"
