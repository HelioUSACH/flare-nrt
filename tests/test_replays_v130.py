"""1.3.0: a Grad-CAM map on every replay frame with a result (``cam_peak``,
``alert``), a location for every alert frame and every event (the result's
own, else computed from the 131 Å FITS; an event's is filled once, never
changed), the image geometry, ``--fits-root``, the loop's location when its
131 Å rates are not in memory, and the robust fallback of ``localize``.
Display only: detection is untouched (tests/test_snapshot.py)."""

import datetime as dt
import json
import shutil

import numpy as np
import pytest
from conftest import T0, write_fits, write_slot
from fastapi.testclient import TestClient
from test_display import _event, _live, _synthetic_frames

from flare_nrt import render, replays
from flare_nrt.artifacts import cam_path, find_cam, gradcam_path
from flare_nrt.detector import Event
from flare_nrt.display import FitsLocator, render_path
from flare_nrt.grid import CADENCE, file_name, iso, parse_iso
from flare_nrt.localize import METHOD, METHOD_ROBUST, localize, rate_map
from flare_nrt.model import Ensemble
from flare_nrt.retention import Retention
from flare_nrt.server import create_app
from flare_nrt.store import ResultStore, append_event, fill_event_location, read_events, write_json_atomic

LOC_KEYS = {"centroid_arcsec", "bbox_arcsec", "area_px", "area_arcsec2", "on_limb", "method", "confidence"}


def _ens(cfg):
    return Ensemble.load(cfg.weights, cfg.weights_sha256)


def _drop_maps(cfg, slot):
    """What a backlog slot leaves: a result, but no Grad-CAM and no heat frame."""
    for a in (False, True):
        gradcam_path(cfg.gradcam_dir, slot, a).unlink(missing_ok=True)
        cam_path(cfg.gradcam_dir, slot, a).unlink(missing_ok=True)
    render_path(cfg.render_dir, slot, "gradcam-heat").unlink(missing_ok=True)


def _strip_locations(cfg):
    """Rewrite the stored rows as event 1's were: alert frames without a location."""
    for p in cfg.results_dir.glob("*.jsonl"):
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        p.write_text("".join(json.dumps(dict(r, location=None)) + "\n" for r in rows))


def _flare_event(cfg, eid, days_ago=0):
    """Live slots through the Processor (threshold 0.5); the stand-in
    ensemble gives ≈ 0.53 on every frame, so the stored rows are rewritten
    as a real event's: only the 3 flare slots alert (with the loop's
    location), the others are quiet (p 0.3, no location)."""
    t0 = None
    if days_ago:
        now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
        t0 = now - dt.timedelta(days=days_ago, minutes=now.minute % 2)
    _, slots, rows = _live(cfg, n=12, threshold=0.5, t0=t0)
    flare = {iso(s) for s in slots[-3:]}
    out = []
    for r in rows:
        if r["status"] == "ok" and r["slot"] not in flare:
            r = dict(r, probability=0.3, alert=False, location=None)
        out.append(r)
    for p in cfg.results_dir.glob("*.jsonl"):
        p.unlink()
    store = ResultStore(cfg.results_dir)
    for r in out:
        store.append(r)
    alerts = [r for r in out if r.get("alert")]
    assert [r["slot"] for r in alerts] == sorted(flare) and all(r["location"] for r in alerts)
    top = max(alerts, key=lambda r: r["probability"])
    ev = {"event_id": eid, "t_start": alerts[0]["slot"], "t_end": alerts[-1]["slot"], "t_pmax": top["slot"],
          "p_max": top["probability"], "n_alert_frames": len(alerts), "location": None}
    return ev, slots, out, parse_iso(top["slot"])


def _env(cfg, monkeypatch):
    monkeypatch.setenv("FLARE_NRT_DATA_DIR", str(cfg.data_dir))
    for k in ("WEIGHTS", "WEIGHTS_SHA256", "NORM_STATS"):
        monkeypatch.setenv(f"FLARE_NRT_{k}", str(getattr(cfg, k.lower())))
    monkeypatch.setenv("FLARE_NRT_REPLAY_DAYS", "100000")


# -- Grad-CAM on every frame -----------------------------------------------------------

def test_every_frame_with_a_result_gets_a_map_also_in_the_loop(cfg):
    ev, slots, rows, peak = _flare_event(cfg, eid=31)
    ok = [parse_iso(r["slot"]) for r in rows if r["status"] == "ok"]
    quiet = [s for s in ok if not next(r for r in rows if r["slot"] == iso(s))["alert"]]
    assert len(quiet) >= 3
    peaks = {s: find_cam(cfg.gradcam_dir, s)["peak"] for s in ok}
    backlog = quiet[:3]                                   # processed without Grad-CAM (a backlog)
    for s in backlog:
        _drop_maps(cfg, s)
    append_event(cfg.events_path, Event(**ev, closed=True))
    now = slots[-1] + dt.timedelta(hours=2)
    # the loop's builder, with the model it already holds
    builder = replays.ReplayBuilder(cfg, _ens(cfg))
    assert builder.maybe_run(now, now) == [31]
    assert builder.gradcam.n == len(backlog)              # only the frames without a stored map
    doc = replays.read_replay(cfg, 31)
    by = {f["slot"]: f for f in doc["frames"]}
    assert doc["missing"]["gradcam-heat"] == []
    for s in ok:
        f = by[iso(s)]
        assert "gradcam-heat" in f["images"], iso(s)
        assert f["cam_peak"] == pytest.approx(peaks[s], abs=1e-5)
    for s in backlog:                                     # same rendering as a live heat frame
        assert replays.frame_path(cfg, 31, s, "gradcam-heat").stat().st_size > 0
    # without the model (1.2.0's loop) those frames stay missing and say so
    shutil.rmtree(replays.replay_dir(cfg, 31))
    for s in backlog:
        _drop_maps(cfg, s)
    doc = replays.build(cfg, ev, now)
    assert sorted(doc["missing"]["gradcam-heat"]) == sorted(iso(s) for s in backlog)
    assert f"{len(backlog)} results have no Grad-CAM" in doc["note"]
    assert all(by_s["cam_peak"] is None for by_s in doc["frames"] if by_s["slot"] in doc["missing"]["gradcam-heat"])


def test_frame_fields_alert_cam_peak_location_geometry(cfg):
    ev, slots, rows, peak = _flare_event(cfg, eid=32)
    doc = replays.build(cfg, ev, slots[-1] + dt.timedelta(hours=2))
    rows_by = {r["slot"]: r for r in rows}
    assert doc["threshold"] == cfg.threshold
    for f in doc["frames"]:
        row = rows_by.get(f["slot"])
        assert {"alert", "cam_peak", "location", "geometry"} <= set(f)
        if row is None or row["status"] != "ok":                    # no result: no map, nothing to say
            assert f["alert"] is None and f["cam_peak"] is None and f["location"] is None
            assert "gradcam-heat" not in f["images"]
            continue
        assert f["alert"] is (row["probability"] >= cfg.threshold) is row["alert"]
        assert 0 < f["cam_peak"] <= 1 and "gradcam-heat" in f["images"]
        if f["alert"]:
            assert f["location"] == row["location"] and set(f["location"]) == LOC_KEYS
        else:
            assert f["location"] is None
        g = f["geometry"]
        assert g["size_px"] == 512 and g["disk_radius_px"] > 0
    # the replay's geometry is the peak frame's; the peak has its location
    assert doc["geometry"] == doc["peak"]["geometry"] == next(f["geometry"] for f in doc["frames"]
                                                              if f["slot"] == iso(peak))
    assert doc["peak"]["location"] == rows_by[iso(peak)]["location"]
    assert doc["missing"]["location"] == [] and "location" not in doc["note"]


def test_geometry_places_the_location_on_the_north_up_image():
    """The flare of the synthetic FITS is at column 350, row 300 (FITS order,
    1024 px): on the 512-px north-up image, column 175.25, row 512 − 150.25."""
    hdr = {"NAXIS1": 1024, "CRPIX1": 512.5, "CRPIX2": 512.5, "CDELT1": 2.4, "CDELT2": 2.4,
           "RSUN_OBS": 960.0, "EXPTIME": 2.9}
    yy, xx = np.mgrid[:1024, :1024]
    sun = np.where((xx - 511.5) ** 2 + (yy - 511.5) ** 2 < 400**2, 100.0, 1.0)
    flare = sun + 3000.0 * np.exp(-((xx - 700) ** 2 + (yy - 600) ** 2) / (2 * 6**2))
    loc = localize(rate_map(flare.astype(np.float32), hdr) - rate_map(sun.astype(np.float32), hdr), hdr)
    g = render.image_geometry(hdr)
    assert g == {"size_px": 512, "disk_center_px": [256.0, 256.0], "disk_radius_px": 200.0,
                 "arcsec_per_px": [4.8, 4.8], "rsun_arcsec": 960.0}
    x, y = loc["centroid_arcsec"]
    col = g["disk_center_px"][0] + x / g["arcsec_per_px"][0]
    row = g["disk_center_px"][1] - y / g["arcsec_per_px"][1]
    # FITS pixel (700, 600), 0-based centre → 512 grid edge coords (700.5/2, 512 − 600.5/2)
    assert abs(col - 350.25) < 1.0 and abs(row - (512 - 300.25)) < 1.0
    # the limb circle of the heat map is the same circle
    cx, cy, r = render.limb_from_header(hdr)
    assert (cx + 0.5, 512 - (cy + 0.5), r) == (256.0, 256.0, 200.0)
    assert render.image_geometry({"NAXIS1": 1024}) is None


# -- every event gets a location ------------------------------------------------------

def test_event_without_location_gets_its_peak_frames_and_it_is_kept(cfg):
    ev, slots, rows, peak = _flare_event(cfg, eid=33)
    expected = {r["slot"]: r["location"] for r in rows if r.get("alert")}
    assert all(expected.values())
    _strip_locations(cfg)                                  # as event 1: no location anywhere
    append_event(cfg.events_path, Event(**ev, closed=True))
    assert read_events(cfg.events_path)[-1]["location"] is None
    now = slots[-1] + dt.timedelta(hours=2)
    doc = replays.build(cfg, ev, now)
    # computed from the FITS the same way the loop does: the same location
    assert doc["peak"]["location"] == expected[iso(peak)]
    assert doc["event"]["location"] == expected[iso(peak)]
    for f in doc["frames"]:
        assert f["location"] == (expected[f["slot"]] if f["alert"] else None)
    # and stored in the event record, served by /api/events
    assert read_events(cfg.events_path)[-1]["location"] == expected[iso(peak)]
    c = TestClient(create_app(cfg))
    win = f"start={iso(slots[0] - dt.timedelta(days=1))}&end={iso(slots[-1] + dt.timedelta(days=1))}"
    api = {e["event_id"]: e for e in c.get("/api/events?" + win).json()["events"]}
    assert api[33]["location"] == expected[iso(peak)] and api[33]["replay"] is True
    # never overwritten: not by a fill, not by a rebuild
    other = dict(expected[iso(peak)], centroid_arcsec=[1.0, 2.0])
    assert not fill_event_location(cfg.events_path, 33, ev["t_start"], other)
    replays.build(cfg, read_events(cfg.events_path)[-1], now)
    assert read_events(cfg.events_path)[-1]["location"] == expected[iso(peak)]


def test_an_existing_event_location_is_never_changed(cfg):
    ev, slots, rows, peak = _flare_event(cfg, eid=34)
    mine = {"centroid_arcsec": [-1.0, -1.0], "method": "kept"}
    append_event(cfg.events_path, Event(**dict(ev, location=mine), closed=True))
    before = cfg.events_path.read_bytes()
    doc = replays.build(cfg, read_events(cfg.events_path)[-1], slots[-1] + dt.timedelta(hours=2))
    assert doc["event"]["location"] == mine                         # the record's, unchanged
    assert doc["peak"]["location"] != mine and doc["peak"]["location"]   # the peak frame's own
    assert cfg.events_path.read_bytes() == before
    # a fill only touches the event with that id *and* t_start, and an empty one
    append_event(cfg.events_path, Event(35, iso(T0), iso(T0), iso(T0), 0.9, 1, None, True))
    assert not fill_event_location(cfg.events_path, 35, iso(T0 + CADENCE), mine)
    assert not fill_event_location(cfg.events_path, 35, iso(T0), None)
    assert fill_event_location(cfg.events_path, 35, iso(T0), mine)
    evs = {e["event_id"]: e for e in read_events(cfg.events_path)}
    assert evs[35]["location"] == mine and evs[34]["location"] == mine
    assert not (cfg.data_dir / "events.csv.tmp").exists()


def test_alert_frame_without_fits_says_why(cfg):
    ev, slots, rows, peak = _flare_event(cfg, eid=36)
    _strip_locations(cfg)
    shutil.rmtree(cfg.staging_dir)                         # FITS past their 4 days
    doc = replays.build(cfg, ev, slots[-1] + dt.timedelta(hours=2))
    alerts = [r["slot"] for r in rows if r.get("alert")]
    assert doc["missing"]["location"] == alerts and doc["event"]["location"] is None
    assert f"{len(alerts)} alert frames have no location" in doc["note"]
    assert read_events(cfg.events_path) == []              # nothing to fill, nothing written


# -- the live loop's location -----------------------------------------------------------

def test_loop_locates_an_alert_whose_131_rates_are_not_in_memory(cfg):
    """1.3.0: the rates missing from memory are read from the FITS."""
    from flare_nrt.loop import Processor
    from flare_nrt.source import ReplaySource

    _, slots, rows = _live(cfg, n=12, threshold=0.5)
    ref = {r["slot"]: r["location"] for r in rows if r.get("alert")}
    assert all(ref[iso(s)] for s in slots[-3:])                    # the flare slots are located
    # the same slots again, into another data directory, with the rates dropped before each alert
    cfg2 = cfg
    cfg2.data_dir = cfg.data_dir.parent / "var2"
    cfg2.ensure_dirs()
    for s in slots:
        write_slot(cfg2.staging_dir, s, flare=400.0 if s >= slots[-3] else 0.0)
    proc = Processor(cfg2, _ens(cfg2), ReplaySource(cfg2.staging_dir), live=True)
    proc.render = False
    got = {}
    for s in slots:
        if iso(s) in ref:
            proc.current(s)                                # the current frame is in memory …
            proc.rate131.clear()                           # … its rates and its baselines' are not
        r = proc.process(s, s + dt.timedelta(minutes=1))
        if r.get("alert"):
            got[r["slot"]] = r["location"]
    assert got == ref


def test_loop_logs_an_alert_it_cannot_locate(cfg, caplog):
    from flare_nrt.loop import Processor
    from flare_nrt.source import ReplaySource

    proc = Processor(cfg, _ens(cfg), ReplaySource(cfg.staging_dir), live=False)
    assert proc._locate(T0, [T0 - dt.timedelta(minutes=8)]) is None
    assert "131 Å FITS missing for" in caplog.text


def _disk(n=512):
    yy, xx = np.mgrid[:n, :n]
    return ((xx - 255.5) ** 2 + (yy - 255.5) ** 2) < 200**2, xx, yy


def test_localize_after_the_131_maximum_falls_back_to_a_robust_sigma():
    """The M4.0 of 2026-07-07 at 14:30: the baselines hold the flare, so the
    difference is −1000s at the kernel and +20 a few arcmin away; mean + 5σ
    was above every positive value and the frame had no location."""
    rng = np.random.default_rng(0)
    disk, xx, yy = _disk()
    hdr = {"NAXIS1": 512, "CRPIX1": 256.5, "CRPIX2": 256.5, "CDELT1": 4.8, "CDELT2": 4.8, "RSUN_OBS": 960.0}
    d = np.where(disk, rng.normal(0, 0.3, disk.shape), 0.0)
    d -= 5000.0 * np.exp(-((xx - 150) ** 2 + (yy - 200) ** 2) / (2 * 6**2))     # the decaying kernel
    d += 20.0 * np.exp(-((xx - 170) ** 2 + (yy - 170) ** 2) / (2 * 4**2))      # loops brightening next to it
    from scipy import ndimage
    g = ndimage.gaussian_filter(d.astype(np.float32), 2.0)
    assert g.max() < g.mean() + 5 * g.std()                                       # the 1.2.0 rule: no region
    loc = localize(d, hdr)
    assert loc["method"] == METHOD_ROBUST and set(loc) == LOC_KEYS
    x, y = loc["centroid_arcsec"]
    assert abs(x - (170.5 + 0.5 - 256.5) * 4.8) < 10 and abs(y - (170.5 + 0.5 - 256.5) * 4.8) < 10
    # a frame the first rule locates is located exactly as before
    d2 = np.where(disk, rng.normal(0, 0.3, disk.shape), 0.0) + 50.0 * np.exp(
        -((xx - 300) ** 2 + (yy - 320) ** 2) / (2 * 5**2))
    assert localize(d2, hdr)["method"] == METHOD


# -- the one-off: --fits-root, upgrade of 1.2.0 replays -------------------------------

def test_fits_root_is_searched_after_staging(cfg, tmp_path):
    s = T0
    keep = tmp_path / "keep"
    write_fits(keep / f"{s:%Y%m%d}" / file_name(s, 131), 131, s, size=64)
    loc = FitsLocator(cfg)
    assert loc.existing(s, 131) is None                    # not a root yet
    cfg.fits_roots = [tmp_path / "nowhere", keep]
    assert FitsLocator(cfg).existing(s, 131) == keep / f"{s:%Y%m%d}" / file_name(s, 131)
    write_fits(cfg.staging_dir / f"{s:%Y%m%d}" / file_name(s, 131), 131, s, size=64)
    assert FitsLocator(cfg).existing(s, 131) == cfg.staging_dir / f"{s:%Y%m%d}" / file_name(s, 131)
    assert FitsLocator(cfg).existing(s, 171) is None


def _as_120(cfg, eid, drop_heat=()):
    """Turn a replay into what 1.2.0 left: no per-frame Grad-CAM fields, no
    locations, no geometry; heat frames only where the loop had made one."""
    d = replays.replay_dir(cfg, eid)
    doc = replays.read_replay(cfg, eid)
    for f in doc["frames"]:
        for k in ("alert", "cam_peak", "location", "geometry"):
            f.pop(k, None)
        if f["slot"] in drop_heat:
            (d / "frames" / replays.frame_name(parse_iso(f["slot"])) / "gradcam-heat.webp").unlink()
            f["images"].remove("gradcam-heat")
            doc["missing"]["gradcam-heat"].append(f["slot"])
    for k in ("location", "geometry"):
        doc["peak"].pop(k, None)
    doc.pop("geometry"), doc.pop("threshold")
    doc["missing"].pop("location")
    doc["event"]["location"] = None
    doc["complete"] = not any(doc["missing"].values())
    doc["model_version"] = "1.2.0"
    write_json_atomic(d / "replay.json", doc)
    return doc


def test_one_off_upgrades_120_replays_with_fits_root(cfg, monkeypatch, tmp_path, capsys):
    """Event 1's case: the replay built by 1.2.0 has the alert frames' heat
    only, no location anywhere; its FITS were moved to a keep directory."""
    _env(cfg, monkeypatch)
    ev, slots, rows, peak = _flare_event(cfg, eid=1, days_ago=1)
    expected = {r["slot"]: r["location"] for r in rows if r.get("alert")}
    peaks = {r["slot"]: find_cam(cfg.gradcam_dir, parse_iso(r["slot"]))["peak"] for r in rows if r["status"] == "ok"}
    quiet = [r["slot"] for r in rows if r["status"] == "ok" and not r["alert"]]
    _strip_locations(cfg)
    now = dt.datetime.now(dt.timezone.utc)
    replays.build(cfg, ev, now)
    old = _as_120(cfg, 1, drop_heat=quiet)
    append_event(cfg.events_path, Event(**ev, closed=True))         # as the loop closed it: no location
    marker = replays.frame_path(cfg, 1, slots[0], "171")
    marker.write_bytes(b"kept from the 1.2.0 replay")
    # what is left on the server: maps of the alert frames only (400 d), FITS moved aside
    for s in quiet:
        _drop_maps(cfg, parse_iso(s))
    shutil.rmtree(cfg.tensors_dir)
    shutil.rmtree(cfg.render_dir)
    keep = tmp_path / "keep"
    shutil.move(str(cfg.staging_dir), str(keep))
    cfg.staging_dir.mkdir()

    assert not replays.current(old)
    assert not replays.pending(cfg, now)                                  # the loop leaves it
    assert [e["event_id"] for e in replays.pending(cfg, now, rebuild_incomplete=True)] == [1]
    # without the FITS: the one-off says what is missing
    assert replays.main(["--no-gradcam"]) == 0
    doc = replays.read_replay(cfg, 1)
    assert replays.current(doc) and doc["event"]["location"] is None
    assert sorted(doc["missing"]["gradcam-heat"]) == sorted(quiet)
    assert doc["missing"]["location"] == sorted(expected)
    # with --fits-root: maps for the non-alert frames, the location, the event record filled
    assert replays.main(["--fits-root", str(keep)]) == 0
    out = capsys.readouterr().out
    assert f"{len(quiet)} Grad-CAM maps recomputed" in out and "location yes" in out and "peak RSS" in out
    doc = replays.read_replay(cfg, 1)
    assert replays.current(doc) and doc["model_version"] == replays.__version__
    assert doc["missing"]["gradcam-heat"] == [] and doc["missing"]["location"] == []
    assert "results have no Grad-CAM" not in doc["note"] and "location" not in doc["note"]   # only unstaged slots
    by = {f["slot"]: f for f in doc["frames"]}
    for s, p in peaks.items():
        assert "gradcam-heat" in by[s]["images"] and by[s]["cam_peak"] == pytest.approx(p, abs=1e-3)
    for s, loc in expected.items():
        assert by[s]["location"] == loc
    assert doc["event"]["location"] == doc["peak"]["location"] == expected[iso(peak)]
    assert read_events(cfg.events_path)[-1]["location"] == expected[iso(peak)]
    assert all(by[iso(s)]["geometry"]["size_px"] == 512 for s in slots)
    assert marker.read_bytes() == b"kept from the 1.2.0 replay"            # reused, not re-rendered


def test_one_off_keeps_new_fields_when_the_fits_are_gone_later(cfg):
    """A 1.3.0 replay rebuilt after its FITS and maps are gone keeps its
    cam_peak, location and geometry."""
    ev, slots, rows, peak = _flare_event(cfg, eid=37)
    now = slots[-1] + dt.timedelta(hours=2)
    first = replays.build(cfg, ev, now)
    shutil.rmtree(cfg.staging_dir)
    shutil.rmtree(cfg.gradcam_dir)
    shutil.rmtree(cfg.render_dir)
    _strip_locations(cfg)
    again = replays.build(cfg, ev, now, FitsLocator(cfg), replays.GradcamRecompute(cfg, _ens(cfg),
                                                                                   FitsLocator(cfg)))
    assert [{k: f[k] for k in ("cam_peak", "location", "geometry", "alert")} for f in again["frames"]] == \
        [{k: f[k] for k in ("cam_peak", "location", "geometry", "alert")} for f in first["frames"]]
    assert again["peak"]["location"] == first["peak"]["location"]
    assert again["geometry"] == first["geometry"]


def test_retention_unchanged_with_130_replays(cfg):
    now = dt.datetime(2027, 6, 1, tzinfo=dt.timezone.utc)
    for eid, age in ((1, 366), (2, 364)):
        t = now - dt.timedelta(days=age)
        ev = _event(cfg, t, t + 2 * CADENCE, eid=eid)
        start, end = replays.window(cfg, ev)
        _synthetic_frames(cfg, start, end, peak=t + CADENCE)
        doc = replays.build(cfg, ev, t + dt.timedelta(hours=2))
        assert replays.current(doc)
    shutil.rmtree(cfg.staging_dir)
    shutil.rmtree(cfg.render_dir)
    Retention(cfg).run(now=now)
    assert not replays.replay_dir(cfg, 1).exists()
    assert replays.current(replays.read_replay(cfg, 2))
    assert len(list((replays.replay_dir(cfg, 2) / "frames").rglob("*.webp"))) == 3 * replays.read_replay(cfg, 2)[
        "n_slots"]
    assert cfg.events_path.exists()                                 # events.csv is not pruned


def test_a_robust_location_never_replaces_an_events_location():
    """The M4.0: 14:20 (p 0.987, kernel, first rule), 14:30 (p 0.997, robust,
    post-flare loops). The event keeps 14:20's, as in 1.2.0; an event
    with no location takes the robust one."""
    from flare_nrt.detector import EventTracker
    from flare_nrt.loop import event_location

    kernel = {"centroid_arcsec": [-732.7, -212.9], "method": METHOD}
    loops = {"centroid_arcsec": [-766.4, -318.8], "method": METHOD_ROBUST}
    t = dt.datetime(2026, 7, 7, 14, 18, tzinfo=dt.timezone.utc)

    def run(frames):
        tr = EventTracker(10)
        for i, (p, loc) in enumerate(frames):
            s = t + i * CADENCE
            tr.update(s, p >= 0.8, p, event_location(loc, tr, s))
        return tr.current.location

    assert run([(0.986, kernel), (0.987, kernel), (0.997, loops)]) == kernel
    assert run([(0.9, loops), (0.95, kernel), (0.97, loops)]) == kernel     # first rule wins when it comes
    assert run([(0.9, None), (0.95, loops), (0.97, loops)]) == loops        # event 1: robust only
    assert run([(0.9, loops), (0.85, kernel)]) == loops                     # as 1.2.0: lower p never replaces
    # a robust location at the first alert of a new event, right after another one closed
    tr = EventTracker(10)
    tr.update(t, True, 0.9, event_location(kernel, tr, t))
    s = t + dt.timedelta(minutes=30)
    tr.update(s, True, 0.9, event_location(loops, tr, s))
    assert tr.current.location == loops
