"""Every 1.0.0 endpoint, called without the 1.1.0 parameters, answers as
1.0.0 does. The 1.0.0 server is the tagged code itself (``git archive
v1.0.0``, imported as ``flare_nrt_v100``), run on a copy of the same data
directory.

The only differences, all asked for in 1.1.0:

- ``/api/image/1600.png?kind=img`` is rescaled (1600 Å was washed out);
- slot-addressed images are ``Cache-Control: immutable`` (was max-age 3600
  for AIA images, 86400 for Grad-CAM); a difference whose result row does
  not exist yet keeps max-age 3600;
- ``/api/events`` rows gain ``replay``;
- ``model_version`` says 1.1.0."""

import datetime as dt
import io
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from conftest import write_slot
from fastapi.testclient import TestClient

from flare_nrt.grid import CADENCE, CHANNELS, iso

ROOT = Path(__file__).resolve().parents[1]
VOLATILE = {"server_time", "seconds_since_update", "input_age_s", "model_version", "disk_free_gb"}
IMMUTABLE = "public, max-age=31536000, immutable"


@pytest.fixture(scope="module")
def v100(tmp_path_factory):
    ok = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "-q", "v1.0.0^{commit}"],
                        capture_output=True).returncode == 0
    if not ok:
        msg = "tag v1.0.0 not in this checkout (CI: actions/checkout with fetch-depth: 0)"
        if os.environ.get("CI"):
            pytest.fail(msg)
        pytest.skip(msg)
    out = tmp_path_factory.mktemp("v100")
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", "v1.0.0", "flare_nrt"], capture_output=True,
                         check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(out, filter="data")
    (out / "flare_nrt").rename(out / "flare_nrt_v100")
    sys.path.insert(0, str(out))
    try:
        import flare_nrt_v100.server as old
    finally:
        sys.path.remove(str(out))
    assert old.__dict__["__version__"] == "1.0.0"
    return old


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE and k != "replay"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def test_v100_endpoints_unchanged(cfg, v100, monkeypatch, tmp_path):
    import flare_nrt.server as new
    from flare_nrt.loop import Processor
    from flare_nrt.model import Ensemble
    from flare_nrt.source import ReplaySource
    from flare_nrt.store import write_json_atomic

    for mod in (new, v100):                      # no network: SWPC's list is empty
        monkeypatch.setattr(mod, "fetch_swpc", lambda url, timeout=30: [])
    now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    t0 = now - dt.timedelta(minutes=now.minute % 2) - 15 * CADENCE
    slots = [t0 + i * CADENCE for i in range(16)]
    for i, s in enumerate(slots):
        write_slot(cfg.staging_dir, s, size=1024, flare=400.0 if 6 <= i <= 8 else 0.0)
    proc = Processor(cfg, Ensemble.load(cfg.weights, cfg.weights_sha256), ReplaySource(cfg.staging_dir), live=True)
    rows = []
    for i, s in enumerate(slots):            # alerts on slots 6–8 only: one event, closed by slot 15
        cfg.threshold = 0.0 if 6 <= i <= 8 else 1.1
        rows.append(proc.process(s, s + dt.timedelta(minutes=1)))
    write_json_atomic(cfg.state_dir / "heartbeat.json", {"heartbeat": iso(now)})
    ok = [r["slot"] for r in rows if r["status"] == "ok"]
    alerts = [r["slot"] for r in rows if r.get("alert")]
    assert ok and alerts and len(alerts) < len(ok)
    evs = TestClient(new.create_app(cfg)).get("/api/events").json()["events"]
    assert len(evs) == 1 and evs[0].get("closed", True)          # one event, closed (in events.csv)

    old_cfg = type(cfg)()
    old_cfg.__dict__.update(cfg.__dict__)
    old_cfg.data_dir = tmp_path / "var_v100"
    shutil.copytree(cfg.data_dir, old_cfg.data_dir)
    a, b = TestClient(v100.create_app(old_cfg)), TestClient(new.create_app(cfg))

    first, last = iso(slots[0]), iso(slots[-1])
    win = f"start={first}&end={last}"
    json_urls = ["/api/status", "/api/events", "/api/events?days=1", f"/api/events?{win}", "/api/series",
                 "/api/series?days=1", f"/api/series?{win}", "/api/frames", "/api/frames?hours=1",
                 f"/api/frames?{win}", "/api/results", "/api/results?hours=2", "/api/channels", "/api/health",
                 "/api/range", f"/api/frame/{ok[0]}", f"/api/frame/{first}", "/api/flares?days=1", f"/api/flares?{win}",
                 "/api/series?days=41", f"/api/series?start={last}&end={first}", "/api/events?days=0",
                 "/api/frame/notatime"]
    for u in json_urls:
        ra, rb = a.get(u), b.get(u)
        assert ra.status_code == rb.status_code, u
        assert ra.headers["content-type"] == rb.headers["content-type"], u
        assert _strip(ra.json()) == _strip(rb.json()), u
    assert all("replay" in e for e in b.get("/api/events").json()["events"])
    assert a.get("/viewer").text == b.get("/viewer").text

    img_urls = []
    for w in CHANNELS:
        for kind in ("img", "diff"):
            img_urls += [f"/api/image/{w}.png?kind={kind}", f"/api/image/{w}.png?kind={kind}&slot={ok[0]}",
                         f"/api/image/{w}.png?kind={kind}&slot={first}"]
        img_urls.append(f"/api/image/{w}.png")
    img_urls += ["/api/gradcam/latest.png", f"/api/gradcam/{ok[0]}.png", f"/api/gradcam/{alerts[0]}.png",
                 f"/api/gradcam/{first}.png", "/api/gradcam/notatime.png", "/api/image/94.png",
                 "/api/image/131.png?kind=raw", "/api/image/131.png?slot=yesterday",
                 "/api/image/131.png?slot=2020-01-01T00:00:00Z"]
    for u in img_urls:
        ra, rb = a.get(u), b.get(u)
        assert ra.status_code == rb.status_code, u
        assert ra.headers["content-type"] == rb.headers["content-type"], u
        if ra.status_code != 200:
            continue
        assert ra.headers["X-Observation-Time"] == rb.headers["X-Observation-Time"], u
        if "/1600.png" in u and "diff" not in u:
            assert ra.content != rb.content, u            # 1.1.0: 1600 Å rescaled
        else:
            assert ra.content == rb.content, u
        if "slot=" in u or ("/api/gradcam/" in u and "latest" not in u):
            row_exists = "image" not in u or "diff" not in u or f"slot={ok[0]}" in u
            assert rb.headers["cache-control"] == (IMMUTABLE if row_exists else "max-age=3600"), u
        else:
            assert ra.headers["cache-control"] == rb.headers["cache-control"] == "max-age=60", u
